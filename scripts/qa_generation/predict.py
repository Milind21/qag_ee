import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, EvalPrediction
from datasets import load_dataset
import argparse
import json
import os
import re
import string
import sys
from collections import Counter
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='train')
parser.add_argument('-x', '--train_task')
parser.add_argument('-y', '--test_task')
parser.add_argument('-m', '--model')
args = parser.parse_args()

train_task = args.train_task
test_task = args.test_task

model_name = args.model

test_dataset = load_dataset('json', data_files=f'../../dataset/converted_test_{test_task}.json')['train']
print(test_dataset)

# Load the saved model and tokenizer
saved_model_path = f"../../results/{model_name}_finetuned_model_{train_task}"
loaded_model = T5ForConditionalGeneration.from_pretrained(saved_model_path)
loaded_tokenizer = T5Tokenizer.from_pretrained(saved_model_path, max_model_length=512)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='../../logs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=938,
    output_dir='../../results',
    push_to_hub=False,
)

tokenizer = T5Tokenizer.from_pretrained(model_name, max_model_length=512)
def tokenize_function(batch):
    inputs = tokenizer(batch['input'], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    targets = tokenizer(batch['target'], padding='max_length', truncation=True, max_length=512, return_tensors="pt")

    decoder_input_ids = targets["input_ids"].clone()
    decoder_input_ids = torch.cat([torch.full_like(decoder_input_ids[:, :1], tokenizer.pad_token_id), decoder_input_ids[:, :-1]], dim=-1)
    inputs["decoder_input_ids"] = decoder_input_ids
    inputs["labels"] = targets["input_ids"]
    return inputs

test_tokenized = test_dataset.map(tokenize_function, batched=True)

trainer = Trainer(
    model=loaded_model,
    args=training_args,
    tokenizer=loaded_tokenizer
)

prediction = trainer.predict(test_tokenized).predictions[0]
predicted_token_ids = np.argmax(prediction, axis=-1)  # Shape should now be (83, 512)

# Decode token IDs to text
decoded_answers = [loaded_tokenizer.decode(token_ids, skip_special_tokens=True) for token_ids in predicted_token_ids]

# Combine _id with the decoded answers
all_data = []
for ori, pred in zip(test_dataset, decoded_answers):
    ori['pred'] = pred
    all_data.append(ori)

# Save to a JSON file
with open(f"../../results/results_{train_task}_{test_task}.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2)

def clean_mention(text):
    """
    Clean up a mention by removing ‘a’, ‘an’, ‘the’ prefixes.
    """
    prefixes = ['the', 'The', 'an', 'An', 'a ', 'A ']
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def update_answer(metrics, prediction, gold):
    f1, prec, recall = f1_score(prediction, gold)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall

metrics = {'f1': 0, 'prec': 0, 'recall': 0}

for d in all_data:
    pred_ans = d['pred']
    gold_ans = d['target']
    update_answer(metrics, pred_ans, gold_ans)

print(f"F1: {metrics['f1']/len(all_data)}")
print(f"Precision: {metrics['prec']/len(all_data)}")
print(f"Recall: {metrics['recall']/len(all_data)}")

