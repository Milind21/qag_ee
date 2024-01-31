import json
import os
import torch
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support

parser = argparse.ArgumentParser(description='train')
parser.add_argument('-t', '--task')
parser.add_argument('-m', '--model')
args = parser.parse_args()

# Initialize the T5 tokenizer
model_name = args.model
tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=512)

task = args.task

# find who words from KAIROS
with open('../../dataset/event_role_KAIROS.json', 'r') as f:
    kairos_role = json.load(f)

total_role_types={}
for key,value in kairos_role.items():
    for i in range(len(value['roles'])):
        if(value['roles'][i] not in total_role_types.keys()):
            total_role_types[value['roles'][i]]=set()
        total_role_types[value['roles'][i]].update(value['role_types'][i])

question_map = {}
for key, value in total_role_types.items():
    if ("per" in value or "org" in value):
        if("Who" not in question_map.keys()):
            question_map["Who"]=[]
        question_map["Who"].append(key)
    elif ("per" in value or "org" in value):
        if("Whom" not in question_map.keys()):
            question_map["Whom"]=[]
        question_map["Whom"].append(key)
    elif ("loc" in value or "fac" in value):
        if("Where" not in question_map.keys()):
            question_map["Where"]=[]
        question_map["Where"].append(key)
    elif ("mhi" in value):
        if("Why" not in question_map.keys()):
            question_map["Why"]=[]
        question_map["Why"].append(key)
    elif ("mhi" in value):
        if("How" not in question_map.keys()):
            question_map["How"]=[]
        question_map["How"].append(key)
    else:
        if("What" not in question_map.keys()):
            question_map["What"]=[]
        question_map["What"].append(key)

def find_wh(role):
    wh_list=[]
    for k,v in question_map.items():
        if(role in v):
            wh_list.append(k) 
    return wh_list

def compose_prompt(doc, question):
    document = '\n'.join(['Document:', doc])
    question = ' '.join(['Question:', question])
    answer = 'Answer: '
    qa_pair = '\n'.join([question, answer])
    message = '\n\n'.join([document, qa_pair])
    return message

def convert_file(split):
    if os.path.exists(f'../../dataset/converted_{split}.json'):
        return
    with open(f'../../dataset/{split}.json', 'r') as f:
        data = json.load(f)

    converted_data = []
    skipped = []
    for doc_id, content in data.items():
        for event in content['events']:
            task_list = task.split('-')

            if 'manual' in task_list:
                for pair in event['manual']:
                    converted_data.append({
                        'input': compose_prompt(content['text'], pair['question']),
                        'target': pair['answer']
                    })

            if 'kairos_argument' in task_list:
                for entity in event['kairos_argument']:
                    role = entity['role']
                    wh = find_wh(role)
                    for wh_word in wh:
                        question = wh_word + ' ' + event['trigger'] + '?'
                        converted_data.append({
                            'input': compose_prompt(content['text'], question),
                            'target': entity['text']
                        })

    with open(f'../../dataset/converted_{split}_{task}_0.1.json', 'w') as f:
        json.dump(converted_data[:int(0.1*len(converted_data))], f, indent=2)
    print(f'{split} done')

convert_file('train')
convert_file('dev')
convert_file('test')

# Load data from JSON files
train_dataset = load_dataset('json', data_files=f'../../dataset/converted_train_{task}_0.1.json')['train']
val_dataset = load_dataset('json', data_files=f'../../dataset/converted_dev_{task}_0.1.json')['train']
test_dataset = load_dataset('json', data_files=f'../../dataset/converted_test_{task}_0.1.json')['train']

def tokenize_function(batch):
    # Tokenize input and target in batches
    inputs = tokenizer(batch['input'], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    targets = tokenizer(batch['target'], padding='max_length', truncation=True, max_length=512, return_tensors="pt")

    # Shift the target sequences to the right
    decoder_input_ids = targets["input_ids"].clone()
    decoder_input_ids = torch.cat([torch.full_like(decoder_input_ids[:, :1], tokenizer.pad_token_id), decoder_input_ids[:, :-1]], dim=-1)

    # Add decoder_input_ids and labels to the returned dictionary
    inputs["decoder_input_ids"] = decoder_input_ids
    inputs["labels"] = targets["input_ids"]
    
    return inputs

# Tokenize datasets
train_tokenized = train_dataset.map(tokenize_function, batched=True)
val_tokenized = val_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)

# Initialize T5 model
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir='../../logs/',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=938,
    output_dir='../../results/',
    push_to_hub=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained(f"../../results/{model_name}_finetuned_model_{task}_0.1")
tokenizer.save_pretrained(f"../../results/{model_name}_finetuned_model_{task}_0.1")

