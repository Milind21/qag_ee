import json
ip_path="GPT4_test_aq.json"
with open(ip_path, 'r') as json_file:
    gold_data = json.load(json_file)
for key,value in gold_data.items():
    for event in value["events"]:
        for k,v in event.items():
            qa_pairs = v.split('\n')
            qa_list = []
            for i in range(0,len(qa_pairs),2):
                try:
                    answer  = qa_pairs[i]
                    question  = qa_pairs[i+1]
                    question = question[3:]  # Remove 'Q: ' prefix
                    answer = answer[3:]      # Remove 'A: ' prefix
                    qa_dict = {'answer': answer,'question': question}
                    qa_list.append(qa_dict)
                except Exception as e:
                    continue
            event[k]=qa_list
            
pretty_json = json.dumps(gold_data, indent=4)
output_file_path = 'GPT4_aq_parsed.json'
with open(output_file_path, 'w') as output_file:
    output_file.write(pretty_json)
print(f"Data saved to {output_file_path}")
