import json
import openai
import re



openai.api_key = API_KEY
openai.organization = ORG
model_id = 'gpt-4-0613'

with open('test2.json', 'r') as f:
        data = json.load(f)
data_copy=data
for key,value in data.items():
        doc = value["text"]
        print(key)
        for event in value["events"]:
                trigger = event["trigger"]
                event_id = event["event_id"]
                doc = re.sub(rf"\b{trigger}\b", f"<b>{trigger}</b>", doc, flags=re.IGNORECASE)
                for qa_pair in event["manual"]:
                        question = qa_pair["question"]
                        answer = qa_pair['answer']
                        prompt = 'Context: {passage}\nTrigger: {trigger}\n Question: {question}\n Answer: '.format(passage=doc, trigger=trigger,question=question)
                        example= 'A settlement has been reached in a $1-million lawsuit filed by a taxi driver accusing police of negligence after he got caught up in the August 2016 take-down of ISIS-sympathizer Aaron Driver.READ MORE: FBI agent whose tip thwarted 2016 ISIS <b>attack</b> in Ontario says he was glad to helpTerry Duffield was injured when Driver detonated a homemade explosive in the back of his cab in August 2016.“I have to be very careful because there is an agreement to not disclose any of the terms of the settlement,” Duffield’s lawyer Kevin Egan told 980 CFPL.“The statement of claim, I guess, speaks for itself in regard to what we alleged.”WATCH: Ontario taxi driver files $1M lawsuit against police2:18 Ontario taxi driver files $1M lawsuit against police Ontario taxi driver files $1M lawsuit against policeThat statement of claim, which Global News obtained a copy of in late March 2018, said police had more than enough time to intervene before Driver got into Duffield’s taxi. The Attorney General of Canada, the Ontario government, Strathroy-Caradoc Police Service and London Police Service were named as defendants.Story continues below advertisementOn the morning of Aug. 10, 2016, U.S. authorities notified the RCMP they had detected a so-called martyrdom video in which a Canadian man said he was about to conduct an attack.The RCMP identified the man in the video as Driver and a tactical team surrounded his house in Strathroy.At 3:45 p.m., Driver called for a cab to take him to Citi Plaza in London. The claim alleged that despite the police presence, Duffield was not stopped from pulling into Driver’s driveway. Driver then came out of the house and got into the back seat of the cab.“When the SWAT team approached the vehicle, [Duffield] turned to Mr. Driver and said, ‘I think they’re here to talk to you’ and he leaned over to get his cigarettes, it’s a bench seat in the front of the taxicab, as he put his head down below the bench seat, the bomb went off.”Story continues below advertisementThe inside of the cab where Aaron Driver detonated an explosive device on Aug. 10, 2016. Handout / RCMPEgan said Duffield had a preexisting back injury and the bomb blast triggered recurring pain. He also noted that his client was psychologically impacted by the event and is no longer able to work as a taxi driver.“He did try it. Got in a vehicle, turned the key on and started to shake and sweat and got out of the vehicle and vomited,” said Egan. Tweet This“He was so traumatized by the event. He realized that any time any potential passenger was approaching the vehicle with a package he would be hyper-vigilant about that and just couldn’t handle it emotionally.”Details of the settlement will not be made public, but Egan noted that no amount of money can properly compensate someone for physical or psychological injuries, but “is the best we can do in the circumstance.” He also noted that, while he was unwilling to disclose too much of Duffield’s personal health, he has received some counselling and is “coping better now than he was then.”Story continues below advertisement– with files from Stewart Bell and Andrew Russell.\nTrigger: Attack\n Question: Who attacked?\nAnswer:ISIS'
                        messages = [
                        {"role": "system", "content": f"you help provide one answer of length not more than {len(answer)} to the question based on context"},
                        {"role": "user", "content": "You are an assistant that reads through a passage and provides the answer based on passage and trigger. The bolded word is the event trigger. Answers MUST be direct quotes from the passage.\n Make sure to generate the answers based on the context, the trigger and corresponding question.In a new line, output the answer. Do not output anything else other than the answer in this last line."},
                        {"role": "user", "content": prompt}]
                        response = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo-0613',
                        messages = messages)
                        pred = response['choices'][0]['message']['content']
                        for k,v in data_copy.items():
                                if(key==k):
                                        for e in v["events"]:
                                                e_id=e["event_id"]
                                                for qap in e['manual']:
                                                        if(qap["question"]==question):
                                                                qap['answer']=pred
                                                                qap['gold_ans']=answer


pretty_json = json.dumps(data_copy, indent=4)
output_file_path = 'ans_id_test_pretty_nodemo.json'
with open(output_file_path, 'w') as output_file:
        output_file.write(pretty_json)
pretty_json = json.dumps(data_copy)
output_file_path = 'ans_id_test_nodemo.json'
with open(output_file_path, 'w') as output_file:
        output_file.write(pretty_json)
print(f"Data saved to {output_file_path}")