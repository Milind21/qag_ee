import json
import openai
import time
import tiktoken
import re 

openai.api_key = API_KEY
openai.organization = ORG
model_id = 'gpt-4-0613'
op_dict={}
tokenizer = tiktoken.get_encoding("cl100k_base")
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
num_token=0
def call(message):
    global num_token
    num_token+=len(tokenizer.encode(message))
    while True:
        try:
            response = openai.ChatCompletion.create(
                            model=model_id,
                            messages=message,
                            max_tokens=100,
                            temperature=0.2
                            )
            break
        except Exception as e:
            time.sleep(2)
            print('Error!!!', str(e))
            print(len(tokenizer.encode(message[0]['content'])))

    prediction = response['choices'][0]['message']['content']

    return prediction

def ask_questions(data):
    global op_dict
    for key,value in data.items():
        doc = value["text"]
        print(key)
        op_dict[key]={"events":[]}           
        for event in value["events"]:
            temp_dict={}
            trigger = event['trigger']
            print(trigger)
            doc = re.sub(rf"\b{trigger}\b", f"<b>{trigger}</b>", doc, flags=re.IGNORECASE)
            event_id = event['event_id']
            prompt = 'Here is the passage: {passage}\nThe trigger is:\n{trigger}'.format(passage=doc, trigger=trigger)
            example= 'A settlement has been reached in a $1-million lawsuit filed by a taxi driver accusing police of negligence after he got caught up in the August 2016 take-down of ISIS-sympathizer Aaron Driver.READ MORE: FBI agent whose tip thwarted 2016 ISIS <b>attack</b> in Ontario says he was glad to helpTerry Duffield was injured when Driver detonated a homemade explosive in the back of his cab in August 2016.“I have to be very careful because there is an agreement to not disclose any of the terms of the settlement,” Duffield’s lawyer Kevin Egan told 980 CFPL.“The statement of claim, I guess, speaks for itself in regard to what we alleged.”WATCH: Ontario taxi driver files $1M lawsuit against police2:18 Ontario taxi driver files $1M lawsuit against police Ontario taxi driver files $1M lawsuit against policeThat statement of claim, which Global News obtained a copy of in late March 2018, said police had more than enough time to intervene before Driver got into Duffield’s taxi. The Attorney General of Canada, the Ontario government, Strathroy-Caradoc Police Service and London Police Service were named as defendants.Story continues below advertisementOn the morning of Aug. 10, 2016, U.S. authorities notified the RCMP they had detected a so-called martyrdom video in which a Canadian man said he was about to conduct an attack.The RCMP identified the man in the video as Driver and a tactical team surrounded his house in Strathroy.At 3:45 p.m., Driver called for a cab to take him to Citi Plaza in London. The claim alleged that despite the police presence, Duffield was not stopped from pulling into Driver’s driveway. Driver then came out of the house and got into the back seat of the cab.“When the SWAT team approached the vehicle, [Duffield] turned to Mr. Driver and said, ‘I think they’re here to talk to you’ and he leaned over to get his cigarettes, it’s a bench seat in the front of the taxicab, as he put his head down below the bench seat, the bomb went off.”Story continues below advertisementThe inside of the cab where Aaron Driver detonated an explosive device on Aug. 10, 2016. Handout / RCMPEgan said Duffield had a preexisting back injury and the bomb blast triggered recurring pain. He also noted that his client was psychologically impacted by the event and is no longer able to work as a taxi driver.“He did try it. Got in a vehicle, turned the key on and started to shake and sweat and got out of the vehicle and vomited,” said Egan. Tweet This“He was so traumatized by the event. He realized that any time any potential passenger was approaching the vehicle with a package he would be hyper-vigilant about that and just couldn’t handle it emotionally.”Details of the settlement will not be made public, but Egan noted that no amount of money can properly compensate someone for physical or psychological injuries, but “is the best we can do in the circumstance.” He also noted that, while he was unwilling to disclose too much of Duffield’s personal health, he has received some counseling and is “coping better now than he was then.”Story continues below advertisement– with files from Stewart Bell and Andrew Russell.\nAttack\nA: ISIS\nQ: Who attacked?\nA: 2016\nQ: When did the attack happen?'
            messages = [
            {"role": "system", "content": "you help provide a maximum of 5 answers and questions per trigger to annotate passages"},
            {"role": "user", "content": "You are an assistant that reads through a passage and provides all possible question and answer pairs to the bolded word. The bolded word is the event trigger, and the questions will help ascertain facts about the event. The questions must be in this template:\nwh* verb subject trigger object1 preposition object2\nWh* is a question word that starts with wh (i.e. who, what, when, where). The subject performs the action. The object is the person, place, or thing being acted upon by the subject\'s verb. A preposition is a word or group of words used before a noun, pronoun, or noun phrase to show direction, time, place, location, spatial relationships, or to introduce an object. Answers MUST be direct quotes from the passage. Do not ask any inference questions.\n Make sure to generate the answers based on trigger and then the corresponding question.\n You have to consistently follow the output format"},
            {"role": "user", "content": "This is a demo of what I want: {demo}".format(demo=example)},
            {"role": "user", "content": prompt}
            ]
            response = openai.ChatCompletion.create(
            model='gpt-4-0613',
            messages = messages)
            pred = response['choices'][0]['message']['content']
            temp_dict[event_id]=pred
            op_dict[key]["events"].append(temp_dict)

if __name__ == '__main__':
    with open('test2.json', 'r') as f:
        data = json.load(f)
    ask_questions(data)
    pretty_json = json.dumps(op_dict, indent=4)
    output_file_path = 'GPT4_test_aq.json'
    with open(output_file_path, 'w') as output_file:
        output_file.write(pretty_json)
    print(f"Data saved to {output_file_path}")

