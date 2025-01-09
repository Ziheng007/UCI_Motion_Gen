from openai import OpenAI
import json
import re
client = OpenAI()
model = 'gpt-4o'

def get_parts(sentence):
    body_parts = {}
    lines = sentence.strip().split('\n')
    for line in lines:
        if ':' in line:
            part, description = line.split(':', 1)
            part = part.strip()
            description = description.strip()
            body_parts[part] = description
    return body_parts

def generate_suggestion(txt_prompt,caption_prompt,file='output1.json'):
    
    template1 = """
        Your task is to generate different body parts motion according to a Motion Description. The body parts are right arm, left arm, right leg and left leg.
        You only need to output motion of different body parts without any explanation. If some body parts are not mentioned in the Motion Description, you need to deduce those body parts by the Motion Description. Ensure that the motion described is rational and appropriate for the specified body part, aligning with the original motion description. In the final motion description, the body parts must be the subject of the sentence.
        The input format is:
            Motion Description: [Insert text here]
        The output format is:
            Right arm: [the final right arm motion description including right arm as the subject.]
            Left arm: [the final left arm motion description including left arm as the subject.]
            Right leg: [the final right leg description including right leg as the subject.]
            Left leg: [the final left leg description including left leg as the subject.]
        """
    text_prompt = 'Motion Description: {}'.format(txt_prompt)
    completion1 = client.chat.completions.create(
        model= model,
        messages=[
            {"role": "system", "content": template1},
            {
                "role": "user",
                "content": text_prompt
            }
        ]
        )
    text_prompt_body_part = get_parts(completion1.choices[0].message.content)
    text_prompt_body_part['motion'] = txt_prompt
    
    caption = "Motion Description: {}".format(caption_prompt)
    completion2 = client.chat.completions.create(
    model= model,
    messages=[
        {"role": "system", "content": template1},
        {
            "role": "user",
            "content": caption
        }
        ]
        )
    caption_body_part = get_parts(completion2.choices[0].message.content)
    caption_body_part['motion'] = caption_prompt

    #print(text_prompt_body_part)
    #print(caption_body_part)
    template2 = """
    You have two groups of motion descriptions stored in dictionaries. Each dictionary contains the following keys: ‘motion’, ‘Right arm’, ‘Left arm’, ‘Right leg’, and ‘Left leg’. The ‘motion’ key describes a person’s overall movement, while the other keys specify the movement of each body part in that motion.

    Your task:
	    Compare the ‘motion’ in two motion descriptions: ‘motion description1’ (the standard motion) and ‘motion description2’ (the observed motion).
	    Determine if the ‘motion’ in ‘motion description2’ approximately matches the ‘motion’ in ‘motion description1’.

    Guidelines:
        Only use 'motion' to do comparision.
	    If there is a mismatch:
	    For upper body mismatches: use the ‘Right arm’ and ‘Left arm’ motions in ‘motion description1’ to help you understand the upper body motion and then generate an upper body motion instruction. 
	    For lower body mismatches: use the ‘Right leg’ and ‘Left leg’ motions in ‘motion description1’ to help you understand the upper body motion and then generate a lower body motion instruction. The lower bdoy motion must be cohesive and naturely. 
        The body part motion is just for reference and help you better understand. Don't directly use the body part motion to generate output. Please start you answer from 'motion' in the motion description1.
	    If the 'motion' of two motion description are approximately same, describing a similar motion, both upper body and lower body output None. You don't need to pay attention to the detail of two motion. We only need two motions are approximately same.
       Approximately same: if two specific and corresponding body part do a same action (raise, jump, ...), they are approximately same. You do not need to pay attention to the height of arm raised and how far a peson jump. This is the detail of one action. You do not need to pay attention to the detail of action.
       For example: the first motion that the man is walking clockwise in a circle while holding something up to his ear with his left arm. The second motion that a man with his left arm raised walk clockwise. The person in two motions both walk clockwise and raise their left arm. So these two motions are approximately same.

    Output Requirements:
	    For mismatched motions, output only the motion instruction for the person’s upper body or lower body without explanation. Use the person as the subject, and the ‘Right arm’ and ‘Left arm’ descriptions as content for upper body; similarly, use ‘Right leg’ and ‘Left leg’ for lower body.
	    For matched motions, simply output “None” for the respective body part.

    Input Format:
	    Motion Description1: [Insert text here]
	    Motion Description2: [Insert text here]

    Output Format:
	    Upper body: [Insert motion or “None”]
	    Lower body: [Insert motion or “None”]
    """
    ask = f"""
    Motion Description1:{text_prompt_body_part}
    Motion Description2:{caption_body_part}
    """
    completion3 = client.chat.completions.create(
    model= model,
    messages=[
        {"role": "system", "content": template2},
        {
            "role": "user",
            "content": ask
        }
        ]
        )
    answer = completion3.choices[0].message.content
    #print(file)
    #print("text prompt")
    #print(text_prompt_body_part)
    #print("caption")
    #print(caption_body_part)
    #print("difference")
    #print('\n')
    #print(answer)
    with open(file, 'w') as output_file:
        json_object = {
            "Original Body Part": text_prompt_body_part,
            "Caption Body Part": caption_body_part,
            "Edit Instruction":get_parts(answer)
        }
        output_file.write(json.dumps(json_object, indent=4) + "\n")
        
def sort_key(filename):
    match = re.search(r'(\d+)', filename) 
    return int(match.group(1)) if match else float('inf')

if __name__ == '__main__':
    import os
    import argparse
    print("############################################")
    print("LLM reasoning.................")
    parser = argparse.ArgumentParser(description="caption")
    parser.add_argument("--caption_dir", type=str,required=True)
    parser.add_argument("--prompt_dir", type=str,required=True)
    parser.add_argument("--instruction_dir", type=str,required=True)
    args = parser.parse_args()
    caption_dir = args.caption_dir
    prompt_dir = args.prompt_dir
    instrution_dir = args.instruction_dir
    os.makedirs(instrution_dir,exist_ok=True)
    instruction_name = []
    captions = []
    caption_list = sorted(os.listdir(caption_dir),key=sort_key)
    for caption_name in caption_list:
        print(caption_name)
        instruction_name.append(caption_name.split('.')[0].strip())
        caption_path = os.path.join(caption_dir,caption_name)
        with open(caption_path,'r') as f:
            for line in f.readlines():
                caption = line.strip()
                break
            captions.append(caption)
    text_prompts = []
    prompt_list =  sorted(os.listdir(prompt_dir),key=sort_key)
    for prompt_name in instruction_name:
        print(prompt_name)
        prompt_path = os.path.join(prompt_dir,prompt_name+'.json')
        with open(prompt_path, "r") as file:
            prompt = json.load(file)['prompt']
            text_prompts.append(prompt)
        
    for i, (caption, text_prompt) in enumerate(zip(captions,text_prompts)):
        print('Example {}:\n caption: {} \n prompt:{}\n'.format(i,caption,text_prompt))
        save_path = os.path.join(instrution_dir,'{}.json'.format(instruction_name[i]) )
        generate_suggestion(text_prompt,caption,save_path)