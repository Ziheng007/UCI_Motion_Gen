from llm_utils import analyze_fine_moton_control_txt_nosequence

from llm_utils  import first_sequence_analyze


import json
import os
texts_path ="./data/complex_1st/texts.txt"
save_dir = "./data/complex_1st/prompt"
os.makedirs(save_dir,exist_ok=True)
actions = []

with open(texts_path,'r') as f:
    for line in f.readlines():
        line = line.split('\n')[0].strip()
        actions.append(line)
        
for i, action in enumerate(actions):
    print(i)
    low_level_prompt = first_sequence_analyze(action)
    temp = dict()
    temp['prompt'] = low_level_prompt
    temp['original prompt'] = action
        
    with open(os.path.join(save_dir,'{}.json'.format(i)), "w") as file:
        json.dump(temp, file, indent=4, ensure_ascii=False)