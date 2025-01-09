

from llm_utils import first_sequence_analyze
from llm_utils import sequence_analyze
from llm_utils import sequence_analyze_nobodypart
from llm_utils import sequence_analyze_nobodypart_jsontool

import json
import os

# Paths and directories
texts_path = "/extra/xielab0/haoyum3/Ask-Anything/videochat_finetue/test/text.txt"
save_dir = "/extra/xielab0/haoyum3/Ask-Anything/videochat_finetue/test/prompt"

os.makedirs(save_dir, exist_ok=True)

# Load actions
actions = []
with open(texts_path, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        if line: 
            actions.append(line)
# Process each action
print('total action: ', len(actions))
for count, action in enumerate(actions):
    print(count)
    # Analyze the action steps
    sequence_analyze_result = sequence_analyze_nobodypart(action)
    print(sequence_analyze_result)
    output = sequence_analyze_nobodypart_jsontool(action, sequence_analyze_result)
    print(output)
    # Loop through the steps in the output
    for i, step in enumerate(output):
        low_level = first_sequence_analyze(step['prompt'])
        temp = {
            'prompt': low_level,
            'original prompt': action
        }
        # Save each step as a separate JSON file
        json_file_path = os.path.join(save_dir, f'{count}_{i}.json')
        with open(json_file_path, "w", encoding='utf-8') as file:
            json.dump(temp, file, indent=4, ensure_ascii=False)
            print(f'Saved {count}_{i}.json')
            print(json.dumps(temp, indent=4, ensure_ascii=False))




