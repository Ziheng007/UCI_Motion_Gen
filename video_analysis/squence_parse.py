

from llm_utils import first_sequence_analyze
from llm_utils import sequence_analyze
from llm_utils import sequence_analyze_nobodypart
from llm_utils import sequence_analyze_nobodypart_jsontool

import json
import os

# Paths and directories
texts_path = "/extra/xielab0/haoyum3/Ask-Anything/videochat_finetue/data/sequential_1st/text.txt"
save_dir = "/extra/xielab0/haoyum3/Ask-Anything/videochat_finetue/data/sequential_1st/prompt"

os.makedirs(save_dir, exist_ok=True)

# Load actions
actions = []
with open(texts_path, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        if line:  # 确保非空行
            actions.append(line)

# Process each action
for count, action in enumerate(actions):
    # Analyze the action steps
    low_level = first_sequence_analyze(action)
    sequence_analyze_result = sequence_analyze_nobodypart(low_level)
    output = sequence_analyze_nobodypart_jsontool(low_level, sequence_analyze_result)

    # Loop through the steps in the output
    for i, step in enumerate(output):
        temp = {
            'prompt': step['prompt'],
            'original prompt': low_level
        }

        # Save each step as a separate JSON file
        json_file_path = os.path.join(save_dir, f'{count}_{i}.json')
        with open(json_file_path, "w", encoding='utf-8') as file:
            json.dump(temp, file, indent=4, ensure_ascii=False)
            print(f'Saved {count}_{i}.json')
            print(json.dumps(temp, indent=4, ensure_ascii=False))

# step1: The man spreads his legs wide and simultaneously engages his core while performing a rowing motion with his arms, pulling them back as if gripping an oar.

# step2: The man's legs then push off the ground, generating upward momentum, while his arms swing forward to aid in the leap, allowing him to clear the obstacle with agility typical of a parkour expert.

# output example AAA： 
# [
#     {
#         "prompt": "The man spreads his legs wide and simultaneously engages his core while performing a rowing motion with his arms, pulling them back as if gripping an oar.",
#         "original prompt": "The person performs a rowing motion with their legs spread wide,and then Leap over an obstacle with the agility of a parkour expert."
#     },
#     {
#         "prompt": "The man's legs then push off the ground, generating upward momentum, while his arms swing forward to aid in the leap, allowing him to clear the obstacle with agility typical of a parkour expert.",
#         "original prompt": "The person performs a rowing motion with their legs spread wide,and then Leap over an obstacle with the agility of a parkour expert."
#     }
# ...
#     {
#         "prompt": "...",
#         "original prompt": "... expert."
#     }
# ]
# 我需要存储在save_dir = "./data/sequential_1st/prompt"目录下的json文件中。
# 以 output example AAA： 为例子，我需要存储n个json文件，分别为count_1.json和count_2.json...count_n.json n为output example AAA 中的步骤数目。


