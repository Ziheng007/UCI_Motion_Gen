from llm_utils import generate_sequence_explanation_prompt
from llm_utils import generate_fine_motion_control_prompt
from llm_utils import generate_sequence_explanation_prompt_json

from .llm_config  import llm

def sequence_analyze(action):
    sequence_explanation_prompt = generate_sequence_explanation_prompt(action)
    sequence_explanation = llm(sequence_explanation_prompt, stop=["<SEQUENCEEND>"]).split("<SEQUENCEEND>")[0].strip()
    print(sequence_explanation)

    # Use the updated format to generate JSON-like output for each body part
    sequence_explanation_prompt2 = generate_sequence_explanation_prompt_json(action, sequence_explanation)
    # print(sequence_explanation_prompt2)
    sequence_explanation2 = llm(sequence_explanation_prompt2, stop=["<SEQUENCEEND>"]).split("<SEQUENCEEND>")[0].strip()

    # Formatting the output into the desired JSON-like format for each body part
    output = (
            "Action: " + "\n" + action + "\n" +
            "Sequence Explanation: " + "\n" + sequence_explanation + "\n" +
            "Fine Motion Control Steps: " + "\n" +
            sequence_explanation2 + "\n" +
            "\n"
    )

    import os

    # 确保目录存在
    os.makedirs("llm_result", exist_ok=True)

    # 写入文件
    with open("llm_result/sequence_explanation.txt", "a") as file:
        file.write(output)


    # Print to console
    print(output)

    # Return results as well
    return sequence_explanation, sequence_explanation2


import re
import json

def parse_sequence_explanation(text):
    # 使用正则表达式匹配步骤
    steps = re.findall(r"(step\d+): (.+?)(?=\s*step\d+:|$)", text)
    # 将步骤转换为字典
    sequence_explanation = {step: desc.strip() for step, desc in steps}
    return sequence_explanation


def sequence_analyze_nobodypart(action):
    sequence_explanation_prompt = generate_sequence_explanation_prompt(action)
    sequence_explanation = llm(sequence_explanation_prompt, stop=["<SEQUENCEEND>"]).split("<SEQUENCEEND>")[0].strip()
    #print(sequence_explanation)

  
    # Return results as well
    return sequence_explanation
 
# text = """step1: The martial artist shifts their weight backward, lowering their center of gravity while bending their knees.   
# step2: The martial artist extends their arms outward to help maintain balance as they begin to fall backward onto the tatami.   
# step3: The martial artist tucks their chin to their chest and rolls onto their back to dissipate the impact smoothly, engaging their core muscles for control."""



import json
import re
import json
import re

# def sequence_analyze_nobodypart_jsontool(action, sequence_explanation):
#     """
#     分析动作序列并生成包含每个步骤描述和原始动作描述的 JSON 列表。

#     :param action: 原始动作描述。
#     :param sequence_explanation: 动作序列的详细步骤说明。
#     :return: 包含多个字典的列表，每个字典包含 "prompt" 和 "original prompt" 字段。
#     """
#     # 清理输入的 sequence_explanation，移除前后空格和多余的换行
#     sequence_explanation = sequence_explanation.strip()

#     # 使用正则表达式匹配每个步骤
#     # 采用非贪婪匹配，并通过断言确保不跨步骤匹配
#     step_pattern = re.compile(r'step\d+:\s*(.*?)\s*(?=step\d+:|$)', re.IGNORECASE | re.DOTALL)
#     steps = step_pattern.findall(sequence_explanation)

#     result = []

#     for step_description in steps:
#         step_description = step_description.strip()
#         if step_description:
#             step_json = {
#                 "prompt": step_description,
#                 "original prompt": action
#             }
#             result.append(step_json)

#     return result

import re
import re
import json

def sequence_analyze_nobodypart_jsontool(action, sequence_explanation):
    """
    解析动作序列说明，并将每个步骤转换为包含 "prompt" 和 "original prompt" 的字典列表。

    参数：
    - action (str): 原始动作描述。
    - sequence_explanation (str): 动作序列的详细步骤说明。

    返回：
    - list: 包含多个字典的列表，每个字典对应一个步骤。
    """
    # 清理输入的 sequence_explanation
    sequence_explanation = sequence_explanation.strip()
    
    # 使用正则表达式匹配所有的步骤和对应的描述
    # 模式解释：
    #   - (?m): 多行模式，使 ^ 和 $ 匹配每一行的开头和结尾
    #   - step\d+: 匹配步骤标签，如 step1:
    #   - \s*: 匹配标签后的任意空白字符
    #   - (.*?)(?=(\nstep\d+:)|$): 非贪婪地匹配描述内容，直到下一个步骤标签或字符串结尾
    pattern = r'(?m)^step\d+:\s*(.*?)(?=(\nstep\d+:)|$)'
    matches = re.findall(pattern, sequence_explanation, re.DOTALL)
    
    result = []
    for match in matches:
        step_description = match[0].strip()
        if step_description:
            step_json = {
                "prompt": step_description,
                "original prompt": action
            }
            result.append(step_json)
    
    return result







# # 示例输入
# action = "The person performs a rowing motion with their legs spread wide, and then leap over an obstacle with the agility of a parkour expert."
# sequence_explanation = """
# step1: The man spreads his legs wide while planting his feet firmly on the ground, then engages his core and pulls his arms back to prepare for the rowing motion.
# step2: The man's arms move forward in a rowing motion, pulling his torso towards his legs, while he simultaneously shifts his weight onto his heels to load his legs for the subsequent leap.
# step3: The man's legs then explosively push off the ground, propelling his body upward and forward as he clears the obstacle with agility, utilizing coordinated arm movement for balance and momentum.
# """



 
# [
#     {
#         "prompt": "The man spreads his legs wide while planting his feet firmly on the ground, then engages his core and pulls his arms back to prepare for the rowing motion.",
#         "original prompt": "The person performs a rowing motion with their legs spread wide, and then leap over an obstacle with the agility of a parkour expert."
#     },
#     {
#         "prompt": "The man's arms move forward in a rowing motion, pulling his torso towards his legs, while he simultaneously shifts his weight onto his heels to load his legs for the subsequent leap.",
#         "original prompt": "The person performs a rowing motion with their legs spread wide, and then leap over an obstacle with the agility of a parkour expert."
#     },
#     {
#         "prompt": "The man's legs then explosively push off the ground, propelling his body upward and forward as he clears the obstacle with agility, utilizing coordinated arm movement for balance and momentum.",
#         "original prompt": "The person performs a rowing motion with their legs spread wide, and then leap over an obstacle with the agility of a parkour expert."
#     }
# ]

