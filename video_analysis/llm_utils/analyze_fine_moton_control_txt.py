import os
import openai
import yaml
import re
import json
import re
from llm_utils import generate_sequence_explanation_prompt
from llm_utils import generate_fine_motion_control_prompt
from llm_utils import generate_sequence_explanation_prompt_json
from .llm_config  import llm

def analyze_fine_moton_control_txt(action):
    # Step 1: Get sequence explanation
    sequence_explanation_prompt = generate_sequence_explanation_prompt(action)
    sequence_explanation = llm(sequence_explanation_prompt, stop=["<SEQUENCEEND>"]).split("<SEQUENCEEND>")[0].strip()
    #print(sequence_explanation)

    # Step 2: Evaluate fine motion control
    fine_moton_control_prompt = generate_fine_motion_control_prompt(action, "none")
    control_evaluation = llm(fine_moton_control_prompt, stop=["<CONTROLEND>"]).split("<CONTROLEND>")[0].strip()

    # Parse the JSON objects from the control evaluation
    json_objects = re.findall(r'\{[^}]+\}', control_evaluation)
    control_results = [json.loads(obj) for obj in json_objects]

    # Output to file as well as print
    # output = {
    #     "Action": action,
    #     "Sequence Explanation": sequence_explanation,
    #     "Fine Motion Control Evaluation": control_results
    # }
    output = control_results

    # Write to file/home/haoyum3/momask-codes-hzh/llm_result/analyze_fine_moton_control_complex.json

    # with open("/home/haoyum3/momask-codes-hzh/llm_result/analyze_fine_moton_control_complex.json", "a") as file:
    #     json.dump(output, file, ensure_ascii=False, indent=2)
    #     file.write("\n")

    output2 = {
        "Action": action,
        "Sequence Explanation": sequence_explanation,
        "Fine Motion Control Evaluation": control_results
    }
    # with open("/home/haoyum3/momask-codes-hzh/llm_utils/fine_control_complex.txt", "a") as file:
    #     file.write(json.dumps(output2, ensure_ascii=False, indent=2))
    #     file.write("\n")

 

 
    return sequence_explanation, control_results





def analyze_fine_moton_control_txt_nosequence(action):
 

    # Step 2: Evaluate fine motion control
    fine_moton_control_prompt = generate_fine_motion_control_prompt(action, "none")
    control_evaluation = llm(fine_moton_control_prompt, stop=["<CONTROLEND>"]).split("<CONTROLEND>")[0].strip()

    # Parse the JSON objects from the control evaluation
    json_objects = re.findall(r'\{[^}]+\}', control_evaluation)
    control_results = [json.loads(obj) for obj in json_objects]

    # Output to file as well as print
    # output = {
    #     "Action": action,
    #     "Sequence Explanation": sequence_explanation,
    #     "Fine Motion Control Evaluation": control_results
    # }
    output = control_results

    # Write to file/home/haoyum3/momask-codes-hzh/llm_result/analyze_fine_moton_control_complex.json

    with open("/home/haoyum3/momask-codes-hzh/llm_result/analyze_fine_moton_control_complex.json", "a") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")

    output2 = {
        "Action": action,
 
        "Fine Motion Control Evaluation": control_results
    }
    # with open("/home/haoyum3/momask-codes-hzh/llm_utils/fine_control_complex.txt", "a") as file:
    #     file.write(json.dumps(output2, ensure_ascii=False, indent=2))
    #     file.write("\n")

    # Print to console
    #print(json.dumps(output, ensure_ascii=False, indent=2))

    # Return results as well
    return  control_results





