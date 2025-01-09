import sys
import subprocess
import os
import shutil
import string
from pathlib import Path
import tempfile

# 导入自定义的函数
sys.path.append("/home/haoyum3/momask-codes-hzh")
from llm_utils import (
    generate_sequence_explanation_prompt,
    generate_fine_motion_control_prompt,
    generate_sequence_explanation_prompt_json,
    sequence_analyze,
    analyze_fine_moton_control_txt,
    generate_comparison_feedback_prompt,
    decide_which_part_needs_editing_prompt,
    decide_which_part_needs_editing,
    test_generate_comparison_feedback
)
actions_to_test =[
    
    "A person walks forward while continuously bumping an imaginary volleyball with forearms, adjusting arms and footwork with each step.",
    "A person jogs in place, lifting knees high, while rhythmically punching forward with alternating arms.",
    "A martial artist performs forward steps, executing alternating punches and front kicks in sync with each step.",
    "A person performs a side shuffle, raising arms overhead to mimic catching a basketball, then lowers arms as they move back.",
    
    "A soccer player dribbles an imaginary ball with their feet, while using arms for balance and occasionally raising one arm as if calling for a pass.",
    "A boxer moves side-to-side in a defensive stance, shifting weight and throwing jabs while keeping their footwork light and responsive.",
    "A person skips forward, alternating between swinging arms up and mimicking a high-five motion with each step.",
    "A dancer performs a samba step, moving forward with a rhythmic sway, arms moving gracefully in sync with each leg movement.",
    
    "A volleyball player jumps to block, extending arms overhead with fingers spread, then lands softly and takes a few steps backward.",
    "A dancer leaps forward, arms sweeping from low to high, and lands into a forward crouch with one knee down and arms extended to the side.",
    "A track athlete performs a triple jump, swinging arms to gain momentum with each hop, step, and jump sequence.",
    "A person jumps with both feet, swinging both arms upward, as if trying to dunk a basketball on a low hoop.",
    
    "A person walks forward while waving with one hand, adjusting their stride and shifting their body to balance.",
    "A person walks briskly, holding a heavy bag in one hand, occasionally adjusting their grip and switching arms for balance.",
    
    "A firefighter charges forward while swinging an axe with one arm, alternating steps to maintain balance with each swing.",
    "A person walks stealthily forward, crouching low with bent knees, arms extended forward like a spy creeping up.",
]
#################edit by Shenghan: read prompts from .txt##################
texts_path = '/home/haoyum3/momask-codes-hzh/examples/texts.txt'
actions_to_test = []
with open(texts_path,'r') as f:
    for line in f.readlines():
        line = line.split('\n')[0].strip()
        actions_to_test.append(line)
############################################################################
def sanitize_filename(text):
    translator = str.maketrans('', '', string.punctuation)
    sanitized = text.translate(translator)
    sanitized = sanitized.replace(' ', '_')
    return sanitized
 
def process_action(action, sample_index=0, repeat_index=0, motion_length=128):
    
    base_dir = "/home/haoyum3/momask-codes-hzh"
    base_dir_nohzh = "/home/haoyum3/momask-codes"
    gen_script_path = Path("/tmp/gen_script.sh")
    generation_dir = os.path.join(base_dir, "generation")
    ext_gen = "trans_multi"
    result_dir_gen = os.path.join(generation_dir, ext_gen)
    joints_dir_gen = os.path.join(result_dir_gen, "joints")
    sample_dir_gen = os.path.join(joints_dir_gen, str(sample_index))
    raw_dir_gen = os.path.join(sample_dir_gen, "raw")
    os.makedirs(raw_dir_gen, exist_ok=True)

    # 分析动作文本，获取序列解释
    sequence_explanation, control_results = analyze_fine_moton_control_txt(action)
    
 
 
    enhanced_llm_exlanation = action + " In detail, " + sequence_explanation
 
    enhanced_llm_exlanation = enhanced_llm_exlanation.replace('\n', ' ')
 
    with open('enhanced_llm_exlanation.txt', 'a', encoding='utf-8') as file:
        file.write(enhanced_llm_exlanation + '\n')   

    

    # 生成 source_motion 的 Bash 脚本内容
    bash_script_content = f"""#!/bin/bash
cd {base_dir}
python gen_t2m_time.py \\
  --gpu_id 0 \\
  --ext {ext_gen} \\
  --checkpoints_dir /extra/xielab0/haoyum3/momask/t2m \\
  --res_name rtrans_time_multi_q \\
  --name trans_multi \\
  --text_prompt "{enhanced_llm_exlanation}" \\
  --which_epoch net_best_fid.tar \\
  --motion_length {motion_length}
"""
    # 保存并执行 Bash 脚本
    gen_script_path.write_text(bash_script_content)
    gen_script_path.chmod(0o755)
    subprocess.run(["/bin/bash", str(gen_script_path)])
    gen_script_path.unlink()

    # 准备保存结果的目录和文件名
    result_npy_dir1 = "/home/haoyum3/momask-codes-hzh/MOTION_npy_result_of_line25and26_enhancedbyllm/"
    os.makedirs(result_npy_dir1, exist_ok=True)
    # 生成安全的文件名
    sanitized_action = sanitize_filename(action)

    # 处理源动作文件
    source_generation_dir = "/home/haoyum3/momask-codes-hzh/generation/trans_multi/joints/0/"
    for file_name in os.listdir(source_generation_dir):
        if file_name.endswith(".npy") and not file_name.startswith("raw"):
            original_path = os.path.join(source_generation_dir, file_name)
            base_name = file_name[:-4]  # 移除 .npy
            new_file_name = f"{base_name}_{sanitized_action}.npy"
            destination_path = os.path.join(result_npy_dir1, new_file_name)
            shutil.copy(original_path, destination_path)
            print(f"源动作文件已复制并重命名为 {destination_path}")

# 主循环：处理所有动作
for action in actions_to_test:
    print(f"正在处理动作：{action}")
    process_action(action)