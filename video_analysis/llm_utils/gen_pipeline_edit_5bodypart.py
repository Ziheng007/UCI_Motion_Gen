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

# [
#     "The person performs a rowing motion with their legs spread wide.",
#     "A woman hops forward while holding a T-pose.",
#     # "The man executes a jump spin mid-air.",
#     "A person crawls on the ground in a baby-like motion.",
#     "A dancer spins gracefully in a ballet twirl.",
#     "The computer science student attempts one-armed push-ups.",
#     "Mimic a gun-slinging motion, drawing from the hip, then aiming in a dramatic stance."
      
#     "Perform a signature James Bond pose with a dramatic turn and gunpoint.",
#     "Fight with the fluid precision and power of Bruce Lee.",
#     "Perform a graceful spinning kick in the style of Jet Li.",
#     "A basketball player dribbles, then jumps high, mimicking a slam dunk motion."
#     "A waiter carries a tray overhead, then sidesteps gracefully to avoid obstacles."
#     # "Perform a slow-motion dive while firing two handguns like in an action movie.",
#     # "Execute a precise sword slash followed by a defensive stance like a samurai.",
#     # "Jump from a ledge and roll upon landing to minimize impact, as seen in martial arts films.",
#     # "Perform a quick disarm move to take an opponent’s weapon away.",
#     # "Perform a quick, tactical reload of a firearm while maintaining a defensive stance."
# ]

# List of actions to test
actions_to_test =[
    
    "A person walks forward while continuously bumping an imaginary volleyball with forearms, adjusting arms and footwork with each step.",
    "A person jogs in place, lifting knees high, while rhythmically punching forward with alternating arms.",
    # "A martial artist performs forward steps, executing alternating punches and front kicks in sync with each step.",
    # "A person performs a side shuffle, raising arms overhead to mimic catching a basketball, then lowers arms as they move back.",
    # "A person lunges forward with one leg while swinging a tennis racket, then quickly steps back, ready to swing again.",
    # "A person runs in place, high-kneeing while mimicking a double-hand volleyball set above their head.",
    
    # "A soccer player dribbles an imaginary ball with their feet, while using arms for balance and occasionally raising one arm as if calling for a pass.",
    # "A boxer moves side-to-side in a defensive stance, shifting weight and throwing jabs while keeping their footwork light and responsive.",
    # "A person skips forward, alternating between swinging arms up and mimicking a high-five motion with each step.",
    # "A dancer performs a samba step, moving forward with a rhythmic sway, arms moving gracefully in sync with each leg movement.",
    # "A basketball player performs a defensive slide, moving side-to-side with hands outstretched, ready to block.",

    # "A volleyball player jumps to block, extending arms overhead with fingers spread, then lands softly and takes a few steps backward.",
    # "A dancer leaps forward, arms sweeping from low to high, and lands into a forward crouch with one knee down and arms extended to the side.",
    # "A track athlete performs a triple jump, swinging arms to gain momentum with each hop, step, and jump sequence.",
    # "A person jumps with both feet, swinging both arms upward, as if trying to dunk a basketball on a low hoop.",
    # "A martial artist performs a jump kick, raising one knee while thrusting the opposite leg forward, arms moving to balance mid-air.",

    # "A person walks forward while waving with one hand, adjusting their stride and shifting their body to balance.",
    # "A person walks briskly, holding a heavy bag in one hand, occasionally adjusting their grip and switching arms for balance.",
    # "A hiker climbs an incline, reaching forward with one arm as if grabbing onto rocks, and then pulls their body up with each step.",
    # "A person walks and lifts their arm to shield their eyes, adjusting steps to mimic looking around.",
    # "A person shuffles forward, both arms extended as if pulling a heavy rope, adjusting steps and shifting weight with each pull.",

    # "A firefighter charges forward while swinging an axe with one arm, alternating steps to maintain balance with each swing.",
    # "A person walks stealthily forward, crouching low with bent knees, arms extended forward like a spy creeping up.",
    # "A knight marches forward with a shield in one hand and a sword in the other, occasionally raising the shield in defense.",
    # "A soldier advances with a rifle, moving forward in a crouched stance and quickly glancing side-to-side.",
    # "A medieval warrior walks forward, raising a spear with both hands, then lunges forward in a dramatic stabbing motion.",
]

# [
#     "一个人向前走的同时用前臂不断垫起一个假想的排球，随着每一步调整手臂和步伐。",
#     "一个人原地慢跑，高抬膝，同时交替向前挥拳。",
#     "一个武术家向前走的同时，交替出拳和前踢，每一步都与动作同步。",
#     "一个人侧身滑步，双臂高举模拟接住篮球，回到起始位置时放下手臂。",
#     "一个人向前弓步，一只手挥动网球拍，然后迅速后退，准备再次挥拍。",
#     "一个人原地跑步，高抬膝，同时双手在头顶模拟排球的传球动作。",
    
#     "一个足球运动员用脚带球，同时用手保持平衡，偶尔抬起一只手像是在招呼队友传球。",
#     "一个拳击手侧身移动，保持防御姿势，脚步轻盈，时不时抛出刺拳。",
#     "一个人向前跳跃步进，手臂随每步摆动，并模拟每一步都高举的击掌动作。",
#     "一个舞者前进，随着节奏摆动，手臂与腿部动作协调，呈现出桑巴舞的步伐。",
#     "一个篮球运动员在防守滑步，侧身移动，双手伸展，随时准备阻挡对手。",

#     "一个排球运动员跳跃封网，双臂在头顶伸展，手指张开，然后轻轻落地并向后退几步。",
#     "一个舞者向前旋转跳跃，手臂从低向高摆动，落地后进入一个前蹲姿势，一膝跪地，双臂侧展。",
#     "一个田径运动员进行三级跳，随着每一步摆动手臂以增加动能，完成整个跳、跨、跳的动作。",
#     "一个人双脚跳起，双臂向上挥动，仿佛试图扣篮。",
#     "一个武术家执行跳踢动作，一膝抬起，同时另一条腿向前踢出，手臂在空中调整平衡。",

#     "一个人一边向前走一边挥手，调整步伐并侧身保持平衡。",
#     "一个人快步前行，一手提着重物，偶尔调整抓握并换手保持平衡。",
#     "一个登山者攀爬斜坡，一手向前抓住岩石，然后用力向上拉起身体。",
#     "一个人走路时抬起手臂遮住眼睛，步伐调整以模拟四处张望。",
#     "一个人向前拖步，双臂前伸，仿佛在拉动一根沉重的绳子，随着每一下调整步伐并转移重量。",

#     "一个消防员前进的同时单手挥动斧头，每次挥动时交替脚步保持平衡。",
#     "一个人偷偷向前走，蹲低膝盖，双臂前伸，像间谍一样偷偷摸摸。",
#     "一个骑士前行，一手持盾，一手持剑，偶尔举盾进行防御。",
#     "一个士兵持枪前进，身体保持半蹲姿势，快速地左右张望。",
#     "一个中世纪战士向前行走，双手高举长矛，然后向前刺出，动作充满戏剧性。",
# ]


# 函数：将动作文本转换为安全的文件名（移除标点符号，替换空格为下划线）
def sanitize_filename(text):
    translator = str.maketrans('', '', string.punctuation)
    sanitized = text.translate(translator)
    sanitized = sanitized.replace(' ', '_')
    return sanitized

# 函数：处理每个动作
def process_action(action, sample_index=0, repeat_index=0, motion_length=128):
    # 基本路径和目录
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

    # 生成 source_motion 的 Bash 脚本内容
    bash_script_content = f"""#!/bin/bash
cd {base_dir}
python gen_t2m_time.py \\
  --gpu_id 0 \\
  --ext {ext_gen} \\
  --checkpoints_dir /extra/xielab0/haoyum3/momask/t2m \\
  --res_name rtrans_time_multi_q \\
  --name trans_multi \\
  --text_prompt "{action}" \\
  --which_epoch net_best_fid.tar \\
  --motion_length {motion_length}
"""
    # 保存并执行 Bash 脚本
    gen_script_path.write_text(bash_script_content)
    gen_script_path.chmod(0o755)
    subprocess.run(["/bin/bash", str(gen_script_path)])
    gen_script_path.unlink()

    # # 构建 source_motion 文件路径
    # source_motion_filename = f"raw_sample{sample_index}_repeat{repeat_index}_len{motion_length}.npy"
    # source_motion_path = os.path.join(
    #     raw_dir_gen,
    #     source_motion_filename
    # )

    # # 验证 source_motion 文件是否存在
    # if not os.path.exists(source_motion_path):
    #     print(f"错误：未找到 source_motion 文件，路径为 {source_motion_path}")
    #     return

    # 分析动作文本，获取各肢体的描述
    sequence_explanation, control_results = analyze_fine_moton_control_txt(action)

    # 提取每个肢体的描述
    body_part_descriptions = {
        "left arm": "",
        "right arm": "",
        "left leg": "",
        "right leg": ""
    }

    for result in control_results:
        part = result.get("body part", "").lower()
        description = result.get("description", "")
        if part in body_part_descriptions:
            body_part_descriptions[part] = action + description
       

 
    bash_script_content = f"""#!/bin/bash
cd /home/haoyum3/momask-codes-hzh
python edit_t2m_time_v2.py \\
    --gpu_id 1 \\
    --ext exp4 \\
    -msec 0.,1.0 \\
    --text_prompt "{action}" \\
    --checkpoints_dir /extra/xielab0/haoyum3/momask/t2m \\
    --res_name rtrans_time_multi_q \\
    --name trans_multi \\
    --which_epoch net_best_fid.tar \\
    --edit_part 0 1 2 3 \\
    --source_motion  /home/haoyum3/momask-codes-hzh/generation/trans_multi/joints/0/raw/raw_sample0_repeat0_len128.npy \\
    --text_l_arm "{body_part_descriptions['left arm']}" \\
    --text_r_arm "{body_part_descriptions['right arm']}" \\
    --text_l_leg "{body_part_descriptions['left leg']}" \\
    --text_r_leg "{body_part_descriptions['right leg']}" \\
    --motion_length {motion_length}
"""

    # 创建临时 Bash 脚本文件，并将其放在指定目录中
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sh", dir="/home/haoyum3/momask-codes-hzh/trash") as temp_script:
        temp_script.write(bash_script_content.encode())
        temp_script_path = temp_script.name

    print(f"Temporary script created at: {temp_script_path}")

    # 给脚本执行权限
    os.chmod(temp_script_path, 0o770)

    # 执行临时 Bash 脚本
    subprocess.run(["bash", temp_script_path])

    # 删除临时脚本
    # os.remove(temp_script_path)

    # 定义 edit_t2m_time.py 生成的编辑后动作文件路径
    # 根据用户的输出，假设编辑后的文件位于 generation/exp4/joints/0/raw/
    edited_ext = "exp4"
    result_dir_edit = os.path.join(base_dir, "editing", edited_ext, "joints", "0")
    # edited_motion_path = os.path.join(
    #     result_dir_edit,
    #     source_motion_filename # 假设文件名与 source_motion 相同
    # )

    # 根据您的说明，编辑后的文件路径为：
    edited_motion_dir = "/home/haoyum3/momask-codes-hzh/editing/exp4/joints/0/"
    
    # 验证编辑后的动作文件是否存在
    # if not os.path.exists(edited_motion_path):
    #     print(f"错误：未找到编辑后的动作文件，路径为 {edited_motion_path}")
    #     return

    # 准备保存结果的目录和文件名
    result_npy_dir1 = "/home/haoyum3/momask-codes-hzh/source_result_npy1102/"
    result_npy_dir2 = "/home/haoyum3/momask-codes-hzh/edit_result_npy1102/"
    
    os.makedirs(result_npy_dir1, exist_ok=True)
    os.makedirs(result_npy_dir2, exist_ok=True)
    # 生成安全的文件名
    sanitized_action = sanitize_filename(action)

    # 1. 处理源动作文件
    source_generation_dir = "/home/haoyum3/momask-codes-hzh/generation/trans_multi/joints/0/"
    for file_name in os.listdir(source_generation_dir):
        if file_name.endswith(".npy") and not file_name.startswith("raw"):
            original_path = os.path.join(source_generation_dir, file_name)
            base_name = file_name[:-4]  # 移除 .npy
            new_file_name = f"{base_name}_{sanitized_action}.npy"
            destination_path = os.path.join(result_npy_dir1, new_file_name)
            shutil.copy(original_path, destination_path)
            print(f"源动作文件原本在 {original_path}")
            print(f"源动作文件已复制并重命名为 {destination_path}")

    # 2. 处理编辑后的动作文件
    for file_name in os.listdir(edited_motion_dir):
        if file_name.endswith(".npy") and file_name.startswith("sample"):
            original_path = os.path.join(edited_motion_dir, file_name)
            base_name = file_name[:-4]  # 移除 .npy
            new_file_name = f"{base_name}_{sanitized_action}.npy"
            destination_path = os.path.join(result_npy_dir2, new_file_name)
            shutil.copy(original_path, destination_path)
            print(f"编辑后动作文件原本在 {original_path}")
            print(f"编辑后动作文件已复制并重命名为 {destination_path}")

# 主循环：处理所有动作
for action in actions_to_test:
    print(f"正在处理动作：{action}")
    process_action(action)
    
    
    
    #/home/haoyum3/momask-codes-hzh/generation/trans_multi/joints/0/raw/