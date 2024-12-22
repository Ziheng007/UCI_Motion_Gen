#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --output=/red/ruogu.fang/shanlinsun/t2m/momask_log/gen_%j_log.out
#SBATCH --error=/red/ruogu.fang/shanlinsun/t2m/momask_log/gen_%j_log.err
#SBATCH --cpus-per-task=1
#SBATCH --partition=hpg-ai
#SBATCH --gpus=1
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --time=4320
cd /home/haoyum3/momask-codes
python gen_t2m_time.py \
    --gpu_id 0 \
    --ext trans_general \
    --checkpoints_dir /extra/xielab0/haoyum3/momask/t2m\
    --res_name rtrans_general_b_base\
    --name trans_general_b \
    --text_prompt "the police officer chases someone on foot rapidly" \
    --text_l_arm "The man's left arm is raised to shoulder height with a slight bend at the elbow, moving forward and backward as he bumps the imaginary volleyball." \
    --text_r_arm "The man's right arm is raised to shoulder height with a slight bend at the elbow, moving in coordination with the left arm to bump the imaginary volleyball." \
    --text_l_leg "The man's left leg is lifted and extended forward with each step as he walks." \
    --text_r_leg "The man's right leg follows the left leg, lifted and extended forward with each step." \
    --which_epoch net_best_top1.tar\
    --motion_length 128\
    --vq_model net_best_fid.tar