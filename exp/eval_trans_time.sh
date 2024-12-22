#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=/home/shanlinsun/code_base/momask-codes/log/test_%j_log.out
#SBATCH --error=/home/shanlinsun/code_base/momask-codes/log/test_%j_log.err
#SBATCH --cpus-per-task=1
#SBATCH --partition=hpg-ai
#SBATCH --gpus=1
#SBATCH --mem=20GB
#SBATCH --nodes=1
#SBATCH --time=4320
cd /home/haoyum3/momask-codes
python eval_t2m_trans_time.py \
    --res_name rtrans_general_b_base \
    --dataset_name t2m \
    --name trans_general_b \
    --checkpoints_dir /extra/xielab0/haoyum3/momask/t2m \
    --gpu_id 0 \
    --cond_scale 4 \
    --time_steps 10 \
    --ext evaluation \
    --recap_name re-captions_4o \
    --which_epoch all \
    --is_recap