#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=/home/shanlinsun/code_base/momask-codes/log/test_%j_log.out
#SBATCH --error=/home/shanlinsun/code_base/momask-codes/log/test_%j_log.err
#SBATCH --cpus-per-task=1
#SBATCH --partition=hpg-ai
#SBATCH --gpus=1
#SBATCH --mem=79GB
#SBATCH --nodes=1
#SBATCH --time=4320
cd /home/haoyum3/momask-codes
python eval_t2m_trans_res.py \
    --res_name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw \
    --dataset_name t2m \
    --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns \
    --recap_name re-captions_4o \
    --gpu_id 0 \
    --cond_scale 4 \
    --time_steps 10 \
    --ext evaluation \
    --is_recap \
    --which_epoch all 