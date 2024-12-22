#!/bin/bash
#SBATCH --job-name=eval_t2m_vq
#SBATCH --output=/home/shanlinsun/code_base/momask-codes/log/eval_t2m_vq_%j_log.out
#SBATCH --error=/home/shanlinsun/code_base/momask-codes/log/eval_t2m_vq_%j_log.err
#SBATCH --cpus-per-task=1
#SBATCH --partition=hpg-ai
#SBATCH --gpus=1
#SBATCH --mem=79GB
#SBATCH --nodes=1
#SBATCH --time=4320

# 切换到工作目录
cd /home/haoyum3/momask-codes

# 运行 Python 脚本
python eval_t2m_vq.py \
    --gpu_id 0 \
    --name rvq_nq6_dc512_nc512_noshare_qdp0.2 \
    --dataset_name t2m \
    --ext rvq_nq6_dc512_nc512_noshare_qdp0.2 \
    --checkpoints_dir /home/haoyum3/MMM/output/rvq