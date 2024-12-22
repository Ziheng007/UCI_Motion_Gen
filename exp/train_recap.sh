#!/bin/bash
#SBATCH --job-name=recap
#SBATCH --output=/red/ruogu.fang/shanlinsun/t2m/momask_log/recap_%j_log.out
#SBATCH --error=/red/ruogu.fang/shanlinsun/t2m/momask_log/recap_%j_log.err
#SBATCH --cpus-per-task=1
#SBATCH --partition=hpg-ai
#SBATCH --gpus=1
#SBATCH --mem=79GB
#SBATCH --nodes=1
#SBATCH --time=4320
cd /home/haoyum3/momask-codes
python train_t2m_Timesformer_recap.py \
    --name trans_general_recap_video \
    --latent_dim 512\
    --n_layers 9\
    --n_heads 8\
    --gpu_id 0 \
    --dataset_name t2m \
    --batch_size 128 \
    --vq_name rvq_general \
    --checkpoints_dir /extra/xielab0/haoyum3/momask/t2m\
    --max_epoch 1000