#!/bin/bash
#SBATCH --job-name=rvq
#SBATCH --output=/home/shanlinsun/code_base/momask-codes/log/%j_log.out
#SBATCH --error=/home/shanlinsun/code_base/momask-codes/log/%j_log.err
#SBATCH --cpus-per-task=1
#SBATCH --partition=hpg-ai
#SBATCH --gpus=1
#SBATCH --mem=79GB
#SBATCH --nodes=1
#SBATCH --time=4320
cd /home/haoyum3/momask-codes
python train_vq_general_new.py \
    --name rvq_general \
    --gpu_id 6 \
    --dataset_name t2m \
    --batch_size 32 \
    --num_quantizers 6 \
    --max_epoch 50 \
    --quantize_dropout_prob 0.2 \
    --gamma 0.05 \
    --checkpoints_dir /extra/xielab0/haoyum3/momask/t2m\
    --teacher_net /extra/xielab0/haoyum3/momask/t2m/t2m/rvq_multi_4_128_t
