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
python gen_t2m_time_batch.py \
    --gpu_id 3 \
    --ext sequential_1st \
    --checkpoints_dir /extra/xielab0/haoyum3/momask/t2m\
    --res_name rtrans_general_b_base\
    --name trans_general_b \
    --vq_model net_best_fid.tar \
    --text_path  /home/haoyum3/momask-codes/generation/sequential_1st/data\
    --which_epoch net_best_top1.tar\
    --motion_length 196\
    --vq_model net_best_fid.tar


# bash /home/haoyum3/momask-codes/exp/gen_time_batch_hzh.sh
# /home/haoyum3/momask-codes/generation/sequential_1st/data