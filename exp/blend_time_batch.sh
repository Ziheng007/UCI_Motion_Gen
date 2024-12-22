cd /home/haoyum3/momask-codes
python blend_t2m_time_batch.py \
    --gpu_id 1 \
    --ext sequential \
    --dataset_name t2m \
    --checkpoints_dir /extra/xielab0/haoyum3/momask/t2m \
    --res_name rtrans_general_b_base\
    --name trans_general_b \
    --which_epoch net_best_top1.tar\
    --source_dir /home/haoyum3/momask-codes/editing/sequential_final/raw\
    --text_dir /home/haoyum3/momask-codes/editing/sequential_final/prompt\
    --group_dir /home/haoyum3/momask-codes/editing/sequential_final/group\ 
    # --use_res_model \