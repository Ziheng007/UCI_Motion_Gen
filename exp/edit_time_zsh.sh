cd /home/haoyum3/momask-codes
# is_final_round
python edit_t2m_time_zsh.py \
    --gpu_id 0 \
    --ext test_2nd \
    -msec 0.,1.0 \
    --dataset_name t2m \
    --checkpoints_dir /extra/xielab0/haoyum3/momask/t2m \
    --res_name rtrans_general_b_base\
    --name trans_general_b \
    --which_epoch net_best_top1.tar\
    --source_dir /home/haoyum3/momask-codes/editing/test/raw/ \
    --instruction_dir  /home/haoyum3/momask-codes/editing/test_2nd/instruction\
    --final_dir /home/haoyum3/momask-codes/editing/test_final\
    --motion_length 0 \
    --vq_model net_best_fid.tar\
    #--is_final_round