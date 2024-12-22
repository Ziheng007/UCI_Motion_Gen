cd /home/haoyum3/momask-codes
python blend_t2m_time.py \
    --gpu_id 1 \
    --ext blend_test \
    --dataset_name t2m \
    --checkpoints_dir /extra/xielab0/haoyum3/momask/t2m \
    --res_name rtrans_general_b_base\
    --name trans_general_b \
    --which_epoch net_best_top1.tar\
    --source_list /home/haoyum3/momask-codes/generation/sequential_1st_fix/raw/2_0.npy /home/haoyum3/momask-codes/generation/sequential_1st_fix/raw/2_1.npy /home/haoyum3/momask-codes/generation/sequential_1st_fix/raw/2_2.npy\
    --text_list 'person spins in the air with arms for balance and jumps.' 'A person bends down, sits on the ground to rest and then get up.' 'a person jumps and spins using their arms for balance.'\
    # --use_res_model \


    # /extra/xielab0/araujog/motion-generation/HumanML3D/new_joint_vecs/012667.npy