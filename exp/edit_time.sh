cd /home/haoyum3/momask-codes
python edit_t2m_time.py \
    --gpu_id 1 \
    --ext slide \
    -msec 0.,1.0 \
    --dataset_name t2m \
    --text_prompt "a person kneels down and slides on his knee"\
    --checkpoints_dir /extra/xielab0/haoyum3/momask/t2m \
    --res_name rtrans_general_b_base\
    --name trans_general_b \
    --which_epoch net_best_top1.tar\
    --edit_part 0\
    --source_motion /home/shanlins/momask-codes/editing/slide/raw/trajectory2.npy \
    --text_l_arm "a parson put his left arm on his side"\
    --text_r_arm "a parson put his left arm on his side"\
    --text_l_leg "a parson put his left arm on his side" \
    --text_r_leg "a parson put his left arm on his side" \
    --motion_length 0 \
    --vq_model net_best_fid.tar\
    #--use_res_model \


#/extra/xielab0/araujog/motion-generation/HumanML3D/new_joint_vecs/012667.npy