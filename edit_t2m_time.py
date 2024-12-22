import os
from os.path import join as pjoin

import torch
import torch.nn.functional as F

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE, LengthEstimator
from models.vq.model_general import RVQVAE_Decode
from models.vq.model_multi import RVQVAE_Multi
from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.fixseed import fixseed
from visualization.joints2bvh import Joint2BVHConvertor

from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain,kit_kinematic_chain

import numpy as np

from gen_t2m_time import load_res_model, load_trans_model, load_vq_model

if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 251 if opt.dataset_name == 'kit' else 263

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('./editing', opt.ext)
    bvh_dir = pjoin(result_dir, 'bvh')
    joints_dir = pjoin(result_dir, 'joints')
    joints_ik_dir = pjoin(joints_dir, 'ik')
    joints_no_ik_dir = pjoin(joints_dir, 'no_ik')
    animation_dir = pjoin(result_dir, 'animations')
    raw_dir = pjoin(result_dir, 'raw')
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir,exist_ok=True)
    os.makedirs(bvh_dir,exist_ok=True)
    os.makedirs(joints_ik_dir,exist_ok=True)
    os.makedirs(joints_no_ik_dir,exist_ok=True)
    os.makedirs(raw_dir,exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)

    #######################
    ######Loading RVQ######
    #######################
    # model_opt.vq_name = 'rvq_multi_4_128_q'
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose
    vq_model, vq_opt = load_vq_model(opt,vq_opt,mean,std)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    #################################
    ######Loading R-Transformer######
    #################################
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    res_model = load_res_model(res_opt, vq_opt, opt)

    # print('res_opt.vq_name', res_opt.vq_name)
    # print('model_opt.vq_name', model_opt.vq_name)
    # assert res_opt.vq_name == model_opt.vq_name

    #################################
    ######Loading M-Transformer######
    #################################
    t2m_transformer = load_trans_model(model_opt, opt, 'net_best_top1.tar')

    t2m_transformer.eval()
    vq_model.eval()
    # res_model.eval()

    # res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)

    ##### ---- Data ---- #####
    max_motion_length = 196
    def inv_transform(data):
        return data * std + mean
    ### We provided an example source motion (from 'new_joint_vecs') for editing. See './example_data/000612.mp4'###
    motion = np.load(opt.source_motion)
    name = opt.source_motion.split('/')[-1].split('.')[0].strip()
    m_length = len(motion)
    motion = (motion - mean) / std
    if max_motion_length > m_length:
        motion = np.concatenate([motion, np.zeros((max_motion_length - m_length, motion.shape[1])) ], axis=0)
    motion = torch.from_numpy(motion)[None].to(opt.device)

    prompt_list = []
    length_list = []
    if opt.motion_length == 0:
        opt.motion_length = m_length
        print("Using default motion length.")
    
    prompt_list.append(opt.text_l_arm)
    prompt_list.append(opt.text_r_arm)
    prompt_list.append(opt.text_l_leg)
    prompt_list.append(opt.text_r_leg)
    length_list.append(opt.motion_length)
    if opt.text_prompt == "":
        raise "Using an empty text prompt."

    token_lens = torch.LongTensor(length_list) // 4
    token_lens = token_lens.to(opt.device).long()

    m_length =(token_lens * 4)
    captions = prompt_list
   
    
    _edit_slice = opt.mask_edit_section
    edit_slice = []
    for eds in _edit_slice:
        _start, _end = eds.split(',')
        _start = eval(_start)
        _end = eval(_end)
        edit_slice.append([_start, _end])

    sample = 0
    if opt.dataset_name == 'kit':
        kinematic_chain = kit_kinematic_chain
    else:
        kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()

    with torch.no_grad():
        tokens, features = vq_model.encode(motion)
    ### build editing mask, TOEDIT marked as 1 ###
    edit_mask = torch.zeros_like(tokens[..., 0])# tokens [bs, body_parts, seq_len, q]
    seq_len = tokens.shape[2]
    if 4 in opt.edit_part:
        edit_part = [0, 1, 2, 3]
        print_captions = ''
        for i in prompt_list:
            print_captions += i
        print('Edit whole body')
    else:
        edit_part = opt.edit_part
        body_parts = ['left arm','right_arm','left_leg','right_leg']
        print('Edit %s'%[body_parts[i] for i in edit_part])
        print_captions = ''
        for i in edit_part:
            print_captions += prompt_list[i]
             
    for _start, _end in edit_slice:
        if isinstance(_start, float):
            _start = int(_start*seq_len)
            _end = int(_end*seq_len)
        else:
            _start //= 4
            _end //= 4
        edit_mask[:, edit_part, _start: _end] = 1
        print_captions = f'{print_captions} [{_start*4/20.}s - {_end*4/20.}s]'
    edit_mask = edit_mask.bool()
    print("captions:", captions)
    with torch.no_grad():
        mids = tokens[..., 0]
        mids = t2m_transformer.edit(
                                    captions, tokens[..., 0].clone(), m_length//4,
                                    timesteps=opt.time_steps,
                                    cond_scale=opt.cond_scale,
                                    temperature=opt.temperature,
                                    topk_filter_thres=opt.topkr,
                                    gsample=opt.gumbel_sample,
                                    force_mask=opt.force_mask,
                                    edit_mask=edit_mask.clone(),
                                    edit_parts=edit_part,
                                    )
        if opt.use_res_model:
            print('Using residual transformer')
            mids = res_model.generate(mids, captions, m_length//4, temperature=1, cond_scale=2)
        else:
            mids.unsqueeze_(-1)
        pred_motions = vq_model.forward_decoder(mids)
        pred_motions = pred_motions.detach().cpu().numpy()
        source_motions = motion.detach().cpu().numpy()
        data = inv_transform(pred_motions)[0]
        source_data = inv_transform(source_motions)[0]


    if opt.dataset_name == 'kit':
        num_joints = 21
    else:
        num_joints = 22
    joint_data = data[:m_length]
    joint = recover_from_ric(torch.from_numpy(joint_data).float(), num_joints).numpy()
    source_data = source_data[:m_length]
    soucre_joint = recover_from_ric(torch.from_numpy(source_data).float(), num_joints).numpy()
    bvh_path = pjoin(bvh_dir, "%s_ik.bvh"%(name))
    _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)
    bvh_path = pjoin(bvh_dir, "%s.bvh" % (name))
    _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)
    save_path = pjoin(animation_dir, "%s.mp4"%(name))
    ik_save_path = pjoin(animation_dir, "%s_ik.mp4"%(name))
    source_save_path = pjoin(animation_dir, "source_%s.mp4"%(name))
    plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=print_captions, fps=20)
    plot_3d_motion(save_path, kinematic_chain, joint, title=print_captions, fps=20)
    plot_3d_motion(source_save_path, kinematic_chain, soucre_joint, title='None', fps=20)
    np.save(pjoin(joints_dir, "%s.npy"%(name)), joint)
    np.save(pjoin(joints_dir, "%s_ik.npy"%(name)), ik_joint)
    np.save(pjoin(joints_dir, "source_%s.npy"%(name)), soucre_joint)
    np.save(pjoin(raw_dir, "%s.npy"%(name)),joint_data )