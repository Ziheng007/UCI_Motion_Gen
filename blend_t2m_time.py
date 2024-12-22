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
    result_dir = pjoin('./blending', opt.ext)
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
    source_motions = []
    length_list = []
    for source in opt.source_list:
        motion = np.load(source)
        motion = (motion - mean) / std
        m_length = len(motion)
        length_list.append(m_length)
        source_motions.append(torch.from_numpy(motion).to(opt.device).float())
    print('length_list:', length_list)
    name = 'blend_motions'

    # m_length = len(motion)
    # motion = (motion - mean) / std
    # if max_motion_length > m_length:
    #     motion = np.concatenate([motion, np.zeros((max_motion_length - m_length, motion.shape[1])) ], axis=0)
    # motion = torch.from_numpy(motion)[None].to(opt.device)

    token_lens = torch.LongTensor(length_list) // 4
    token_lens = token_lens.to(opt.device).long()

    m_length =(token_lens * 4)
    max_m_length = max(m_length)
    length_total = m_length.sum().item()

    sample = 0
    if opt.dataset_name == 'kit':
        kinematic_chain = kit_kinematic_chain
    else:
        kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()

    with torch.no_grad():
        tokens = []
        for motion in source_motions:
            print('motion:', motion.shape)
            token, _ = vq_model.encode(motion[None])
            padding_length = int(max_m_length//4 - token.shape[2])
            token = F.pad(token, (0, 0, 0, padding_length), "constant", model_opt.num_tokens + 1)
            tokens.append(token[..., 0])
        tokens = torch.cat(tokens)
        print('tokens:', tokens.shape)
        # tokens = torch.cat(tokens, dim=0)
    ### build editing mask, TOEDIT marked as 1 ###
    # edit_mask = torch.zeros_like(tokens[..., 0])# tokens [bs, body_parts, seq_len, q]
    # seq_len = tokens.shape[2]

    #TODO use your captions as input here (global text also works)
    captions = []
    for text in opt.text_list:
        captions.append(text)
    captions = tuple(captions)
    print(captions)
    #captions = ('A person falls down softly, bending knees and extending arms to avoid injury.','A parson pushes upwards with her hands and legs to stand, balancing their body.','A parson raises their right arm upwards and then lowers it to their side.','A parson jumps with both arms raised up.')
    with torch.no_grad():
        mids = t2m_transformer.long_range_alpha(captions, m_length//4, index_motion=tokens)
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

    joint_data = data
    joint = recover_from_ric(torch.from_numpy(joint_data).float(), num_joints).numpy()
    # soucre_joint = recover_from_ric(torch.from_numpy(source_data).float(), num_joints).numpy()
    bvh_path = pjoin(bvh_dir, "%s_ik.bvh"%(name))
    _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)
    bvh_path = pjoin(bvh_dir, "%s.bvh" % (name))
    _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)
    save_path = pjoin(animation_dir, "%s.mp4"%(name))
    ik_save_path = pjoin(animation_dir, "%s_ik.mp4"%(name))
    source_save_path = pjoin(animation_dir, "source_%s.mp4"%(name))
    plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title='ik_motion', fps=20)
    plot_3d_motion(save_path, kinematic_chain, joint, title='motion', fps=20)
    # plot_3d_motion(source_save_path, kinematic_chain, soucre_joint, title='source motion', fps=20)
    np.save(pjoin(joints_dir, "%s.npy"%(name)), joint)
    np.save(pjoin(joints_dir, "%s_ik.npy"%(name)), ik_joint)
    # np.save(pjoin(joints_dir, "source_%s.npy"%(name)), soucre_joint)
    np.save(pjoin(raw_dir, "%s.npy"%(name)),joint_data )