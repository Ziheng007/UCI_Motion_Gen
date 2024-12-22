import os
from os.path import join as pjoin
import json
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

parser = EvalT2MOptions()
opt = parser.parse()
fixseed(opt.seed)
opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
torch.autograd.set_detect_anomaly(True)
root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
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
dim_pose = 251 if opt.dataset_name == 'kit' else 263
mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
    
def load_model():
    #######################
    ######Loading RVQ######
    #######################
    # model_opt.vq_name = 'rvq_multi_4_128_q'
    
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
    return t2m_transformer, vq_model, res_model

def inv_transform(data):
    return data * std + mean

def process_motion(motion):
    ##### ---- Data ---- #####
    max_motion_length = 196
    ### We provided an example source motion (from 'new_joint_vecs') for editing. See './example_data/000612.mp4'###
    m_length = len(motion)
    motion = (motion - mean) / std
    if max_motion_length > m_length:
        motion = np.concatenate([motion, np.zeros((max_motion_length - m_length, motion.shape[1])) ], axis=0)
    motion = torch.from_numpy(motion)[None].to(opt.device)
    return motion
    
def preapare_data(edit_instruction,m_length, part):

    prompt_list = []
    #if opt.motion_length != 0:
    #    m_length = opt.motion_length
    body_parts = ["left arm","right arm", 'left leg', 'right leg']
    edit_part = []
    prompt = ''
    if part == 0:
        body_part_edit = edit_instruction["Upper body"]
        if body_part_edit != 'None':
            edit_part = [0,1]
            prompt = body_part_edit
    elif part == 1:
        body_part_edit = edit_instruction["Lower body"]
        if body_part_edit != 'None':
            edit_part = [2,3]
            prompt = body_part_edit
    prompt_list.append(prompt)
    prompt_list = prompt_list*4
    print('Edit %s'%[body_parts[i] for i in edit_part])
    
    token_lens = torch.LongTensor([m_length]) // 4
    token_lens = token_lens.to(opt.device).long()
    m_length = token_lens * 4
    captions = prompt_list
    
    if not edit_part:
        flag = True
    else:
        flag = False
    return captions, m_length,edit_part, prompt,flag


def save_reasults(joint_data,name,print_captions):
    if opt.dataset_name == 'kit':
        kinematic_chain = kit_kinematic_chain
    else:
        kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()
    if opt.dataset_name == 'kit':
        num_joints = 21
    else:
        num_joints = 22
    joint = recover_from_ric(torch.from_numpy(joint_data).float(), num_joints).numpy()
    bvh_path = pjoin(bvh_dir, "%s_ik.bvh"%(name))
    _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)
    bvh_path = pjoin(bvh_dir, "%s.bvh" % (name))
    _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)
    save_path = pjoin(animation_dir, "%s.mp4"%(name))
    #ik_save_path = pjoin(animation_dir, "%s_ik.mp4"%(name))
    #plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=print_captions, fps=20)
    plot_3d_motion(save_path, kinematic_chain, joint, title=print_captions, fps=20)
    np.save(pjoin(joints_no_ik_dir, "%s.npy"%(name)), joint)
    np.save(pjoin(joints_ik_dir, "%s.npy"%(name)), ik_joint)
    np.save(pjoin(raw_dir, "%s.npy"%(name)),joint_data )

def save_results_final(motion,name,print_caption):
    if opt.final_dir:
        final_raw_dir = pjoin(opt.final_dir, 'raw')
        final_animation_dir = pjoin(opt.final_dir, 'animation')
        final_joints_dir = pjoin(opt.final_dir, 'joints')
        final_joints_ik_dir = pjoin(opt.final_dir, 'joints_ik')
        final_bvh_dir = pjoin(opt.final_dir, 'bvh')
        os.makedirs(final_animation_dir, exist_ok=True)
        os.makedirs(final_raw_dir, exist_ok=True)
        os.makedirs(final_joints_dir, exist_ok=True)
        os.makedirs(final_joints_ik_dir, exist_ok=True)
        os.makedirs(final_bvh_dir, exist_ok=True)
        if opt.dataset_name == 'kit':
            kinematic_chain = kit_kinematic_chain
        else:
            kinematic_chain = t2m_kinematic_chain
        converter = Joint2BVHConvertor()
        if opt.dataset_name == 'kit':
            num_joints = 21
        else:
            num_joints = 22
        joint = recover_from_ric(torch.from_numpy(motion).float(), num_joints).numpy()
        bvh_path = pjoin(bvh_dir, "%s_ik.bvh"%(name))
        _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)
        bvh_path = pjoin(final_bvh_dir, "%s.bvh" % (name))
        _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)
        save_path = pjoin(final_animation_dir, "%s.mp4"%(name))
        plot_3d_motion(save_path, kinematic_chain, joint,title=print_caption, fps=20)
        np.save(pjoin(final_joints_ik_dir, "%s.npy"%(name)), ik_joint)
        np.save(pjoin(final_joints_dir, "%s.npy"%(name)), joint)
        np.save(pjoin(final_raw_dir, "%s.npy"%(name)), motion)
def main():
    # load model
    t2m_transformer, vq_model, res_model = load_model()
    
    _edit_slice = opt.mask_edit_section
    edit_slice = []
    for eds in _edit_slice:
        _start, _end = eds.split(',')
        _start = eval(_start)
        _end = eval(_end)
        edit_slice.append([_start, _end])
        
    motion_dir = opt.source_dir
    motion_list = os.listdir(motion_dir)    
    for motion_file in motion_list:
        motion_path = pjoin(motion_dir,motion_file)
        name = motion_file.split('.')[0].strip()
        print("############################")
        print("processing:", motion_path)
        instruction_path = pjoin(opt.instruction_dir,name+'.json')
        with open(instruction_path,'r') as f:
            instruction = json.load(f)
        edit_instruction = instruction['Edit Instruction']
        motion = np.load(motion_path)
        motion_seq_len = motion.shape[0]
        count = 0
        print_caption = ''
        for i in range(2):
            # 0: left arm 1: right arm 2: lower body
            # prepare data 
            captions, m_length,edit_part, prompt,flag = preapare_data(edit_instruction,motion_seq_len,part=i)
            if flag:
                count += 1
                if i == 0:
                    print("Upper body parts don't need to be edited")
                else:
                    print("Lower body parts don't need to be edited")
                continue
            motion = process_motion(motion) # standerization, fix length, expand dimension
            print("prompt:", captions)
            with torch.no_grad():
                tokens, features = vq_model.encode(motion)
            ### build editing mask, TOEDIT marked as 1 ###
            edit_mask = torch.zeros_like(tokens[..., 0])# tokens [bs, body_parts, seq_len, q]
            seq_len = tokens.shape[2]
            for _start, _end in edit_slice:
                if isinstance(_start, float):
                    _start = int(_start*seq_len)
                    _end = int(_end*seq_len)
                else:
                    _start //= 4
                    _end //= 4
                edit_mask[:, edit_part, _start: _end] = 1
                print_caption += prompt
                print_captions = f'{print_caption} [{_start*4/20.}s - {_end*4/20.}s]'
            edit_mask = edit_mask.bool()

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
                data = inv_transform(pred_motions)[0]
            motion =  data[:m_length]
        if count < 2:
            save_reasults(motion,name,print_captions)
            if opt.is_final_round:
                print("Final Editting")
                save_results_final(motion,name,instruction["Original Body Part"]["motion"])
        else:
            if opt.final_dir:
                joint_data = np.load(motion_path)
                save_results_final(joint_data,name,instruction["Original Body Part"]["motion"])
            print("motion({}) do not need to be edited. The motion is {}".format(motion_path,instruction['Original Body Part']['motion']))
            

    
if __name__ == "__main__":
    main()