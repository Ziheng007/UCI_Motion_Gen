# import os
# from os.path import join as pjoin

# import torch
# import torch.nn.functional as F

# from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
# from models.vq.model import RVQVAE, LengthEstimator
# from models.vq.model_general import RVQVAE_Decode
# from models.vq.model_multi import RVQVAE_Multi
# from options.eval_option import EvalT2MOptions
# from utils.get_opt import get_opt

# from utils.fixseed import fixseed
# from visualization.joints2bvh import Joint2BVHConvertor

# from utils.motion_process import recover_from_ric
# from utils.plot_script import plot_3d_motion

# from utils.paramUtil import t2m_kinematic_chain,kit_kinematic_chain
# from utils.kit_utils import KIT_UPPER_JOINT_Y_MASK,KIT_LEFT_ARM_MASK,KIT_RIGHT_ARM_MASK,KIT_LEFT_LEG_MASK,KIT_RIGHT_LEG_MASK,KIT_OVER_LAP_UPPER_MASK,KIT_OVER_LAP_LOWER_MASK,KIT_OVER_LAP_UPPER_MASK
# import numpy as np

# from gen_t2m_time import load_res_model, load_trans_model
# from scipy.ndimage import gaussian_filter

# def motion_temporal_filter(motion, sigma=1):
#     motion = motion.reshape(motion.shape[0], -1)
#     # print(motion.shape)
#     for i in range(motion.shape[1]):
#         motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
#     return motion.reshape(motion.shape[0], -1, 3)

# def load_vq_model(vq_opt,mean,std):
#     dim_pose = 251 if vq_opt.dataset_name == 'kit' else 263
#     # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')

#     vq_model = RVQVAE_Multi(vq_opt,
#                 dim_pose,
#                 vq_opt.nb_code,
#                 vq_opt.code_dim,
#                 vq_opt.code_dim,
#                 vq_opt.down_t,
#                 vq_opt.stride_t,
#                 vq_opt.width,
#                 vq_opt.depth,
#                 vq_opt.dilation_growth_rate,
#                 vq_opt.vq_act,
#                 vq_opt.vq_norm,
#                 mean,
#                 std)
#     # vq_model = RVQVAE_Decode(vq_opt,
#     #         dim_pose,
#     #         vq_opt.nb_code,
#     #         vq_opt.code_dim,
#     #         vq_opt.code_dim,
#     #         vq_opt.down_t,
#     #         vq_opt.stride_t,
#     #         vq_opt.width,
#     #         vq_opt.depth,
#     #         vq_opt.dilation_growth_rate,
#     #         vq_opt.vq_act,
#     #         vq_opt.vq_norm,
#     #         mean,
#     #         std)
#     ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),map_location='cpu')
#     model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
#     vq_model.load_state_dict(ckpt[model_key])
#     vq_epoch = ckpt['ep'] if 'ep' in ckpt else -1
#     print(f'Loading VQ Model {vq_opt.name} Completed!, Epoch {vq_epoch}')
#     return vq_model, vq_opt

# if __name__ == '__main__':
#     parser = EvalT2MOptions()
#     opt = parser.parse()
#     fixseed(opt.seed)

#     opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
#     torch.autograd.set_detect_anomaly(True)

#     dim_pose = 251 if opt.dataset_name == 'kit' else 263

#     root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
#     model_dir = pjoin(root_dir, 'model')
#     result_dir = pjoin('./editing', opt.ext)
#     joints_dir = pjoin(result_dir, 'joints')
#     animation_dir = pjoin(result_dir, 'animations')
#     os.makedirs(joints_dir, exist_ok=True)
#     os.makedirs(animation_dir,exist_ok=True)

#     model_opt_path = pjoin(root_dir, 'opt.txt')
#     model_opt = get_opt(model_opt_path, device=opt.device)

#     #######################
#     ######Loading RVQ######
#     #######################
#     model_opt.vq_name = 'rvq_multi_kit'
#     mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
#     std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
#     vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
#     vq_opt = get_opt(vq_opt_path, device=opt.device)
#     vq_opt.dim_pose = dim_pose
#     vq_opt.name = 'rvq_multi_kit'
#     vq_model, vq_opt = load_vq_model(vq_opt,mean,std)

#     # model_opt.num_tokens = vq_opt.nb_code
#     # model_opt.num_quantizers = vq_opt.num_quantizers
#     # model_opt.code_dim = vq_opt.code_dim

#     #################################
#     ######Loading R-Transformer######
#     #################################
#     # res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
#     # res_opt = get_opt(res_opt_path, device=opt.device)
#     # res_model = load_res_model(res_opt, vq_opt, opt)

#     # print('res_opt.vq_name', res_opt.vq_name)
#     # print('model_opt.vq_name', model_opt.vq_name)
#     # assert res_opt.vq_name == model_opt.vq_name

#     #################################
#     ######Loading M-Transformer######
#     #################################
#     # t2m_transformer = load_trans_model(model_opt, opt, 'net_best_top1.tar')

#     # t2m_transformer.eval()
#     vq_model.eval()
#     # res_model.eval()

#     # res_model.to(opt.device)
#     # t2m_transformer.to(opt.device)
#     vq_model.to(opt.device)

#     ##### ---- Data ---- #####
#     max_motion_length = 196
#     def inv_transform(data):
#         return data * std + mean
#     ### We provided an example source motion (from 'new_joint_vecs') for editing. See './example_data/000612.mp4'###
#     motion1 = np.load('/extra/xielab0/araujog/motion-generation/KIT-ML/new_joint_vecs/03508.npy')
#     motion2 = np.load('/home/haoyum3/momask-codes/generation/kit/01979.npy')

#     m_length = len(motion1)
#     motion1 = (motion1 - mean) / std
#     if max_motion_length > m_length:
#         motion1 = np.concatenate([motion1, np.zeros((max_motion_length - m_length, motion1.shape[1])) ], axis=0)
#     motion1 = torch.from_numpy(motion1)[None].to(opt.device)

#     m_length = len(motion2)
#     motion2 = (motion2 - mean) / std
#     if max_motion_length > m_length:
#         motion2 = np.concatenate([motion2, np.zeros((max_motion_length - m_length, motion2.shape[1])) ], axis=0)
#     motion2 = torch.from_numpy(motion2)[None].to(opt.device)

#     prompt_list = []
#     length_list = []
#     if opt.motion_length == 0:
#         opt.motion_length = m_length
#         print("Using default motion length.")
    
#     prompt_list.append(opt.text_l_arm)
#     prompt_list.append(opt.text_r_arm)
#     prompt_list.append(opt.text_l_leg)
#     prompt_list.append(opt.text_r_leg)
#     length_list.append(opt.motion_length)
#     if opt.text_prompt == "":
#         raise "Using an empty text prompt."

#     token_lens = torch.LongTensor(length_list) // 4
#     token_lens = token_lens.to(opt.device).long()

#     m_length = token_lens * 4
#     captions = prompt_list
#     print_captions = captions[0]
    
#     _edit_slice = opt.mask_edit_section
#     edit_slice = []
#     for eds in _edit_slice:
#         _start, _end = eds.split(',')
#         _start = eval(_start)
#         _end = eval(_end)
#         edit_slice.append([_start, _end])

#     sample = 0
#     if opt.dataset_name == 'kit':
#         kinematic_chain = kit_kinematic_chain
#         print('Using kit kinematic chain')
#     else:
#         kinematic_chain = t2m_kinematic_chain
#     converter = Joint2BVHConvertor()

#     with torch.no_grad():
#         tokens, features = vq_model.encode(motion1)
#     ### build editing mask, TOEDIT marked as 1 ###
#     edit_mask = torch.zeros_like(tokens[..., 0])# tokens [bs, body_parts, seq_len, q]
#     seq_len = tokens.shape[2]
#     if 4 in opt.edit_part:
#         edit_part = [0, 1, 2, 3]
#         print('Edit whole body')
#     else:
#         edit_part = opt.edit_part
#         body_parts = ['left arm','right_arm','left_leg','right_leg']
#         print('Edit %s'%[body_parts[i] for i in edit_part])
#     for _start, _end in edit_slice:
#         if isinstance(_start, float):
#             _start = int(_start*seq_len)
#             _end = int(_end*seq_len)
#         else:
#             _start //= 4
#             _end //= 4
#         edit_mask[:, edit_part, _start: _end] = 1
#         print_captions = f'{print_captions} [{_start*4/20.}s - {_end*4/20.}s]'
#     edit_mask = edit_mask.bool()
#     for r in range(opt.repeat_times):
#         print("-->Repeat %d"%r)
#         with torch.no_grad():
#             # mids = tokens[..., 0]
#             # mids = t2m_transformer.edit(
#             #                             captions, tokens[..., 0].clone(), m_length//4,
#             #                             timesteps=opt.time_steps,
#             #                             cond_scale=opt.cond_scale,
#             #                             temperature=opt.temperature,
#             #                             topk_filter_thres=opt.topkr,
#             #                             gsample=opt.gumbel_sample,
#             #                             force_mask=opt.force_mask,
#             #                             edit_mask=edit_mask.clone(),
#             #                             edit_parts=edit_part,
#             #                             )
#             # mids1, all_codes = vq_model.encode(motion1)
#             # mids2, all_codes = vq_model.encode(motion2)
#             # mids2[:,2:,:,:] = mids1[:,2:,:,:]
#             # if opt.use_res_model:
#             #     print('Using residual transformer')
#             #     mids = res_model.generate(mids, captions, m_length//4, temperature=1, cond_scale=2)
#             # else:
#             #     mids.unsqueeze_(-1)
#             # pred_motions = vq_model.forward_decoder(mids)
#             # pred_motions = vq_model.forward_decoder(mids2)
#             pred_motions = motion2
#             pred_motions[:,:,KIT_LEFT_ARM_MASK] = motion1[:,:,KIT_LEFT_ARM_MASK]
#             pred_motions[:,:,KIT_RIGHT_ARM_MASK] = motion1[:,:,KIT_RIGHT_ARM_MASK]
#             pred_motions = pred_motions.detach().cpu().numpy()
#             print(pred_motions.shape)
#             source_motions = motion1.detach().cpu().numpy()

#             data = inv_transform(pred_motions)
#             source_data = inv_transform(source_motions)

#         for k, (caption, joint_data, source_data)  in enumerate(zip(captions, data, source_data)):
#             print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
#             animation_path = pjoin(animation_dir, str(k))
#             joint_path = pjoin(joints_dir, str(k))

#             os.makedirs(animation_path, exist_ok=True)
#             os.makedirs(joint_path, exist_ok=True)

#             if opt.dataset_name == 'kit':
#                 num_joints = 21
#             else:
#                 num_joints = 22

#             joint_data = joint_data[:m_length[k]]
#             joint = recover_from_ric(torch.from_numpy(joint_data).float(), num_joints).numpy()

#             source_data = source_data[:m_length[k]]
#             soucre_joint = recover_from_ric(torch.from_numpy(source_data).float(), num_joints).numpy()
#             soucre_joint = motion_temporal_filter(soucre_joint, sigma=1)
#             joint = motion_temporal_filter(joint, sigma=1)

#             bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.bvh"%(k, r, m_length[k]))
#             # _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)

#             bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d.bvh" % (k, r, m_length[k]))
#             # _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)


#             save_path = pjoin(animation_path, "sample%d_repeat%d_len%d.mp4"%(k, r, m_length[k]))
#             ik_save_path = pjoin(animation_path, "sample%d_repeat%d_len%d_ik.mp4"%(k, r, m_length[k]))
#             source_save_path = pjoin(animation_path, "sample%d_source_len%d.mp4"%(k, m_length[k]))

#             plot_3d_motion(save_path, kinematic_chain, joint, title=print_captions, fps=12.5, radius=240 * 8)
#             plot_3d_motion(source_save_path, kinematic_chain, soucre_joint, title='None', fps=12.5, radius=240 * 8)
#             np.save(pjoin(joint_path, "sample%d_repeat%d_len%d.npy"%(k, r, m_length[k])), joint)
#             np.save(pjoin(joint_path, "source%d_repeat%d_len%d_ik.npy"%(k, r, m_length[k])), soucre_joint)
#             print('Npy saved %s'%joint_path)

import torch

# 示例 scores 张量
scores = torch.tensor([[[0.2, 0.5, 0.1],
                        [0.4, 0.3, 0.6]],
                       [[0.8, 0.7, 0.9],
                        [0.1, 0.4, 0.2]]])

# 对 scores 的第三个维度进行排序
sorted_indices = scores.argsort(dim=2)
print("Sorted Indices:\n", sorted_indices)

# 计算排名
ranks = sorted_indices.argsort(dim=2)
print("Ranks:\n", ranks)