import os
from os.path import join as pjoin
import json
import torch
import torch.nn.functional as F

from models.mask_transformer.transformer import MaskTransformer, TimesTransformer, ResidualTimesformer
from models.vq.model_multi import RVQVAE_Multi
from models.vq.model_general import RVQVAE_Decode
from models.vq.model import RVQVAE, LengthEstimator

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.fixseed import fixseed
from visualization.joints2bvh import Joint2BVHConvertor
from torch.distributions.categorical import Categorical


from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain

import numpy as np
clip_version = 'ViT-B/32'

def load_vq_model(opt,vq_opt,mean,std):
    dim_pose = 251 if vq_opt.dataset_name == 'kit' else 263
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')

    # vq_model = RVQVAE_Multi(vq_opt,
    #             dim_pose,
    #             vq_opt.nb_code,
    #             vq_opt.code_dim,
    #             vq_opt.code_dim,
    #             vq_opt.down_t,
    #             vq_opt.stride_t,
    #             vq_opt.width,
    #             vq_opt.depth,
    #             vq_opt.dilation_growth_rate,
    #             vq_opt.vq_act,
    #             vq_opt.vq_norm,
    #             mean,
    #             std)
    vq_model = RVQVAE_Decode(vq_opt,
            dim_pose,
            vq_opt.nb_code,
            vq_opt.code_dim,
            vq_opt.code_dim,
            vq_opt.down_t,
            vq_opt.stride_t,
            vq_opt.width,
            vq_opt.depth,
            vq_opt.dilation_growth_rate,
            vq_opt.vq_act,
            vq_opt.vq_norm,
            mean,
            std)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', opt.vq_model),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    vq_epoch = ckpt['ep'] if 'ep' in ckpt else -1
    print(f'Loading VQ Model {vq_opt.name} Completed!, Epoch {vq_epoch}')
    return vq_model, vq_opt

def load_trans_model(model_opt,opt,which_model):
    t2m_transformer = TimesTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location=opt.device)
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
    # t2m_transformer.load_state_dict(ckpt[model_key])
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Mask Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def load_res_model(res_opt,vq_opt,opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTimesformer(code_dim=vq_opt.code_dim,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            opt=res_opt)
    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_loss.tar'),map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    print(unexpected_keys)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

def load_len_estimator(opt):
    model = LengthEstimator(512, 50)
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_estimator', 'model', 'finest.tar'),
                      map_location=opt.device)
    model.load_state_dict(ckpt['estimator'])
    print(f'Loading Length Estimator from epoch {ckpt["epoch"]}!')
    return model


if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 251 if opt.dataset_name == 'kit' else 263

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    result_dir = pjoin('./generation', opt.ext)
    raw_dir = pjoin(result_dir, 'raw')
    joints_dir = pjoin(result_dir, 'joints')
    joints_ik_dir = pjoin(joints_dir,'ik')
    joints_no_ik_dir = pjoin(joints_dir,'no_ik')
    animation_dir = pjoin(result_dir, 'animations')
    bvk_dir = pjoin(result_dir, 'bvk')
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(joints_ik_dir, exist_ok=True)
    os.makedirs(joints_no_ik_dir, exist_ok=True)
    os.makedirs(animation_dir,exist_ok=True)
    os.makedirs(raw_dir,exist_ok=True)
    os.makedirs(bvk_dir,exist_ok=True)
    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)


    #######################
    ######Loading RVQ######
    #######################
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose
    mean = np.load(pjoin(opt.checkpoints_dir,opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
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

    # assert res_opt.vq_name == model_opt.vq_name

    #################################
    ######Loading M-Transformer######
    #################################
    t2m_transformer = load_trans_model(model_opt, opt, 'net_best_fid.tar')

    ##################################
    #####Loading Length Predictor#####
    ##################################
    length_estimator = load_len_estimator(model_opt)

    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()
    length_estimator.eval()

    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)
    length_estimator.to(opt.device)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
    def inv_transform(data):
        return data * std + mean

    prompt_list = []
    length_list = []
    ori_list = []
    name_list = []
    est_length = False
    if opt.text_prompt != "":
        name_list.append("0")
        temp = []
        temp.append([opt.text_prompt + opt.text_l_arm])
        temp.append([opt.text_prompt + opt.text_r_arm])
        temp.append([opt.text_prompt + opt.text_l_leg])
        temp.append([opt.text_prompt + opt.text_r_leg])
        prompt_list.append(opt.temp)
        ori_list.appned(opt.text_prompt)
        if opt.motion_length == 0:
            est_length = True
        else:
            length_list.append(opt.motion_length)
    elif opt.text_path != "":
        for prompt_name in os.listdir(opt.text_path):
            name_list.append(prompt_name.split('.')[0])
            prompt_path = os.path.join(opt.text_path,prompt_name)
            with open(prompt_path, "r") as file:
                prompt = json.load(file)
            text_prompt = prompt['prompt']
            temp = []
            temp.append([text_prompt ])
            temp.append([text_prompt])
            temp.append([text_prompt ])
            temp.append([text_prompt ])
            prompt_list.append(temp)
            ori_list.append(text_prompt)
            length_list.append(128)
        est_length = True
    else:
        raise "A text prompt, or a file a text prompts are required!!!"
    # print('loading checkpoint {}'.format(file))
    if est_length:
        print("Since no motion length are specified, we will use estimated motion lengthes!!")
        text_embedding = t2m_transformer.encode_text(ori_list)
        pred_dis = length_estimator(text_embedding)
        probs = F.softmax(pred_dis, dim=-1)  # (b, ntoken)
        token_lens = Categorical(probs).sample()  # (b, seqlen)
        #token_lens = torch.clip(token_lens,max=25) # for sequential analysis
        # lengths = torch.multinomial()
    else:
        token_lens = torch.LongTensor(length_list) // 4
        token_lens = token_lens.to(opt.device).long()
    m_length = token_lens * 4
    captions = prompt_list
    sample = 0
    kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()
    for i, caption in enumerate(captions):
        name = name_list[i]
        print("---->Sample %s: %s %d"%(name, caption, m_length[i]))
        
        with torch.no_grad():
            token_len = torch.tensor([token_lens[i]]).to(opt.device)
            mids = t2m_transformer.generate(caption, token_len,
                                        timesteps=opt.time_steps,
                                        cond_scale=opt.cond_scale,
                                        temperature=opt.temperature,
                                        topk_filter_thres=opt.topkr,
                                        gsample=opt.gumbel_sample)

            mids = res_model.generate(mids, caption, token_len, temperature=1, cond_scale=5)
            pred_motions = vq_model.forward_decoder(mids)
            pred_motions = pred_motions.detach().cpu().numpy()

            data = inv_transform(pred_motions)
            if data.shape[0] == 1:
                data = data[0]
        joint_data = data[:m_length[i]]
        np.save(pjoin(raw_dir, "{}.npy".format(name)), torch.from_numpy(joint_data).float())
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
        bvh_path = pjoin(bvk_dir, "{}_ik.bvh".format(name))
        _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)
        bvh_path = pjoin(bvk_dir, "{}.bvh".format(name))
        _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)

        save_path = pjoin(animation_dir, "%s.mp4"%(name))
        ik_save_path = pjoin(animation_dir, "%s_ik.mp4"%(name))

        plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=ori_list[i], fps=20)
        plot_3d_motion(save_path, kinematic_chain, joint, title=ori_list[i], fps=20)
        np.save(pjoin(joints_no_ik_dir, "{}.npy".format(name)), joint)
        np.save(pjoin(joints_ik_dir, "{}.npy".format(name)), ik_joint)