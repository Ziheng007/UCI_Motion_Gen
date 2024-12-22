import numpy as np
from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric
from visualization.joints2bvh import Joint2BVHConvertor
from utils.paramUtil import t2m_kinematic_chain,kit_kinematic_chain
import torch
from os.path import join as pjoin
import os


dir = '/red/ruogu.fang/shanlinsun/data/GEM/generation/171'
for i in os.listdir(dir):
    print(i)
    name = i.split('.')[0].strip()
    path = pjoin(dir,i)
    joint_data = np.load(path)
    print(joint_data.shape)
    kinematic_chain = t2m_kinematic_chain
    num_joints = 22
    converter = Joint2BVHConvertor()
    joint = recover_from_ric(torch.from_numpy(joint_data).float(), num_joints).numpy()
    bvh_path = pjoin('/red/ruogu.fang/shanlinsun/data/GEM/generation/run/bvh', "%s_ik.bvh"%(name))
    _, ik_joint = converter.convert(joint, filename=bvh_path, iterations=100)
    bvh_path = pjoin('/red/ruogu.fang/shanlinsun/data/GEM/generation/run/bvh', "%s.bvh" % (name))
    _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)
    save_path =pjoin("/red/ruogu.fang/shanlinsun/data/GEM/generation/run/animations", "%s.mp4"%(name))
    plot_3d_motion(save_path, kinematic_chain, joint, title='None', fps=20)
    np.save(pjoin('/red/ruogu.fang/shanlinsun/data/GEM/generation/run/motion', "%s.npy"%(name)), ik_joint)