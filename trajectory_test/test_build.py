import os
from os.path import join as pjoin
import sys
 

 
sys.path.append(os.path.abspath('/home/haoyum3/momask-codes-hzh'))

import torch
import torch.nn.functional as F

 
from visualization.joints2bvh import Joint2BVHConvertor
 
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain

import numpy as np
            
            
            
 
if __name__ == '__main__':
    # Initialize the converter
    converter = Joint2BVHConvertor()
    
    # Define names and paths
    # try_name = "walk_and_wave"
    try_name = "heart"
    result_dir = pjoin('/home/haoyum3/momask-codes-hzh/trajectory_try/heart/', try_name)
    animation_dir = pjoin(result_dir, 'animations')
    video_dir = pjoin(result_dir, 'video')
    joint_path = pjoin(result_dir, 'video_npy')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(joint_path, exist_ok=True)
    os.makedirs(animation_dir, exist_ok=True)
    
    # raw_path = pjoin('/home/haoyum3/momask-codes-hzh/test_motions/', try_name, 'raw')
    # name = "raw_sample0_repeat0_len128.npy"
    # file_path = pjoin(raw_path, name)
    
    raw_path = result_dir
    
    name = try_name
    file_path = pjoin(raw_path, name)
    
    # Load joint data
    joint_data = np.load(file_path, allow_pickle=True)
    
    # Recover joint positions
    joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
    
    # Convert to BVH with inverse kinematics
    bvh_path_ik = pjoin(animation_dir, f"{try_name}_ik.bvh")
    _, ik_joint = converter.convert(joint, filename=bvh_path_ik, iterations=100)
    
    # Convert to BVH without inverse kinematics
    bvh_path = pjoin(animation_dir, f"{try_name}.bvh")
    _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)
    
    # Define paths for saving MP4 animations
    save_path = pjoin(video_dir, f"{try_name}.mp4")
    ik_save_path = pjoin(video_dir, f"{try_name}_ik.mp4")
    
    # Plot and save the 3D motions (uncomment if the functions are available)
    # plot_3d_motion(ik_save_path, kinematic_chain, ik_joint, title=caption, fps=20)
    # plot_3d_motion(save_path, kinematic_chain, joint, title=caption, fps=20)
    
    # Save the joint data as .npy files

    np.save(pjoin(joint_path, f"{try_name}_joint.npy"), joint)
    np.save(pjoin(joint_path, f"{try_name}_ik_joint.npy"), ik_joint)
