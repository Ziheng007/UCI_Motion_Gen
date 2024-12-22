import os
from os.path import join as pjoin
import sys
 
sys.path.append(os.path.abspath('/home/haoyum3/momask-codes-hzh'))

import torch
import torch.nn.functional as F

from utils.motion_process import recover_from_ric
# from visualization.joints2bvh import Joint2BVHConvertor  # 移除与 BVH 相关的导入

import numpy as np

# 如果有可视化需求，可以保留相关导入
# from utils.plot_script import plot_3d_motion
# from utils.paramUtil import t2m_kinematic_chain

if __name__ == '__main__':
    # 定义尝试名称
    try_name = "Sshapewalk"

    # 定义结果目录及其子目录
    result_dir = pjoin('/home/haoyum3/momask-codes-hzh/test_motions/', try_name)
    animation_dir = pjoin(result_dir, 'animations')  # 如果不需要动画，可以忽略
    video_dir = pjoin(result_dir, 'video')            # 如果不需要视频，可以忽略
    joint_path = pjoin(result_dir, 'video_npy')       # 保留用于保存 .npy 文件
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)            # 如果不需要视频，可以移除
    os.makedirs(joint_path, exist_ok=True)
    os.makedirs(animation_dir, exist_ok=True)        # 如果不需要动画，可以移除

    # 定义要加载的 .npy 文件路径
    raw_path = pjoin('/home/haoyum3/momask-codes-hzh/test_motions/', try_name)
    name = "data_structure_choice2.npy"
    file_path = pjoin(raw_path, name)

    # 加载关节数据
    joint_data = np.load(file_path, allow_pickle=True)
    print(f"Loaded joint_data shape: {joint_data.shape}")  # 输出数据形状以供验证

    # 使用 recover_from_ric 处理关节数据
    # 假设 recover_from_ric 接受 torch.Tensor 并返回 torch.Tensor
    joint_tensor = torch.from_numpy(joint_data).float()
    joint_processed = recover_from_ric(joint_tensor, 22)  # 22 是关节数量
    joint_processed_np = joint_processed.numpy()
    print(f"Processed joint data shape: {joint_processed_np.shape}")  # 输出处理后数据形状

    # 如果需要可视化，可以启用以下代码（确保 plot_3d_motion 函数可用）
    # plot_3d_motion(save_path, t2m_kinematic_chain, joint_processed_np, title=f"{try_name} Processed", fps=20)

    # 保存处理后的关节数据到 .npy 文件
    # 生成保存路径
    save_joint_path = pjoin(joint_path, f"{try_name}_processed2.npy")
    np.save(save_joint_path, joint_processed_np)
    print(f"Processed joint data saved to {save_joint_path}")

    # 如果有其他保存需求，可以在此添加
    # 例如，保存其他格式的数据或进一步处理的数据

    # 移除与 BVH 相关的所有代码部分
    # 以下部分已被移除：
    # - Joint2BVHConvertor 的初始化和使用
    # - 生成 BVH 文件路径和保存
    # - 生成和保存 MP4 动画文件

    # 如果需要可视化或其他处理，请根据需求添加相应代码
# /home/haoyum3/momask-codes-hzh/test_motions/Sshapewalk/video_npy/Sshapewalk_processed.npy