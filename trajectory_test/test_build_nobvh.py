# import os
# from os.path import join as pjoin
# import sys
 
# sys.path.append(os.path.abspath('/home/haoyum3/momask-codes-hzh'))

# import numpy as np
# import torch
# from utils.motion_process import recover_from_ric
# import zipfile


# import torch
# import torch.nn.functional as F

# from utils.motion_process import recover_from_ric
# # from visualization.joints2bvh import Joint2BVHConvertor  # 移除与 BVH 相关的导入

# import numpy as np

# # 如果有可视化需求，可以保留相关导入
# # from utils.plot_script import plot_3d_motion
# # from utils.paramUtil import t2m_kinematic_chain

# def process_motion_files(input_dir, output_dir, joint_count=22):
#     """
#     处理指定目录下的所有 .npy 文件并保存处理后的结果到输出目录。

#     :param input_dir: 输入目录，包含 .npy 文件
#     :param output_dir: 输出目录，保存处理后的 .npy 文件
#     :param joint_count: 关节数量，默认为 22
#     :return: 无
#     """
#     # 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)

#     # 获取输入目录下的所有 .npy 文件
#     files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

#     # 处理每一个文件
#     for file_name in files:
#         print(f"Processing {file_name}")
#         input_file_path = pjoin(input_dir, file_name)

#         # 加载关节数据
#         joint_data = np.load(input_file_path, allow_pickle=True)
#         print(f"Loaded {file_name} shape: {joint_data.shape}")

#         # 使用 recover_from_ric 处理关节数据
#         joint_tensor = torch.from_numpy(joint_data).float()
#         joint_processed = recover_from_ric(joint_tensor, joint_count)
#         joint_processed_np = joint_processed.numpy()
#         print(f"Processed {file_name} shape: {joint_processed_np.shape}")

#         # 生成保存路径并保存处理后的数据
#         output_file_path = pjoin(output_dir, f"{os.path.splitext(file_name)[0]}_processed.npy")
#         np.save(output_file_path, joint_processed_np)
#         print(f"Processed data saved to {output_file_path}")

# def zip_directory(output_dir, zip_filename):
#     """
#     将目录压缩成 zip 文件。

#     :param output_dir: 要压缩的目录
#     :param zip_filename: 压缩后的 .zip 文件名
#     :return: 无
#     """
#     with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
#         # 遍历目录并添加所有文件到 zip
#         for root, dirs, files in os.walk(output_dir):
#             for file in files:
#                 file_path = pjoin(root, file)
#                 arcname = os.path.relpath(file_path, start=output_dir)  # 只保存相对路径
#                 zipf.write(file_path, arcname=arcname)
#         print(f"Directory {output_dir} has been zipped into {zip_filename}")

# if __name__ == "__main__":
#     # 定义输入和输出目录
#     input_dir_1 = '/home/haoyum3/momask-codes-hzh/test_motions/motion_change_npy/1'
#     output_dir_1 = '/home/haoyum3/momask-codes-hzh/test_motions/motion_change_npy/videonpy/1'

#     input_dir_2 = '/home/haoyum3/momask-codes-hzh/test_motions/motion_change_npy/2'
#     output_dir_2 = '/home/haoyum3/momask-codes-hzh/test_motions/motion_change_npy/videonpy/2'

#     # 处理第一个目录下的所有文件
#     process_motion_files(input_dir_1, output_dir_1)

#     # 处理第二个目录下的所有文件
#     process_motion_files(input_dir_2, output_dir_2)

#     # # 压缩视频目录为 zip 文件
#     # zip_directory(output_dir_1, '/home/haoyum3/momask-codes-hzh/test_motions/motion_change_npy/video_1.zip')
#     # zip_directory(output_dir_2, '/home/haoyum3/momask-codes-hzh/test_motions/motion_change_npy/video_2.zip')
