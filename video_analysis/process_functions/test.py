#!/usr/bin/env python3
# move_and_process_files.py

import os
import shutil
import argparse
import numpy as np
from pathlib import Path

def move_txt_files(source_dir, dest_dir):
    """
    移动源文件夹中的所有 .txt 文件到目标文件夹。

    :param source_dir: 源文件夹路径
    :param dest_dir: 目标文件夹路径
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # 创建目标文件夹（如果不存在）
    dest_path.mkdir(parents=True, exist_ok=True)

    txt_files = list(source_path.glob('*.txt'))

    if not txt_files:
        print(f"在 '{source_dir}' 中未找到任何 .txt 文件。")
        return

    for txt_file in txt_files:
        try:
            shutil.move(str(txt_file), str(dest_path / txt_file.name))
            print(f"已移动 '{txt_file.name}' 到 '{dest_dir}'。")
        except Exception as e:
            print(f"移动 '{txt_file.name}' 时出错: {e}")

def process_npy_files(source_dir):
    """
    处理源文件夹中的所有 .npy 文件，移除最前面的维度（如果该维度为1）。

    :param source_dir: 源文件夹路径
    """
    source_path = Path(source_dir)
    npy_files = list(source_path.glob('*.npy'))

    if not npy_files:
        print(f"在 '{source_dir}' 中未找到任何 .npy 文件。")
        return

    for npy_file in npy_files:
        try:
            array = np.load(npy_file)
            original_shape = array.shape

            if len(original_shape) == 0:
                print(f"文件 '{npy_file.name}' 中的数组为空，跳过。")
                continue

            if original_shape[0] == 1:
                new_array = np.squeeze(array, axis=0)
                np.save(npy_file, new_array)
                print(f"已处理 '{npy_file.name}': {original_shape} -> {new_array.shape}")
            else:
                print(f"文件 '{npy_file.name}' 的第一个维度大小为 {original_shape[0]}，不是1，跳过。")
        except Exception as e:
            print(f"处理 '{npy_file.name}' 时出错: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="移动当前文件夹下的所有 .txt 文件到指定文件夹，并处理 .npy 文件，移除其最前面的维度。"
    )
    parser.add_argument(
        '--source_dir',
        type=str,
        default='/extra/xielab0/shanlins/code_base/videochat/data/motiongpt',
        help='源文件夹路径（默认是当前目录）。'
    )
    parser.add_argument(
        '--dest_dir',
        type=str,
        default='/extra/xielab0/shanlins/code_base/videochat/data/motiongpt_text',
        help='目标文件夹路径，用于存放移动的 .txt 文件。'
    )

    args = parser.parse_args()

    source_dir = args.source_dir
    dest_dir = args.dest_dir

    print(f"源文件夹: {source_dir}")
    print(f"目标文件夹: {dest_dir}")
    print("\n开始移动 .txt 文件...")
    move_txt_files(source_dir, dest_dir)
    print("\n开始处理 .npy 文件...")
    process_npy_files(source_dir)
    print("\n所有操作完成。")

if __name__ == "__main__":
    main()
