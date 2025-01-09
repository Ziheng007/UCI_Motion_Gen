#!/usr/bin/env python3
# rename_files_preview.py

import os
import re
import sys
import argparse

def clean_filename(name):
    """
    去除文件名开头和结尾的非字母符号。
    """
    # 去除开头的非字母字符
    name = re.sub(r'^[^A-Za-z]+', '', name)
    # 去除结尾的非字母字符
    name = re.sub(r'[^A-Za-z]+$', '', name)
    return name

def preview_rename(directory, dry_run=True):
    """
    预览重命名操作。如果 dry_run 为 False，则实际执行重命名。
    """
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # 仅处理文件，跳过子目录
        if os.path.isfile(filepath):
            base, ext = os.path.splitext(filename)
            new_base = clean_filename(base)
            
            # 检查是否需要重命名
            if new_base != base:
                new_filename = f"{new_base}{ext}"
                new_filepath = os.path.join(directory, new_filename)
                
                if dry_run:
                    print(f"将 '{filename}' 重命名为 '{new_filename}'")
                else:
                    # 检查目标文件是否已存在，避免覆盖
                    if os.path.exists(new_filepath):
                        print(f"跳过重命名 '{filename}'，因为目标文件 '{new_filename}' 已存在。")
                    else:
                        try:
                            os.rename(filepath, new_filepath)
                            print(f"已将 '{filename}' 重命名为 '{new_filename}'")
                        except Exception as e:
                            print(f"重命名 '{filename}' 失败：{e}")
            else:
                print(f"文件名 '{filename}' 无需更改。")

def main():
    """
    主函数，处理命令行参数并执行预览或实际重命名操作。
    """
    
    directory = "/extra/xielab0/shanlins/code_base/videochat/data/finemogen_new"
    perform_rename = True
    
    if not os.path.isdir(directory):
        print(f"错误: '{directory}' 不是一个有效的目录。")
        sys.exit(1)
    
    if perform_rename:
        print("实际执行重命名操作：")
    else:
        print("预览重命名结果：")
    
    preview_rename(directory, dry_run=not perform_rename)

if __name__ == "__main__":
    main()
