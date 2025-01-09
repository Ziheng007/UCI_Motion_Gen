import os

def delete_non_mp4_files(folder_path):
    """
    删除文件夹中所有非 .mp4 后缀的文件
    """
    if not os.path.exists(folder_path):
        print(f"路径 {folder_path} 不存在。")
        return
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path) and not filename.lower().endswith('.mp4'):
            try:
                os.remove(file_path)
                print(f"已删除文件: {file_path}")
            except Exception as e:
                print(f"删除文件失败: {file_path}, 错误: {e}")
folder_path = "/extra/xielab0/shanlins/code_base/videochat/data/1000/video"
delete_non_mp4_files(folder_path)