import os
from os.path import join as pjoin
import sys
 
sys.path.append(os.path.abspath('/home/haoyum3/momask-codes-hzh'))

import shutil
import numpy as np
from os.path import join as pjoin
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain
import time

def countdown(minutes):
    """Displays a countdown timer in the console."""
    total_seconds = minutes * 60
    while total_seconds > 0:
        mins, secs = divmod(total_seconds, 60)
        countdown_timer = f'{mins:02}:{secs:02}'
        print(f'\rStarting in {countdown_timer}...', end="")
        time.sleep(1)
        total_seconds -= 1
    print("\nTime's up! The program is starting...")

def process_and_compress_videos(source_dirs, output_dirs, kinematic_chain, countdown_minutes=0):
    """
    Processes .npy files in the given source directories, creates videos, 
    and compresses the output video directories into zip files.

    Args:
        source_dirs (list of str): List of directories containing .npy files.
        output_dirs (list of str): Corresponding list of directories to save videos.
        kinematic_chain (object): The kinematic chain structure for 3D plotting.
        countdown_minutes (int): Countdown time in minutes before the process starts.
    """
    # Countdown before starting the process
    if countdown_minutes > 0:
        countdown(countdown_minutes)

    print("Program is running...")

    # Create and save videos for each source directory
    for source_dir, output_dir, label in zip(source_dirs, output_dirs, ["Source", "Edit"]):
        os.makedirs(output_dir, exist_ok=True)
        for file in os.listdir(source_dir):
            if file.endswith(".npy"):
                print(f"Processing {label} - {file}")
                joint = np.load(pjoin(source_dir, file), allow_pickle=True)
                save_path = pjoin(output_dir, file.replace(".npy", ".mp4"))
                plot_3d_motion(save_path, kinematic_chain, joint, title='None', fps=20)

    # Compress directories
    for output_dir, label in zip(output_dirs, ["Source", "Edit"]):
        zip_path = f"{output_dir}.zip"
        if os.path.exists(output_dir):
            shutil.make_archive(base_name=zip_path.replace('.zip', ''), format='zip', root_dir=output_dir)
            print(f"{output_dir} has been successfully compressed to {zip_path}")
        else:
            print(f"Directory {output_dir} does not exist, unable to compress")

if __name__ == '__main__':
    kinematic_chain = t2m_kinematic_chain
    # Define paths and kinematic chain
    # source_dirs = [
    #     "/home/haoyum3/momask-codes-hzh/source_result_npy1102/",
    #     "/home/haoyum3/momask-codes-hzh/edit_result_npy1102/"
    # ]
    # output_dirs = [
    #     "/home/haoyum3/momask-codes-hzh/source_result_npy1102_video2/",
    #     "/home/haoyum3/momask-codes-hzh/edit_result_npy1102_video2/"
    # ]
    
    # source_dirs = [
    #     "/home/haoyum3/momask-codes-hzh/test_motions/walk_and_wave/video_npy/",
        
    # ]
    # output_dirs = [
    #     "/home/haoyum3/momask-codes-hzh/test_motions/walk_and_wave/video/",
 
    # ]



    # source_dirs = [
    #     "/home/haoyum3/momask-codes-hzh/test_motions/motion_change_npy/videonpy/2/",
    #     "/home/haoyum3/momask-codes-hzh/test_motions/motion_change_npy/videonpy/1/",
        
    # ]
    # output_dirs = [
    #     "/home/haoyum3/momask-codes-hzh/test_motions/motion_change_npy/video/2/",
    #     "/home/haoyum3/momask-codes-hzh/test_motions/motion_change_npy/video/1/",
 
    # ]
    
    
    # source_dirs = [
    #     "/home/haoyum3/momask-codes-hzh/test_motions/Sshapewalk/video_npy/",
        
    # ]
    # output_dirs = [
    #     "/home/haoyum3/momask-codes-hzh/test_motions/Sshapewalk/video/",
 
    # ]
    source_dirs = [
        "/home/shanlins/result_tra/trajectory_data/22_3_npy/",

    ]
    output_dirs = [
        "/home/shanlins/result_tra/trajectory_data/video/",

    ]
    # Call the function with a countdown of 8 minutes
    process_and_compress_videos(source_dirs, output_dirs, kinematic_chain, countdown_minutes=0)
