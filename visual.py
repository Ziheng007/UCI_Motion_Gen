import os
from os.path import join as pjoin
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain
from utils.paramUtil import kit_kinematic_chain
import numpy as np

if __name__ == '__main__':
        path ='/home/haoyum3/momask-codes/editing/kit/joints/0'
        if 'kit' in path:
            kinematic_chain = kit_kinematic_chain
        else:
            kinematic_chain = t2m_kinematic_chain
        for file in os.listdir(path):
            if file.endswith(".npy"):
                joint = np.load(pjoin(path, file), allow_pickle=True)
                save_path = pjoin(path, file.replace(".npy", ".mp4"))
                plot_3d_motion(save_path, kinematic_chain, joint, title='None', fps=20)