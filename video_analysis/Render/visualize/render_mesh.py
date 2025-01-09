import os
os.environ['PYOPENGL_PLATFORM'] = "osmesa"
from tqdm import tqdm
import numpy as np
from visualize import render_utils

def render2mesh(motion_path, outdir='./data/mesh', device_id=0):
    print("Using cuda:",device_id[0])

    motion_name = os.path.basename(motion_path).replace('.npy','_smpl.npy')
    smpl_path = os.path.join(outdir,motion_name)
    if not os.path.exists(smpl_path):
        npy2obj = render_utils.npy2obj(motion_path,device=device_id[0])
        print('Saving SMPL params to [{}]'.format(os.path.abspath(smpl_path)))
        npy2obj.save_npy(smpl_path)
    vertices = np.load(smpl_path)
    return vertices
