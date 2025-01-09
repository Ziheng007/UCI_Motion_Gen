from models.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
import torch
from visualize.simplify_loc2rot import joints2smpl

class npy2obj:
    def __init__(self, npy_path, device=0, cuda=True):
        self.npy_path = npy_path
        self.motions = np.load(self.npy_path)
        self.rot2xyz = Rotation2xyz(device='cpu')
        self.faces = self.rot2xyz.smpl_model.faces
        assert len(self.motions.shape) == 3
        self.nframes ,self.njoints, self.nfeats=self.motions.shape
        self.num_frames = self.nframes
        self.opt_cache = {}
        self.j2s = joints2smpl(num_frames=self.num_frames, device_id=device, cuda=cuda)
        if self.nfeats == 3:
            print(f'Running SMPLify For sample, it may take a few minutes.')
            motion_tensor, opt_dict = self.j2s.joint2smpl(self.motions)  # [nframes, njoints, 3]
            self.motions = motion_tensor.cpu().numpy()
        elif self.nfeats == 6:
            self.motions = self.motions
        self.bs, self.njoints, self.nfeats, self.nframes = self.motions.shape

        self.vertices = self.rot2xyz(torch.tensor(self.motions), mask=None,
                                     pose_rep='rot6d', translation=True, glob=True,
                                     jointstype='vertices',
                                     # jointstype='smpl',  # for joint locations
                                     vertstrans=True)
        self.vertices = np.transpose(self.vertices[0],(2,0,1))
        self.root_loc = self.motions[:, -1, :3, :].reshape(1, 1, 3, -1)
        # self.vertices += self.root_loc

    def get_vertices(self, sample_i, frame_i):
        return self.vertices[sample_i, :, :, frame_i].squeeze().tolist()

    def get_trimesh(self, sample_i, frame_i):
        return Trimesh(vertices=self.get_vertices(sample_i, frame_i),
                       faces=self.faces)

    def save_obj(self, save_path, frame_i):
        mesh = self.get_trimesh(0, frame_i)
        with open(save_path, 'w') as fw:
            mesh.export(fw, 'obj')
        return save_path
    
    def save_npy(self, save_path):
        np.save(save_path, self.vertices)