from torch.utils.data import Dataset
import glob
import os
import nibabel as nib
import numpy as np
from monai.transforms import RandRotate, GaussianSmooth, RandGaussianSmooth
from scipy.spatial.transform import Rotation
import time
import pyvista as pv

def read_mesh_file(file):
    with open(file, 'rb') as f:
        mid = np.fromfile(f, 'int32', 1)[0]
        numverts = np.fromfile(f, 'int32', 1)[0]
        numtris = np.fromfile(f, 'int32', 1)[0]
        n = np.fromfile(f, 'int32', 1)[0]
        if n == -1:
            orient = np.fromfile(f, 'int32', 3)
            dim = np.fromfile(f, 'int32', 3)
            sz = np.fromfile(f, 'float32', 3)
            color = np.fromfile(f, 'int32', 3)
        else:
            color = np.fromfile(f, 'int32', 2)
        vertices = np.fromfile(f, 'float32', numverts * 3)
        vertices = vertices.reshape((-1, 3))
        triangles = np.fromfile(f, 'int32', numtris * 3)
        triangles = triangles.reshape((-1, 3))
        return vertices, triangles

def write_mesh_file(vertices, triangles, file):
    data = [0, vertices.shape[0], triangles.shape[0], 255, 0, 0]
    with open(file, 'wb') as f:
        f.write(np.array(data, dtype='int32').tobytes())
        f.write(vertices.flatten().astype(np.float32).tobytes())
        f.write(triangles.flatten().astype(np.int32).tobytes())
    return 0

def mesh_to_pyvista(vertices, triangles):
    faces_4d = np.zeros((triangles.shape[0], 4))
    faces_4d[:, 1:4] = triangles
    faces_4d[:, 0] = 3
    faces_flatten = faces_4d.flatten().astype(np.int32)
    mesh = pv.PolyData(vertices, faces_flatten)
    return mesh

class CochlearCTCrop(Dataset):

    def __init__(self, data_root, aug=False) -> None:
        super().__init__()
        self.data_root = data_root
        self.vol_names = glob.glob(os.path.join(data_root + '/images', '*'))
        self.vtx_names = [x.replace('images', 'gt_mesh_points').replace('.nii.gz', '.pkl') for x in self.vol_names]
        self.atlas, self.atlas_affine, self.atlas_vtx = self.load_atlas()
        self.crop_size = self.atlas.shape
        self.aug = aug
    
    def __getitem__(self, index):
        postop_ct = nib.load(self.vol_names[index]).get_fdata().astype(np.float32)
        vtx = np.load(open(self.vtx_names[index], 'rb'), allow_pickle=True)
        if self.aug:
            [postop_ct], vtx = self.do_augmentation([postop_ct], vtx)
        # add channel dim
        postop_ct, atlas = postop_ct[None, :], self.atlas[None, :]

        return {'vol': postop_ct, 'vtx': vtx, 'atlas': atlas.copy(), 'atlas_vtx': self.atlas_vtx.copy(), 'index': index}

    def load_atlas(self):
        atlas_nii = nib.load('%s/atlas/atlas.nii.gz' % self.data_root)
        atlas, atlas_affine = atlas_nii.get_fdata().astype(np.float32), atlas_nii.affine
        atlas = 2 * (atlas - atlas.min()) / (atlas.max() - atlas.min()) - 1
        atlas_vtx = np.load(open('%s/atlas/atlas_all_vtx.pkl' % self.data_root, 'rb'), allow_pickle=True)
        return atlas, atlas_affine, atlas_vtx

    def do_augmentation(self, images, points):
        theta = np.pi * 15 * (2 * np.random.uniform(0, 1) - 1) / 180 # -15 to 15 degrees
        rot_axis = ['x', 'y', 'z'][np.random.randint(0, 3)]
        params = {'x': 0, 'y': 0, 'z': 0}
        params[rot_axis] = (theta, theta)
        rot_matrix = Rotation.from_euler(rot_axis, theta).as_matrix()
        
        images = [image[None, :, :, :] for image in images]

        # 4D input (c * H*W*D)
        T = RandRotate(range_x=params['x'], range_y=params['y'], range_z=params['z'], prob=1, padding_mode='zeros')
        images = [T(image) for image in images]

        center = self.crop_size[0] / 2 - 1
        points -= center
        points = points @ rot_matrix
        points += center

        images = [image[0] for image in images]
        # print(theta, sigma, rot_axis)
        return images, points


    def __len__(self):
        return len(self.vol_names)