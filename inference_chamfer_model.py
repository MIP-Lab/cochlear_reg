import os
from dataset.cochlear_ct import CochlearCTCrop, read_mesh_file, write_mesh_file
from model.kcl_networks import LocalNet
from utils.pycip_utils import to_pyvista
from utils.training.torch_trainer import Trainer
from torch.utils.data import DataLoader
import torch
from utils.transform import SpatialTransformer, trilinear_interpolation3_torch
import random
import numpy as np
import pickle
import pyvista as pv

_, MD_triangles = read_mesh_file('data/atlas/atlas_MD.mesh')
_, ST_triangles = read_mesh_file('data/atlas/atlas_ST.mesh')
_, SV_triangles = read_mesh_file('data/atlas/atlas_SV.mesh')
_, CC_triangles = read_mesh_file('data/atlas/atlas_CC.mesh')

class MyTrainer(Trainer):

    def test_loss(self, model, input_data):

        x = torch.cat([input_data['vol'], input_data['atlas']], dim=1)
        # x0 = torch.zeros_like(input_data['vol'])
        # atlas_activation = model.dp1_conv1(torch.cat([x0, input_data['atlas']], dim=1))

        # print(x.mean())
        ddf = model(x)
        trans = SpatialTransformer(inshape)
        y_source = trans(input_data['vol'], ddf)

        atlas_vtx_sample = input_data['atlas_vtx']

        vtx_pred = trilinear_interpolation3_torch(atlas_vtx_sample, ddf)
        vtx_pred = vtx_pred.detach().cpu().numpy()

        MD_pred = vtx_pred[0][: 17947, :]
        ST_pred = vtx_pred[0][17947: 21291, :]
        SV_pred = vtx_pred[0][21291: 24423, :]
        CC_pred = vtx_pred[0][24423: , :]

        case = test_ds.cases[int(input_data['index'][0].detach().cpu().numpy())]

        if not os.path.exists(f'{self.exp_dir}/evaluation/predicted_mesh'):
            os.mkdir(f'{self.exp_dir}/evaluation/predicted_mesh')
            os.mkdir(f'{self.exp_dir}/evaluation/predicted_mesh/MD')
            os.mkdir(f'{self.exp_dir}/evaluation/predicted_mesh/ST')
            os.mkdir(f'{self.exp_dir}/evaluation/predicted_mesh/SV')
            os.mkdir(f'{self.exp_dir}/evaluation/predicted_mesh/CC')
        
        MD_mesh = to_pyvista(MD_pred, MD_triangles)
        ST_mesh = to_pyvista(ST_pred, ST_triangles)
        SV_mesh = to_pyvista(SV_pred, SV_triangles)
        CC_mesh = to_pyvista(CC_pred, CC_triangles)

        MD_mesh.save(f'{self.exp_dir}/evaluation/predicted_mesh/MD/{case}.ply')
        ST_mesh.save(f'{self.exp_dir}/evaluation/predicted_mesh/ST/{case}.ply')
        SV_mesh.save(f'{self.exp_dir}/evaluation/predicted_mesh/SV/{case}.ply')
        CC_mesh.save(f'{self.exp_dir}/evaluation/predicted_mesh/CC/{case}.ply')
        
        return {'ddf': ddf, 'warped': y_source}, {'loss': 0}

random.seed(2023)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False

inshape = [128, 128, 128]  
model = LocalNet(inshape[0])

test_ds = CochlearCTCrop(nii_path='data/images', cases=['PID_1689_PLANID_1096_LEFT'], 
                         vtx_path=None, atlas_case_name='default', aug=False)
test_loader = DataLoader(dataset=test_ds, batch_size=1, num_workers=0, shuffle=False)

trainer = MyTrainer(save_dir='checkpoints', name='Exp_task16-kcl-chamfer-reg4p0-post_1')
trainer.evaluate(model, exp_id=1, epoch='best', test_loader=test_loader, device='cuda', tag='postONpost_write_predicted_mesh')


print(1)




