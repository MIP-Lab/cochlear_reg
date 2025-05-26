import os
from dataset.cochlear_ct import CochlearCTCrop, read_mesh_file, write_mesh_file
from model.segnet import SharedSkipUNet
from utils.training.torch_trainer import Trainer
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
from skimage import measure

_, MD_triangles = read_mesh_file('data/atlas/atlas_MD.mesh')
_, ST_triangles = read_mesh_file('data/atlas/atlas_ST.mesh')
_, SV_triangles = read_mesh_file('data/atlas/atlas_SV.mesh')
_, CC_triangles = read_mesh_file('data/atlas/atlas_CC.mesh')

class MyTrainer(Trainer):

    def test_loss(self, model, input_data):
        
        pred = model(input_data['vol'])
        
        md_mask = (torch.sigmoid(pred[0, 0]).detach().cpu().numpy() > 0.5).astype(np.uint8)
        st_mask = (torch.sigmoid(pred[0, 1]).detach().cpu().numpy() > 0.5).astype(np.uint8)
        sv_mask = (torch.sigmoid(pred[0, 2]).detach().cpu().numpy() > 0.5).astype(np.uint8)
        cc_mask = (torch.sigmoid(pred[0, 3]).detach().cpu().numpy() > 0.5).astype(np.uint8)
        
        md_verts, md_faces, _, _ = measure.marching_cubes(md_mask, 0)
        st_verts, st_faces, _, _ = measure.marching_cubes(st_mask, 0)
        sv_verts, sv_faces, _, _ = measure.marching_cubes(sv_mask, 0)
        cc_verts, cc_faces, _, _ = measure.marching_cubes(cc_mask, 0)
        
        case_path = test_ds.vol_names[int(input_data['index'][0].detach().cpu().numpy())]
        case = case_path.split('\\')[-1].replace('.nii.gz', '')

        if not os.path.exists(f'{self.exp_dir}/evaluation/predicted_mesh'):
            os.mkdir(f'{self.exp_dir}/evaluation/predicted_mesh')
            os.mkdir(f'{self.exp_dir}/evaluation/predicted_mesh/MD')
            os.mkdir(f'{self.exp_dir}/evaluation/predicted_mesh/ST')
            os.mkdir(f'{self.exp_dir}/evaluation/predicted_mesh/SV')
            os.mkdir(f'{self.exp_dir}/evaluation/predicted_mesh/CC')
        
        write_mesh_file(md_verts, MD_triangles, f'{self.exp_dir}/evaluation/predicted_mesh/MD/{case}.mesh')
        write_mesh_file(st_verts, ST_triangles, f'{self.exp_dir}/evaluation/predicted_mesh/ST/{case}.mesh')
        write_mesh_file(sv_verts, SV_triangles, f'{self.exp_dir}/evaluation/predicted_mesh/SV/{case}.mesh')
        write_mesh_file(cc_verts, CC_triangles, f'{self.exp_dir}/evaluation/predicted_mesh/CC/{case}.mesh')
        
        return {}, {'loss': 0}

random.seed(2023)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False

model = SharedSkipUNet(n_labels=4)

test_ds = CochlearCTCrop(data_root='data', aug=False)
test_loader = DataLoader(dataset=test_ds, batch_size=1, num_workers=0, shuffle=False)

trainer = MyTrainer(save_dir='checkpoints', name='Exp_shared_skip_unet_cochlear_post_1')
trainer.evaluate(model, exp_id=1, epoch='best', test_loader=test_loader, device='cuda', tag='postONpost_write_predicted_mesh')


print(1)




