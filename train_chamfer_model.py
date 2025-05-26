import os

from dataset.cochlear_ct import CochlearCTCrop
from model.kcl_networks import LocalNet, LocalNetNoBN
from model.loss import pw_focal_loss, p2p_loss, MSE, Grad
from utils.training.torch_trainer import Trainer
from torch.utils.data import DataLoader
from torch import nn
import torch
from utils.transform import SpatialTransformer, trilinear_interpolation3_torch
import random
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)


class MyTrainer(Trainer):

    WEIGHT_CHAMFER = 1
    WEIGHT_REG = 0.1

    def train_loss(self, model, input_data):

        x = torch.cat([input_data['post'], input_data['atlas']], dim=1)
        # print(x.mean())
        ddf = model(x)
        trans = SpatialTransformer(inshape)
        y_source = trans(input_data['post'], ddf)

        vtx_sample_ind = random.sample(range(0, input_data['atlas_vtx'].shape[1]), 10000)

        vtx_pat_sample = input_data['vtx'][:, vtx_sample_ind, :]
        atlas_vtx_sample = input_data['atlas_vtx'][:, vtx_sample_ind, :]

        vtx_pred_sample = trilinear_interpolation3_torch(atlas_vtx_sample, ddf)

        reg_loss_func = Grad('l2').loss
        loss_reg = reg_loss_func(None, ddf)
        loss_chamfer, _ = chamfer_distance(vtx_pred_sample, vtx_pat_sample)

        loss = self.WEIGHT_REG * loss_reg + self.WEIGHT_CHAMFER * loss_chamfer

        return {'ddf': ddf}, {'reg': loss_reg, 'chamfer': loss_chamfer, 'total_loss': loss}

random.seed(2023)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False

inshape = [128, 128, 128]  
model = LocalNet(inshape[0])

train_ds = CochlearCTCrop('E:/dingjie/mipresearch/reg_cochlear_monai/data_generation/data_128_inAtlas', 'train', aug=True)
val_ds = CochlearCTCrop('E:/dingjie/mipresearch/reg_cochlear_monai/data_generation/data_128_inAtlas', 'val', aug=False)
train_loader = DataLoader(dataset=train_ds, batch_size=1, num_workers=0, shuffle=True)
val_loader = DataLoader(dataset=val_ds, batch_size=1, num_workers=0, shuffle=False)

trainer = MyTrainer(save_dir='exp', name='task12-kcl-chamfer-reg0p1-post', num_epoch=200, log_freq=10)

optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)

trainer.fit_and_val(model, optimizer, train_loader=train_loader, val_loader=val_loader)

print(1)




