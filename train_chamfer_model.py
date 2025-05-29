import os

from dataset.cochlear_ct import CochlearCTCrop
from model.kcl_networks import LocalNet, LocalNetNoBN
from model.loss import pw_focal_loss, p2p_loss, chamfer_loss, MSE, Grad
from utils.training.torch_trainer import Trainer
from torch.utils.data import DataLoader
from torch import nn
import torch
from utils.transform import SpatialTransformer, trilinear_interpolation3_torch
import random
import numpy as np

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--nii_folder", help="path to training nifti files")
parser.add_argument("--vtx_folder", help="path to .pkl files containing the ground-truth vertices for each training image")
parser.add_argument("--atlas_name", help="name of the case used as the atlas")
parser.add_argument("--experiment_name", help="pexperiment id for checkpoints_and_logs")

args = parser.parse_args()

class MyTrainer(Trainer):

    WEIGHT_CHAMFER = 1
    WEIGHT_REG = 4.0

    def train_loss(self, model, input_data):

        x = torch.cat([input_data['vol'], input_data['atlas']], dim=1)
        # print(x.mean())
        ddf = model(x)
        trans = SpatialTransformer(inshape)
        y_source = trans(input_data['vol'], ddf)

        vtx_sample_ind1 = random.sample(range(0, input_data['vtx'].shape[1]), 10000)
        vtx_sample_ind2 = random.sample(range(0, input_data['atlas_vtx'].shape[1]), 10000)

        vtx_pat_sample = input_data['vtx'][:, vtx_sample_ind1, :]
        atlas_vtx_sample = input_data['atlas_vtx'][:, vtx_sample_ind2, :]

        vtx_pred_sample = trilinear_interpolation3_torch(atlas_vtx_sample, ddf)

        reg_loss_func = Grad('l2').loss
        loss_reg = reg_loss_func(None, ddf)
        loss_chamfer = chamfer_loss(vtx_pred_sample, vtx_pat_sample)

        loss = self.WEIGHT_REG * loss_reg + self.WEIGHT_CHAMFER * loss_chamfer

        return {'ddf': ddf}, {'reg': loss_reg, 'chamfer': loss_chamfer, 'total_loss': loss}

random.seed(2023)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False

inshape = [128, 128, 128]  
model = LocalNet(inshape[0])

all_cases = [item.replace('.nii.gz', '') for item in os.listdir(args.nii_folder)]
rand_index = np.random.permutation(len(all_cases))
num_training = int(0.8 * len(all_cases))

train_cases = [all_cases[i] for i in rand_index[: num_training]]
val_cases = [all_cases[i] for i in rand_index[num_training: ]]

train_ds = CochlearCTCrop(nii_path=args.nii_folder, cases=train_cases, 
                         vtx_path=args.vtx_folder, atlas_case_name=args.atlas_name, aug=True)
val_ds = CochlearCTCrop(nii_path=args.nii_folder, cases=val_cases, 
                         vtx_path=args.vtx_folder, atlas_case_name=args.atlas_name, aug=False)

train_loader = DataLoader(dataset=train_ds, batch_size=1, num_workers=0, shuffle=True)
val_loader = DataLoader(dataset=val_ds, batch_size=1, num_workers=0, shuffle=False)

trainer = MyTrainer(save_dir='checkpoints', name=args.experiment_name)

optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)

trainer.fit_and_val(model, optimizer, train_loader=train_loader, val_loader=val_loader, 
                    total_epochs=200, log_per_iteration=10, save_per_epoch=1)

print(1)




