import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.cochlear_ct import read_mesh_file
import nibabel as nib
import numpy as np

case = 'PID_1689_PLANID_1096_LEFT'

for method in ['dice', 'chamfer', 'p2p', 'dice']:
    for strcture in ['MD', 'ST', 'SV', 'CC']:
        img_pred, _ = read_mesh_file(
            f'/data/mipresearch/reg_cochlear_monai/TBME/revision2/output/mesh/{method}/{strcture}/{case}.mesh')
        activation_pred, _ = read_mesh_file(
            f'../data/predictions_from_activation/{method}/{strcture}/{case}.mesh')
        
        diff = np.abs(img_pred - activation_pred).max()

        print(method, strcture, diff)

for strcture in ['MD', 'ST', 'SV', 'CC']:
    segnet_img_pred = nib.load(
        f'/data/mipresearch/reg_cochlear_monai/TBME/revision2/output/mask/segnet/{strcture}/{case}.nii.gz').get_fdata()
    segnet_activation_pred = nib.load(
        f'../data/predictions_from_activation/segnet/{strcture}/{case}.nii.gz').get_fdata()
    dice = 2 * (segnet_img_pred * segnet_activation_pred).sum() / ((segnet_img_pred + segnet_activation_pred).sum() + 1e-6)
    
    print('segnet', strcture, dice)