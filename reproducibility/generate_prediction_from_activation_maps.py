import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.kcl_networks import LocalNet
from model.segnet import SharedSkipUNet
import nibabel as nib
import numpy as np
from utils.transform import trilinear_interpolation3_torch
import random
from dataset.cochlear_ct import read_mesh_file, write_mesh_file

_, MD_triangles = read_mesh_file('../data/atlas/atlas_MD.mesh')
_, ST_triangles = read_mesh_file('../data/atlas/atlas_ST.mesh')
_, SV_triangles = read_mesh_file('../data/atlas/atlas_SV.mesh')
_, CC_triangles = read_mesh_file('../data/atlas/atlas_CC.mesh')

atlas_vtx = np.load(open('../data/atlas/atlas_all_vtx.pkl', 'rb'), allow_pickle=True)

input_size = (128, 128, 128)

chamfer_model = LocalNet(input_size)
p2p_model = LocalNet(input_size)
dice_model = LocalNet(input_size)
segnet = SharedSkipUNet(n_labels=4)

chamfer_model.load_state_dict(torch.load('../checkpoints/Exp_task16-kcl-chamfer-reg4p0-post_1/checkpoints/best.pth')['model'])
p2p_model.load_state_dict(torch.load('../checkpoints/Exp_task06-kcl-p2p-reg4p0-post_1/checkpoints/best.pth')['model'])
dice_model.load_state_dict(torch.load('../checkpoints/Exp_task51-kcl-chamfer0-dice-mse0-cc0-reg4p0-post_1/checkpoints/best.pth')['model'])
segnet.load_state_dict(torch.load('../checkpoints/Exp_shared_skip_unet_cochlear_post_1/checkpoints/best.pth')['model'])

chamfer_model.dp1_conv1 = torch.nn.Identity()
p2p_model.dp1_conv1 = torch.nn.Identity()
dice_model.dp1_conv1 = torch.nn.Identity()
segnet.conv11 = torch.nn.Identity()

chamfer_model.to('cuda')
# chamfer_model1.to('cuda')
p2p_model.to('cuda')
dice_model.to('cuda')
segnet.to('cuda')

chamfer_model.eval()
# chamfer_model1.eval()
p2p_model.eval()
dice_model.eval()
segnet.eval()

# create folder
for method in ['chamfer', 'p2p', 'dice', 'segnet']:
    for structure in ['MD', 'ST', 'SV', 'CC']:
        try:
            os.makedirs(f'../data/predictions_from_activation/{method}/{structure}')
        except FileExistsError:
            raise

cases = [item.replace('.nii.gz', '') for item in os.listdir('../data/images')]

atlas_vtx = torch.from_numpy(atlas_vtx[None]).to('cuda')

for case in cases:

    chamfer_activation = np.array(nib.load(f'../data/activation_maps/chamfer/{case}.nii.gz').dataobj)
    p2p_activation = np.array(nib.load(f'../data/activation_maps/p2p/{case}.nii.gz').dataobj)
    dice_activation = np.array(nib.load(f'../data/activation_maps/dice/{case}.nii.gz').dataobj)
    segnet_activation = np.array(nib.load(f'../data/activation_maps/segnet/{case}.nii.gz').dataobj)
    
    chamfer_activation = torch.from_numpy(chamfer_activation[None]).to('cuda')
    p2p_activation = torch.from_numpy(p2p_activation[None]).to('cuda')
    dice_activation = torch.from_numpy(dice_activation[None]).to('cuda')
    segnet_activation = torch.from_numpy(segnet_activation[None]).to('cuda')

    for method_name, model, activation in zip(['chamfer', 'p2p', 'dice'], 
                                              [chamfer_model, p2p_model, dice_model],
                                              [chamfer_activation, p2p_activation, dice_activation]
                                              ):
        with torch.no_grad():
            ddf = model(activation)
            vtx_pred = trilinear_interpolation3_torch(atlas_vtx, ddf)
            vtx_pred = vtx_pred.detach().cpu().numpy()

            MD_pred = vtx_pred[0][: 17947, :]
            ST_pred = vtx_pred[0][17947: 21291, :]
            SV_pred = vtx_pred[0][21291: 24423, :]
            CC_pred = vtx_pred[0][24423: , :]
        
            write_mesh_file(MD_pred, MD_triangles, 
                            f'../data/predictions_from_activation/{method_name}/MD/{case}.mesh')
            write_mesh_file(ST_pred, ST_triangles, 
                            f'../data/predictions_from_activation/{method_name}/ST/{case}.mesh')
            write_mesh_file(SV_pred, SV_triangles, 
                            f'../data/predictions_from_activation/{method_name}/SV/{case}.mesh')
            write_mesh_file(CC_pred, CC_triangles, 
                            f'../data/predictions_from_activation/{method_name}/CC/{case}.mesh')
    
    with torch.no_grad():
        segnet_pred = segnet(segnet_activation)
        
        MD_mask = (torch.sigmoid(segnet_pred[0, 0]).detach().cpu().numpy() > 0.5).astype(np.uint8)
        ST_mask = (torch.sigmoid(segnet_pred[0, 1]).detach().cpu().numpy() > 0.5).astype(np.uint8)
        SV_mask = (torch.sigmoid(segnet_pred[0, 2]).detach().cpu().numpy() > 0.5).astype(np.uint8)
        CC_mask = (torch.sigmoid(segnet_pred[0, 3]).detach().cpu().numpy() > 0.5).astype(np.uint8)

        affine = np.array([
            [-0.2, 0, 0, 0],
            [0, -0.2, 0, 0],
            [0, 0, 0.2, 0],
            [0, 0, 0, 1]
        ])

        nib.save(nib.Nifti1Image(MD_mask, affine), f'../data/predictions_from_activation/segnet/MD/{case}.nii.gz')
        nib.save(nib.Nifti1Image(ST_mask, affine), f'../data/predictions_from_activation/segnet/ST/{case}.nii.gz')
        nib.save(nib.Nifti1Image(SV_mask, affine), f'../data/predictions_from_activation/segnet/SV/{case}.nii.gz')
        nib.save(nib.Nifti1Image(CC_mask, affine), f'../data/predictions_from_activation/segnet/CC/{case}.nii.gz')

    # break