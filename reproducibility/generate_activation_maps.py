import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.kcl_networks import LocalNet
from model.segnet import SharedSkipUNet
import nibabel as nib
import numpy as np

input_size = (128, 128, 128)

chamfer_model = LocalNet(input_size)
p2p_model = LocalNet(input_size)
dice_model = LocalNet(input_size)
segnet = SharedSkipUNet(n_labels=4)

chamfer_model.load_state_dict(torch.load('../checkpoints/Exp_task16-kcl-chamfer-reg4p0-post_1/checkpoints/best.pth')['model'])
p2p_model.load_state_dict(torch.load('../checkpoints/Exp_task06-kcl-p2p-reg4p0-post_1/checkpoints/best.pth')['model'])
dice_model.load_state_dict(torch.load('../checkpoints/Exp_task51-kcl-chamfer0-dice-mse0-cc0-reg4p0-post_1/checkpoints/best.pth')['model'])
segnet.load_state_dict(torch.load('../checkpoints/Exp_shared_skip_unet_cochlear_post_1/checkpoints/best.pth')['model'])

chamfer_model.to('cuda')
p2p_model.to('cuda')
dice_model.to('cuda')
segnet.to('cuda')

chamfer_model.eval()
p2p_model.eval()
dice_model.eval()
segnet.eval()


atlas_nii = nib.load('../data/atlas/atlas.nii.gz').get_fdata().astype(np.float32)
atlas = torch.from_numpy(atlas_nii).to('cuda')[None, None]
atlas = 2 * (atlas - atlas.min()) / (atlas.max() - atlas.min()) - 1

files = os.listdir('../data/images')

# files = ['PID_2674_PLANID_1312_LEFT.nii.gz']

for f in files:

    print(f)

    post_nii = nib.load(f'../data/images/{f}').get_fdata().astype(np.float32)
    post = torch.from_numpy(post_nii).to('cuda')[None, None]

    with torch.no_grad():
        # chamfer_activation = chamfer_model.dp1_conv1(torch.concat([post, atlas], dim=1))
        # p2p_activation = p2p_model.dp1_conv1(torch.concat([post, atlas], dim=1))
        dice_activation = dice_model.dp1_conv1(torch.concat([post, atlas], dim=1))
        # segnet_activation = segnet.conv11(post)
    
    # nib.save(nib.Nifti1Image(chamfer_activation.detach().cpu().numpy()[0], np.eye(4)), f'../data/activation_maps/chamfer/{f}')
    # nib.save(nib.Nifti1Image(p2p_activation.detach().cpu().numpy()[0], np.eye(4)), f'../data/activation_maps/p2p/{f}')
    nib.save(nib.Nifti1Image(dice_activation.detach().cpu().numpy()[0], np.eye(4)), f'../data/activation_maps/dice/{f}')
    # nib.save(nib.Nifti1Image(segnet_activation.detach().cpu().numpy()[0], np.eye(4)), f'../data/activation_maps/segnet/{f}')

    # break