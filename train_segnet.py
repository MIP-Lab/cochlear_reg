import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import RandRotate, GaussianSmooth, RandGaussianSmooth
from scipy.spatial.transform import Rotation
import os
import glob
import nibabel as nib
import numpy as np

class CochlearCTSeg(Dataset):

    def __init__(self, data_root, phase, case_filter=None, aug=False) -> None:
        super().__init__()
        self.data_root = data_root

        case_filter = case_filter or (lambda x: True)

        self.vol_names = glob.glob(os.path.join(data_root + '/%s/PrePost_img' % phase, '*'))
        self.vol_names = [self.vol_names[i] for i in range(len(self.vol_names)) if case_filter(i)]

        self.vtx_names = [x.replace('PrePost_img', 'PrePost_vtx').replace('.nii.gz', '.pkl') for x in self.vol_names]
        self.cc_mask_names = [x.replace('PrePost_img', 'CC_mask').replace('.nii.gz', '.nii.gz') for x in self.vol_names]
        self.md_mask_names = [x.replace('PrePost_img', 'MD_mask').replace('.nii.gz', '.nii.gz') for x in self.vol_names]
        self.st_mask_names = [x.replace('PrePost_img', 'ST_mask') for x in self.vol_names]
        self.sv_mask_names = [x.replace('PrePost_img', 'SV_mask') for x in self.vol_names]
        self.crop_size = (128, 128, 128)
        self.aug = aug
        self.case_names = [x.split('\\')[-1].replace('.nii.gz', '') for x in self.vol_names]
    
    def __getitem__(self, index):
        
        # return {'pre': np.random.rand(1, 128, 128, 128).astype(np.float32), 
        #         'post': np.random.rand(1, 128, 128, 128).astype(np.float32), 'vtx': np.random.rand(10000, 3).astype(np.float32), 
        #         'atlas': np.random.rand(1, 128, 128, 128).astype(np.float32), 'atlas_vtx':np.random.rand(10000, 3).astype(np.float32)}
        if not os.path.exists(self.md_mask_names[index]):
            index = 10 
        prepost = nib.load(self.vol_names[index]).get_fdata().astype(np.float32)
        pre = prepost[:, :self.crop_size[1] , :]
        post = prepost[:, self.crop_size[1]:, :]
        vtx = np.load(open(self.vtx_names[index], 'rb'), allow_pickle=True)
        cc_mask = nib.load(self.cc_mask_names[index]).get_fdata().astype(np.float32)
        md_mask = nib.load(self.md_mask_names[index]).get_fdata().astype(np.float32)
        st_mask = nib.load(self.st_mask_names[index]).get_fdata().astype(np.float32)
        sv_mask = nib.load(self.sv_mask_names[index]).get_fdata().astype(np.float32)
        # except:
        #     pass
        if self.aug:
            [pre, post, md_mask, st_mask, sv_mask, cc_mask], vtx = self.do_augmentation([pre, post, md_mask, st_mask, sv_mask, cc_mask], vtx)
        # add channel dim
        pre, post = pre[None, :], post[None, :]

        mask = np.stack([md_mask, st_mask, sv_mask, cc_mask], axis=0)

        return {'post': post, 'vtx': vtx, 'index': index, 'mask': mask}

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
        
        # blur images
        # sigma = 0
        # if np.random.uniform(0, 1) > 0.25:
        #     sigma = np.random.uniform(0, 1) + 0.5 # 0.5 to 1.5
        #     S = RandGaussianSmooth(sigma_x=(sigma, sigma), sigma_y=(sigma, sigma), sigma_z=(sigma, sigma), prob=1)
        #     images = [S(image) for image in images]

        center = self.crop_size[0] / 2 - 1
        points -= center
        points = points @ rot_matrix
        points += center

        images = [image[0] for image in images]
        # print(theta, sigma, rot_axis)
        return images, points


    def __len__(self):
        return len(self.vol_names)
    
class SharedSkipUNet(nn.Module):

    def __init__(self, n_labels):
        super().__init__()

        self.act = nn.ReLU()

        self.conv11 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, padding='same')
        self.conv12 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.conv13 = nn.Conv3d(in_channels=64 + 128, out_channels=96, kernel_size=3, padding='same')
        self.conv14 = nn.Conv3d(in_channels=96, out_channels=64, kernel_size=3, padding='same')

        self.out = nn.Conv3d(in_channels=64, out_channels=n_labels, kernel_size=1, padding='same')

        self.max_pooling = nn.MaxPool3d(kernel_size=2)

        self.conv21 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.conv22 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.conv23 = nn.Conv3d(in_channels=64 + 128, out_channels=128, kernel_size=3, padding='same')
        self.conv24 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding='same')

        self.tconv1 = nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=2, stride=2)
    

    def forward(self, x):

        x1 = self.act(self.conv11(x))

        x1 = self.act(self.conv12(x1))

        x2 = self.max_pooling(x1)
        x2 = self.act(self.conv21(x2))
        x2 = self.act(self.conv22(x2))

        x2up = self.tconv1(x2)

        x1 = torch.concat([x1, x2up], dim=1)

        x1 = self.act(self.conv13(x1))
        x1 = self.act(self.conv14(x1))

        x2down = self.max_pooling(x1)

        x2 = torch.concat([x2down, x2], dim=1)

        x2 = self.act(self.conv23(x2))
        x2 = self.act(self.conv24(x2))

        x2up = self.tconv2(x2)

        x1 = torch.concat([x1, x2up], dim=1)
        x1 = self.act(self.conv13(x1))
        x1 = self.act(self.conv14(x1))

        y = self.out(x1)

        return y
    
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=32):
        super(UNet3D, self).__init__()
        
        self.encoder1 = DoubleConv(in_channels, base_channels)
        self.encoder2 = DoubleConv(base_channels, base_channels * 2)
        self.encoder3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.encoder4 = DoubleConv(base_channels * 4, base_channels * 8)
        
        self.pool = nn.MaxPool3d(2)
        
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
        
        self.upconv4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(base_channels * 16, base_channels * 8)
        
        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(base_channels * 8, base_channels * 4)
        
        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(base_channels * 2, base_channels)
        
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        bottleneck = self.bottleneck(self.pool(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1)

if __name__ == '__main__':
    from pycip.training.torch_trainer import Trainer
    from torch.optim import Adam
    from monai.losses.dice import DiceCELoss

    loss_fun = DiceCELoss(sigmoid=True)

    class MyTrainer(Trainer):

        def train_loss(self, model, input_data):
            
            pred = model(input_data['post'])

            loss = loss_fun(pred, input_data['mask'])

            return {}, {'total_loss': loss}

    cases_no_md = set(os.listdir())

    data_root = '/data/mipresearch/data/cochlear/data_128_inAtlas_99clip'

    train_ds = CochlearCTSeg(data_root, 'train')
    val_ds = CochlearCTSeg(data_root, 'val')

    train_loader = DataLoader(dataset=train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset=val_ds, batch_size=1, shuffle=False)
    
    model = SharedSkipUNet(n_labels=4)
    optimizer = Adam(model.parameters(), lr=1e-4)

    trainer = MyTrainer('exp', name='shared_skip_unet_cochlear_post')

    trainer.fit_and_val(model, optimizer, train_loader=train_loader, val_loader=val_loader, total_epochs=200, log_per_iteration=20,
                        validate_per_epoch=1, save_per_epoch=1, save_best=True)