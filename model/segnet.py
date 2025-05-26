import torch.nn.functional as F
import torch.nn as nn
import torch

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