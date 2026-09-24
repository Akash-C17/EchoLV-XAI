import torch
import torch.nn as nn
from .unet import DoubleConv, Up

class UNetPlusPlus(nn.Module):
    """
    Simplified U-Net++ implementation.
    Includes dense skip connections.
    """
    def __init__(self, n_channels=1, n_classes=1):
        super(UNetPlusPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Downsampling
        self.conv0_0 = DoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv1_0 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv2_0 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv3_0 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv4_0 = DoubleConv(512, 1024)

        # Dense Skip Connections
        self.up1_0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv0_1 = DoubleConv(64 + 64, 64)

        self.up2_0 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1_1 = DoubleConv(128 + 128, 128)
        self.up1_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv0_2 = DoubleConv(64 + 64 + 64, 64)

        self.up3_0 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2_1 = DoubleConv(256 + 256, 256)
        self.up2_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1_2 = DoubleConv(128 + 128 + 128, 128)
        self.up1_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv0_3 = DoubleConv(64 + 64 + 64 + 64, 64)

        self.up4_0 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv3_1 = DoubleConv(512 + 512, 512)
        self.up3_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2_2 = DoubleConv(256 + 256 + 256, 256)
        self.up2_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1_3 = DoubleConv(128 + 128 + 128 + 128, 128)
        self.up1_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv0_4 = DoubleConv(64 + 64 + 64 + 64 + 64, 64)

        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def _pad(self, x_up, x_target):
        diffY = x_target.size()[2] - x_up.size()[2]
        diffX = x_target.size()[3] - x_up.size()[3]
        if diffY > 0 or diffX > 0:
            import torch.nn.functional as F
            x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
        return x_up

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool1(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._pad(self.up1_0(x1_0), x0_0)], 1))

        x2_0 = self.conv2_0(self.pool2(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._pad(self.up2_0(x2_0), x1_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._pad(self.up1_1(x1_1), x0_0)], 1))

        x3_0 = self.conv3_0(self.pool3(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._pad(self.up3_0(x3_0), x2_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._pad(self.up2_1(x2_1), x1_0)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._pad(self.up1_2(x1_2), x0_0)], 1))

        x4_0 = self.conv4_0(self.pool4(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._pad(self.up4_0(x4_0), x3_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._pad(self.up3_1(x3_1), x2_0)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._pad(self.up2_2(x2_2), x1_0)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._pad(self.up1_3(x1_3), x0_0)], 1))

        output = self.final(x0_4)
        return output
