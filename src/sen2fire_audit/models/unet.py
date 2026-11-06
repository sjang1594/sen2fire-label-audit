"""U-Net and Attention U-Net.

Attribute names and the order in which layers are created are part of the interface: the first keeps
released checkpoints loadable, the second keeps seeded initialization reproducible.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, x_ch, g_ch, inter_ch):
        super().__init__()
        
        self.theta_x = nn.Conv2d(x_ch, inter_ch, 1, bias=False)
        self.phi_g = nn.Conv2d(g_ch, inter_ch, 1, bias=True)
        self.psi = nn.Conv2d(inter_ch, 1, 1, bias=True)

    def forward(self, x, g):
        return x * torch.sigmoid(self.psi(F.relu(self.theta_x(x) + self.phi_g(g))))

def _up(x, ref):
    return F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=False)

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes=2, base_ch=32):
        super().__init__()
        
        c = base_ch
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.bot = ConvBlock(c * 8, c * 16)
        self.dec4 = ConvBlock(c * 16 + c * 8, c * 8)
        self.dec3 = ConvBlock(c * 8 + c * 4, c * 4)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.dec1 = ConvBlock(c * 2 + c, c)
        self.out = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(F.max_pool2d(c1, 2))
        c3 = self.enc3(F.max_pool2d(c2, 2))
        c4 = self.enc4(F.max_pool2d(c3, 2))
        c5 = self.bot(F.max_pool2d(c4, 2))
        d4 = self.dec4(torch.cat([_up(c5, c4), c4], 1))
        d3 = self.dec3(torch.cat([_up(d4, c3), c3], 1))
        d2 = self.dec2(torch.cat([_up(d3, c2), c2], 1))
        d1 = self.dec1(torch.cat([_up(d2, c1), c1], 1))
        return self.out(d1)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels, num_classes=2, base_ch=32):
        super().__init__()
        
        c = base_ch
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.bot = ConvBlock(c * 8, c * 16)
        self.att4 = AttentionGate(c * 8, c * 16, c * 4)
        self.att3 = AttentionGate(c * 4, c * 8, c * 2)
        self.att2 = AttentionGate(c * 2, c * 4, c)
        self.att1 = AttentionGate(c, c * 2, c // 2)
        self.dec4 = ConvBlock(c * 16 + c * 8, c * 8)
        self.dec3 = ConvBlock(c * 8 + c * 4, c * 4)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.dec1 = ConvBlock(c * 2 + c, c)
        self.out = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(F.max_pool2d(c1, 2))
        c3 = self.enc3(F.max_pool2d(c2, 2))
        c4 = self.enc4(F.max_pool2d(c3, 2))
        c5 = self.bot(F.max_pool2d(c4, 2))
        u4 = _up(c5, c4)
        d4 = self.dec4(torch.cat([u4, self.att4(c4, u4)], 1))
        u3 = _up(d4, c3)
        d3 = self.dec3(torch.cat([u3, self.att3(c3, u3)], 1))
        u2 = _up(d3, c2)
        d2 = self.dec2(torch.cat([u2, self.att2(c2, u2)], 1))
        u1 = _up(d2, c1)
        d1 = self.dec1(torch.cat([u1, self.att1(c1, u1)], 1))
        return self.out(d1)
