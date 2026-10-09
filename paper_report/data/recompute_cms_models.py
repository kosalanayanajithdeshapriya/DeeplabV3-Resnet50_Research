import torch, torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.block = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(s, x): return s.block(x)
class UNet(nn.Module):
    def __init__(s, c=32):
        super().__init__()
        s.enc1, s.enc2, s.enc3, s.enc4 = DoubleConv(3, c), DoubleConv(c, 2*c), DoubleConv(2*c, 4*c), DoubleConv(4*c, 8*c)
        s.pool = nn.MaxPool2d(2); s.bottleneck = DoubleConv(8*c, 16*c)
        s.up4, s.dec4 = nn.ConvTranspose2d(16*c, 8*c, 2, stride=2), DoubleConv(16*c, 8*c)
        s.up3, s.dec3 = nn.ConvTranspose2d(8*c, 4*c, 2, stride=2), DoubleConv(8*c, 4*c)
        s.up2, s.dec2 = nn.ConvTranspose2d(4*c, 2*c, 2, stride=2), DoubleConv(4*c, 2*c)
        s.up1, s.dec1 = nn.ConvTranspose2d(2*c, c, 2, stride=2), DoubleConv(2*c, c)
        s.out_conv = nn.Conv2d(c, 1, 1)
    def forward(s, x):
        e1 = s.enc1(x); e2 = s.enc2(s.pool(e1)); e3 = s.enc3(s.pool(e2)); e4 = s.enc4(s.pool(e3))
        b = s.bottleneck(s.pool(e4))
        d4 = s.dec4(torch.cat([s.up4(b), e4], 1)); d3 = s.dec3(torch.cat([s.up3(d4), e3], 1))
        d2 = s.dec2(torch.cat([s.up2(d3), e2], 1)); d1 = s.dec1(torch.cat([s.up1(d2), e1], 1))
        return s.out_conv(d1)

