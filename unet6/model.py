import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                         nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                         nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True))


class UNet6(nn.Module):
    """Early fusion: [pre RGB, post RGB] -> five unnormalized class logits."""
    def __init__(self, base=32):
        super().__init__()
        self.encoders = nn.ModuleList()
        incoming = 6
        for outgoing in [base, base*2, base*4, base*8]:
            self.encoders.append(DoubleConv(incoming, outgoing))
            incoming = outgoing
        self.pool = nn.MaxPool2d(2)
        self.bridge = DoubleConv(base*8, base*16)
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for width in [base*8, base*4, base*2, base]:
            self.ups.append(nn.ConvTranspose2d(width*2, width, 2, stride=2))
            self.decoders.append(DoubleConv(width*2, width))
        self.head = nn.Conv2d(base, 5, 1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bridge(x)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = decoder(torch.cat([skip, x], dim=1))
        return self.head(x)
