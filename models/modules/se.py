import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16):
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.ratio = ratio
        self.relu = nn.ReLU(True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.transform = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_avg = self.global_pool(x).view(x.size(0), -1)
        mask = self.transform(x_avg)
        return x * mask.view(x.size(0), -1, 1, 1)

