import torch
import torch.nn as nn
import torch.nn.functional as F
from .build_network import NETWORK_REGISTRY

import numpy as np
import torchvision.utils
import matplotlib.pyplot as plt


class SpatialTransformerNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(SpatialTransformerNetwork, self).__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(input_size, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 60 * 28, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.output_size = output_size

    def forward(self, x):
        loc = self.localization(x)
        loc = loc.view(x.shape[0], -1)
        theta = self.fc_loc(loc)
        theta[:, 0].clamp_(0.8, 1.2)
        theta[:, 4].clamp_(0.8, 1.2)
        theta[:, 1].clamp_(-0.3, 0.3)
        theta[:, 3].clamp_(-0.3, 0.3)
        theta[:, 2] = 0
        theta[:, 5] = 0
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x


@NETWORK_REGISTRY.register()
def spatial_transformer_network(**kwargs):
    stn_net = SpatialTransformerNetwork(kwargs["input_size"], kwargs["output_size"])

    return stn_net
