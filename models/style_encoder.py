import torch
import torch.nn as nn
import numpy as np
from models.utils import print_debug as _print_debug
import math
from misc.utils import PRINT, to_var


# ==================================================================#
# ==================================================================#
class StyleEncoder(nn.Module):
    def __init__(self, config, debug=False):
        super(StyleEncoder, self).__init__()
        self.image_size = config.image_size
        self.color_dim = config.color_dim
        self.c_dim = config.c_dim
        self.config = config
        style_dim = config.style_dim
        conv_dim = config.g_conv_dim // 2

        layers = []
        layers.append(
            nn.Conv2d(
                self.color_dim,
                conv_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        down = [2, 1]
        conv_repeat = int(math.log(self.image_size, 2)) - \
            down[0]    # 1 until 2x2, 2 for 4x4
        curr_dim = conv_dim
        for i in range(conv_repeat):
            layers.append(
                nn.Conv2d(
                    curr_dim,
                    curr_dim * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        layers.append(
            nn.Conv2d(
                curr_dim,
                curr_dim * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False))
        self.main = nn.Sequential(*layers)
        curr_dim = curr_dim * 2

        conv_repeat = conv_repeat + down[1]
        k_size = int(self.image_size / np.power(2, conv_repeat))
        layers = []
        layers.append(nn.Linear(curr_dim * k_size * k_size, 256, bias=True))
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(256, 256, bias=True))
        layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(256, style_dim))
        self.fc = nn.Sequential(*layers)
        if debug:
            self.debug()

    def print_debug(self, x, v):
        return _print_debug(x, v, file=self.config.log)

    def debug(self):
        feed = to_var(
            torch.ones(1, self.color_dim, self.image_size, self.image_size),
            volatile=True,
            no_cuda=True)
        PRINT(self.config.log, '-- StyleEncoder:')
        features = self.print_debug(feed, self.main)
        fc_in = features.view(features.size(0), -1)
        self.print_debug(fc_in, self.fc)

    def forward(self, x):
        features = self.main(x)
        fc_input = features.view(features.size(0), -1)
        style = self.fc(fc_input)
        return style
