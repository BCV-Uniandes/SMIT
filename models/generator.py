import torch
import torch.nn as nn
from models.utils import print_debug as _print_debug
import math
from misc.utils import PRINT, to_var
from misc.blocks import (ResidualBlock, LayerNorm)


# ==================================================================#
# ==================================================================#
class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""

    def __init__(self, config, debug=False, **kwargs):
        super(Generator, self).__init__()
        layers = []
        repeat_num = config.g_repeat_num
        self.config = config
        self.image_size = config.image_size
        self.c_dim = config.c_dim
        self.color_dim = config.color_dim
        self.style_dim = config.style_dim
        self.Deterministic = config.Deterministic

        conv_dim = config.g_conv_dim
        if not config.Slim_Generator:
            conv_dim *= 2

        layers.append(
            nn.Conv2d(
                self.color_dim,
                conv_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        if config.Slim_Generator:
            conv_repeat = int(math.log(self.image_size,
                                       2)) - 5 if self.image_size > 64 else 2
        else:
            conv_repeat = 2
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
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(
                ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, AdaIn=True))

        # Up-Sampling
        for i in range(conv_repeat):
            if self.config.UpSample:
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
                layers.append(
                    nn.Conv2d(
                        curr_dim,
                        curr_dim // 2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False))
            else:
                layers.append(
                    nn.ConvTranspose2d(
                        curr_dim,
                        curr_dim // 2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False))
            if not self.Deterministic:
                layers.append(LayerNorm(curr_dim // 2))
            else:
                layers.append(
                    nn.InstanceNorm2d(curr_dim // 2, affine=True)
                )  # undesirable to generate images in vastly different styles
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        layers0 = []
        layers0.append(
            nn.Conv2d(
                curr_dim,
                self.color_dim,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False))
        layers0.append(nn.Tanh())
        self.img_reg = nn.Sequential(*layers0)

        layers1 = []
        layers1.append(
            nn.Conv2d(
                curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers1.append(nn.Sigmoid())
        self.attn_reg = nn.Sequential(*layers1)

        if debug and self.Deterministic:
            self.debug()

    def debug(self):
        def print_debug(x, v):
            return _print_debug(x, v, file=self.config.log)

        PRINT(self.config.log, '-- Generator:')
        feed = to_var(
            torch.ones(1, self.color_dim, self.image_size, self.image_size),
            volatile=True,
            no_cuda=True)
        features = print_debug(feed, self.main)
        print_debug(features, self.img_reg)
        print_debug(features, self.attn_reg)

    def forward(self, x):
        features = self.main(x)
        fake_img = self.img_reg(features)
        mask_img = self.attn_reg(features)
        fake_img = mask_img * x + (1 - mask_img) * fake_img
        output = [fake_img, mask_img]
        return output
