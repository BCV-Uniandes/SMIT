import torch
import torch.nn as nn
import numpy as np
from models.utils import get_SN
from models.utils import print_debug as _print_debug
import math
from misc.utils import PRINT, to_var
from collections import OrderedDict


# ==================================================================#
# ==================================================================#
class MultiDiscriminator(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, config, debug=False):
        super(MultiDiscriminator, self).__init__()

        self.image_size = config.image_size
        conv_dim = config.d_conv_dim
        conv_dim = conv_dim if config.image_size <= 256 else conv_dim // 2
        conv_dim = conv_dim if config.image_size <= 512 else conv_dim // 2
        self.conv_dim = conv_dim

        self.repeat_num = config.d_repeat_num
        self.c_dim = config.c_dim
        self.color_dim = config.color_dim
        self.config = config
        self.Norm = get_SN(True)

        self.downsample = nn.AvgPool2d(
            3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns_main = nn.ModuleList()
        self.cnns_src = nn.ModuleList()
        self.cnns_aux = nn.ModuleList()
        for idx in range(config.MultiDis):
            cnns_main, cnns_src, cnns_aux = self._make_net(idx)
            self.cnns_main.append(cnns_main)
            self.cnns_src.append(cnns_src)
            self.cnns_aux.append(cnns_aux)

        if debug:
            self.debug()

    def debug(self):
        feed = to_var(
            torch.ones(1, self.color_dim, self.image_size,
                       self.image_size),
            volatile=True,
            no_cuda=True)
        modelList = zip(self.cnns_main, self.cnns_src, self.cnns_aux)
        for idx, outs in enumerate(modelList):
            PRINT(config.log, '-- MultiDiscriminator ({}):'.format(idx))
            features = self.print_debug(feed, outs[-3])
            self.print_debug(features, outs[-2])
            self.print_debug(features, outs[-1]).view(feed.size(0), -1)
            feed = self.downsample(feed)

    def print_debug(self, x, v):
        return _print_debug(x, v, file=self.config.log)

    def _make_net(self, idx=0):
        image_size = self.image_size / (2**(idx))
        self.repeat_num = int(math.log(image_size, 2) - 1)
        k_size = int(image_size / np.power(2, self.repeat_num))
        layers = []
        conv = self.Norm(
            nn.Conv2d(
                self.color_dim,
                self.conv_dim,
                kernel_size=4,
                stride=2,
                padding=1))
        layers += [('conv_' + str(self.conv_dim), conv)]
        layers += [('relu_' + str(self.conv_dim),
                    nn.LeakyReLU(0.01, inplace=True))]
        curr_dim = self.conv_dim
        for _ in range(1, self.repeat_num):
            conv = self.Norm(
                nn.Conv2d(
                    curr_dim, curr_dim * 2, kernel_size=4, stride=2,
                    padding=1))
            layers += [('conv_' + str(curr_dim * 2), conv)]
            layers += [('relu_' + str(curr_dim * 2),
                        nn.LeakyReLU(0.01, inplace=True))]
            curr_dim *= 2

        main = nn.Sequential(OrderedDict(layers))

        src_conv = nn.Conv2d(
            curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        src = nn.Sequential(OrderedDict([('src', src_conv)]))

        aux_conv = nn.Conv2d(
            curr_dim, self.c_dim, kernel_size=k_size, bias=False)
        aux = nn.Sequential(OrderedDict([('aux', aux_conv)]))

        return main, src, aux

    def forward(self, x):
        outs_src = []
        outs_aux = []
        modelList = zip(self.cnns_main, self.cnns_src, self.cnns_aux)
        for outs in modelList:
            main = outs[0](x)
            _src = outs[1](main)
            _aux = outs[2](main).view(main.size(0), -1)
            outs_src.append(_src)
            outs_aux.append(_aux)
            x = self.downsample(x)
        return outs_src, outs_aux,
