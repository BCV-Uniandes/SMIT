import torch.nn as nn
from models.spectral import SpectralNorm as SpectralNormalization
from misc.blocks import ResidualBlock, LinearBlock, Conv2dBlock
from misc.utils import PRINT


def get_SN(bool):
    if bool:
        return SpectralNormalization
    else:
        return lambda x: x


def print_debug(feed, layers, file=None):
    if file is not None:
        PRINT(file, feed.size())
    else:
        print(feed.size())
    for layer in layers:
        try:
            feed = layer(feed)
        except BaseException:
            raise BaseException(
                "Type of layer {} not compatible with input {}.".format(
                    layer, feed))
        if isinstance(layer, nn.Conv2d) or isinstance(
                layer, nn.ConvTranspose2d) or isinstance(
                    layer, nn.Linear) or isinstance(
                        layer, ResidualBlock) or isinstance(
                            layer, LinearBlock) or isinstance(
                                layer, Conv2dBlock) or isinstance(
                                    layer, SpectralNormalization):
            _str = '{}, {}'.format(str(layer).split('(')[0], feed.size())
            if file is not None:
                PRINT(file, _str)
            else:
                print(_str)
    if file is not None:
        PRINT(file, ' ')
    else:
        print(' ')
    return feed
