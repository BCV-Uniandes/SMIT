import torch.nn as nn
from models.spectral import SpectralNorm as SpectralNormalization
from models.GroupNorm import GroupNorm
from misc.blocks import ResidualBlock, LinearBlock, Conv2dBlock
from misc.utils import PRINT
from torch.nn import init


def get_SN(bool):
    if bool:
        return SpectralNormalization
    else:
        return lambda x: x


def get_GN(bool):
    if bool:
        return GroupNorm
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


def init_weights(net, init_type="normal", gain=0.02, bias=0.0):
    def init_func(m):
        classname = m.__class__.__name__
        _conv = classname.find("Conv")
        _linear = classname.find("Linear")
        if hasattr(m, "weight") and (_conv != -1 or _linear != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" %
                    init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, bias)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    # print("initialize network with %s" % init_type)
    net.apply(init_func)


def init_net(net, init_type="normal", init_gain=0.02, init_bias=0.0):
    init_weights(net, init_type, gain=init_gain, bias=init_bias)
    return net
