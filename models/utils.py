import torch.nn as nn
from models.spectral import SpectralNorm as SpectralNormalization
from misc.blocks import AdaptiveInstanceNorm2d, ResidualBlock, LinearBlock, Conv2dBlock, LayerNorm
import ipdb

def get_SN(bool):
  if bool:
    return SpectralNormalization
  else:
    return lambda x:x

def print_debug(feed, layers):
  print(feed.size())
  for layer in layers:
    try:feed = layer(feed)
    except: ipdb.set_trace()
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d) \
                                    or isinstance(layer, nn.Linear) \
                                    or isinstance(layer, ResidualBlock) \
                                    or isinstance(layer, LinearBlock) \
                                    or isinstance(layer, Conv2dBlock) \
                                    or isinstance(layer, SpectralNormalization):
      print(str(layer).split('(')[0], feed.size())
  print(' ')
  return feed
