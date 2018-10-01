import os
import torch
import torch.nn as nn
import ipdb

class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.meta = {'name': 'ImageNet',
                     'mean':[0.485, 0.456, 0.406],
                     'std':[0.229, 0.224, 0.225],
                     'imSize': [224, 224, 3]}        
        self.features = features

    def forward(self, x):
        x = self.features(x)
        return x

def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]}

def ImageNet(weights_path='./models/perceptual/ImageNet.pth', **kwargs):
    model = VGG(make_layers(cfg['D']), **kwargs)
    print(' -- Loading ImageNet weights from '+os.path.abspath(weights_path))
    state_dict = torch.load(weights_path)
    state_dict = {key:param for key,param in state_dict.items() if 'classifier' not in key}
    # ipdb.set_trace()
    model.load_state_dict(state_dict)
    return model
