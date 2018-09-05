import os
import torch
import torch.nn as nn
import ipdb

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.meta = {'name': 'EmoNet',
                     'mean': [123.675, 116.25, 103.53],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}

    def forward(self, x):
        x = self.features(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]}

def EmoNet(weights_path='./models/perceptual/EmoNet.pth', **kwargs):
    model = VGG(make_layers(cfg['D']), **kwargs)
    print(' -- Loading perceptual weights from '+os.path.abspath(weights_path))
    state_dict = torch.load(weights_path)
    state_dict = {key.replace('model.',''):param for key,param in state_dict.items() if 'classifier' not in key}
    # ipdb.set_trace()
    model.load_state_dict(state_dict)
    return model
