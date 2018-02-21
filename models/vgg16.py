from __future__ import division
import ipdb
import inspect
import os
import time
import math
import glob
import numpy as np
from six.moves import xrange
import pickle
import sys
from vgg_pytorch import vgg16 as model_vgg16
import config as cfg
import torch.nn as nn
import torch.legacy.nn as nn_legacy
from torch.autograd import Variable
import math
import torch
# torch.backends.cudnn.enabled=False

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from caffe_tensorflow import convert as caffe_tf


# class Discriminator(nn.Module):
#     """Discriminator. PatchGAN."""
#     def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
#         super(Discriminator, self).__init__()

#         layers = []
#         layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
#         layers.append(nn.LeakyReLU(0.01, inplace=True))

#         curr_dim = conv_dim
#         for i in range(1, repeat_num):
#             layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
#             layers.append(nn.LeakyReLU(0.01, inplace=True))
#             curr_dim = curr_dim * 2

#         k_size = int(image_size / np.power(2, repeat_num))
#         self.main = nn.Sequential(*layers)
#         self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

#     def forward(self, x):
#         h = self.main(x)
#         # ipdb.set_trace()
#         out_real = self.conv1(h)
#         out_aux = self.conv2(h)
#         return out_real.squeeze(), out_aux.squeeze()

class Discriminator(nn.Module):

  def __init__(self, finetuning='/npy/weights'):

    super(Discriminator, self).__init__()
    self.finetuning = finetuning
    self.std = np.array((0.229, 0.224, 0.225))

    self.mean = Variable(torch.FloatTensor(np.array([102.9801, 115.946, 122.7717]).reshape(1,-1,1,1)))

    if torch.cuda.is_available():
      self.mean  = self.mean.cuda()

    self._initialize_weights()

  def _initialize_weights(self):

    if 'aunet' in self.finetuning or 'emonet' in self.finetuning:
      self.model, filename = self.load_facexnet()
      mode=self.finetuning.upper()+' '+filename  

    elif 'imagenet' in self.finetuning: 
      mode='Imagenet'
      self.model = model_vgg16(pretrained=True, num_classes=1000)
      modules = self.model.modules()
      for m in modules:
        if isinstance(m, nn.Linear) and m.weight.data.size()[0]==1000:
          w1 = m.weight.data[1:3].view(2,-1)
          b1 = torch.FloatTensor(np.array((m.bias.data[1], m.bias.data[2])).reshape(-1))
      mod = list(self.model.classifier)
      mod.pop()
      mod.append(torch.nn.Linear(4096,2))
      new_classifier = torch.nn.Sequential(*mod)
      self.model.classifier = new_classifier
      for m in modules:
        if isinstance(m, nn.Linear) and m.weight.data.size()[0]==1000:
          m.weight.data = w1
          m.bias.data = b1

    else:
      mode='RANDOM'
      self.model = model_vgg16(pretrained=False, num_classes=2)      
    print("[OK] Weights initialized from %s"%(mode))
    
  def load_facexnet(self):
    npy_file = glob.glob(os.path.join(self.finetuning, '*_pytorch.npy'))
    if len(npy_file)==0:
        caffemodel = glob.glob(os.path.join(self.finetuning, '*.caffemodel'))[0]
        npy_file = caffemodel.replace('.caffemodel', '_pytorch.npy')
        self.caffemodel2npy(caffemodel, npy_file)    
    else: 
        npy_file = npy_file[0]
    print(" [*] Loading weights from: "+npy_file)
    params = np.load(npy_file, encoding='latin1').item()
    params={k.encode("utf-8"): v for k,v in params.iteritems()}
    #model.modules[31]=nn_legacy.View(-1,25088) #modules_caffe[32].weight.cpu().numpy().shape[1]
    #model.modules[-1].weight = model.modules[-1].weight[1].view(1,-1)
    #model.modules[-1].bias = torch.FloatTensor(np.array(model.modules[-1].bias[1]).reshape(-1))

    params_keys = sorted(params.keys())

    if 'aunet' in self.finetuning: n_classes = 2
    elif 'emonet' in self.finetuning: n_classes = 8

    conv_w = [params[m][0] for m in params_keys if 'conv' in m]
    conv_b = [params[m][1] for m in params_keys if 'conv' in m]
    fc_w = [params[m][0] for m in params_keys if 'fc' in m]
    fc_b = [params[m][1] for m in params_keys if 'fc' in m]

    if 'emonet' in self.finetuning or 'imagenet' in self.finetuning:
      fc_w[-1] = fc_w[-1][:1].reshape(1,-1)
      fc_b[-1] = fc_b[-1][:1].reshape(-1)

    # ipdb.set_trace()

    model = model_vgg16(pretrained=False, num_classes=n_classes)
    for m in model.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data = torch.FloatTensor(conv_w.pop(0))
        m.bias.data = torch.FloatTensor(conv_b.pop(0))
      elif isinstance(m, nn.Linear):
        m.weight.data = torch.FloatTensor(fc_w.pop(0))
        m.bias.data = torch.FloatTensor(fc_b.pop(0))

    # ipdb.set_trace()
        
    return model, npy_file

  def caffemodel2npy(self, caffemodel, npy_file):
    from caffe2npy import convert
    def_path = 'models/pretrained/aunet/vgg16/deploy_Test.pt'
    assert os.path.isfile(caffemodel), caffemodel+" must exist to finetune"
    # convert(def_path, caffemodel, npy_file) 
    # ipdb.set_trace()
    convert(def_path, caffemodel, npy_file, 'test')    
    # caffe_tf.convert(def_path, caffemodel, npy_file, npy_file.replace('npy','py'), 'test')

  def image2vgg(self, image):
    image_ = image
    image_ = ((image_+1.)/2.)*255.
    # ipdb.set_trace()
    image_ = image_ - self.mean  
    return image_  

  def forward(self, image):
    image_ = self.image2vgg(image)
    x = self.model(image_)
    return x      