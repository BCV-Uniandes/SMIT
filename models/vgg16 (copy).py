from __future__ import division
import inspect
import os
import time
import math
from glob import glob
#import tensorflow as tf
import glob
import numpy as np
from six.moves import xrange
import pickle
import sys
from vgg_pytorch import vgg16 as model_vgg16
from model_utils import MODEL_UTILS
import config as cfg
import torch.nn as nn
import torch.legacy.nn as nn_legacy
import math
import torch
import ipdb
torch.backends.cudnn.enabled=False

class VGG16(MODEL_UTILS):

  def __init__(self, batch_size=64, learning_rate=0.0001, au=0,
         pretrained_model_location='', c_dim=3, dataset_name='default', fold=0,
         finetuning_file='', checkpoint_dir=None, logs_dir='logs', generator=object):

    self.name = type(self).__name__.lower() 
    self.c_dim = c_dim
    self.model_size = 224
    self.batch_size = batch_size   
    self.from_caffe=True

    super(VGG16, self).__init__()

    self.au = au
    self.fold = fold
    self.logs_dir=logs_dir
    self.grayscale = (self.c_dim == 1)
    self.generator = generator
    self.dataset_name = dataset_name
    self.checkpoint_dir = checkpoint_dir
    self.finetuning = finetuning_file

    self.learning_rate = learning_rate
    self.pretrained_model = pretrained_model_location

    
    self.std = np.array((0.229, 0.224, 0.225))

    if self.from_caffe:
      self.mean = np.array([102.9801, 115.946, 122.7717]).reshape(1,1,1,-1)
    else: 
      self.mean = np.array((0.485, 0.456, 0.406))

    self._initialize_weights()

  def _initialize_weights(self):

    if self.finetuning == 'aunet' or self.finetuning == 'emonet':
      self.model, filename = self.load_facexnet()
      mode=self.finetuning.upper()+' '+filename  
    elif self.finetuning == 'imagenet': 
      mode='Imagenet'
      self.model = model_vgg16(pretrained=True, num_classes=1000)
      modules = self.model.modules()
      for m in modules:
        if isinstance(m, nn.Linear) and m.weight.data.size()[0]==1000:
          #ipdb.set_trace()
          m.weight.data = m.weight.data[1:3].view(2,-1)
          m.bias.data = torch.FloatTensor(np.array((m.bias.data[1], m.bias.data[2])).reshape(-1))
    else:
      mode='RANDOM'
      self.model = model_vgg16(pretrained=False, num_classes=2)      
    print("[OK] Weights initialized from %s"%(mode))
    
  def load_facexnet(self):
    npy_file = glob.glob(os.path.join(self.pretrained_model, '*.npy'))
    if len(npy_file)==0:
      npy_file = self.caffemodel2npy()
    else: 
      npy_file = npy_file[0]
    params = np.load(npy_file, encoding='latin1').item()
    #model.modules[31]=nn_legacy.View(-1,25088) #modules_caffe[32].weight.cpu().numpy().shape[1]
    #model.modules[-1].weight = model.modules[-1].weight[1].view(1,-1)
    #model.modules[-1].bias = torch.FloatTensor(np.array(model.modules[-1].bias[1]).reshape(-1))

    params_keys = sorted(params.keys())

    if self.finetuning=='aunet': n_classes = 2
    elif self.finetuning=='emonet': n_classes = 8

    conv_w = [params[m][0] for m in params_keys if 'conv' in m]
    conv_b = [params[m][1] for m in params_keys if 'conv' in m]
    fc_w = [params[m][0] for m in params_keys if 'fc' in m]
    fc_b = [params[m][1] for m in params_keys if 'fc' in m]

    fc_w[-1] = fc_w[-1][:2].reshape(2,-1)
    fc_b[-1] = fc_b[-1][:2].reshape(-1)

    model = model_vgg16(pretrained=False, num_classes=2)
    for m in model.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data = torch.FloatTensor(conv_w.pop(0))
        m.bias.data = torch.FloatTensor(conv_b.pop(0))
      elif isinstance(m, nn.Linear):
        m.weight.data = torch.FloatTensor(fc_w.pop(0))
        m.bias.data = torch.FloatTensor(fc_b.pop(0))
        
    return model, npy_file

  def caffemodel2npy(self):
    def_path = os.path.join(cfg.pretrained_models, self.finetuning, self.name, 'deploy_Test.pt')
    caffemodel = glob.glob(os.path.join(self.pretrained_model, '*.caffemodel'))
    assert len(caffemodel)>0, "There is no [caffemodel] model in "+self.pretrained_model
    caffemodel = caffemodel[0]
    npy_file = caffemodel.replace('caffemodel', 'npy')
    os.system('./caffe2npy.py -- --prototxt %s --caffemodel %s --output %s'%(def_path, caffemodel, npy_file))
    print("model [npy] created at {}".format(npy_file))
    return npy_file

  def forward(self, image):
    x = self.model(image)
    return x      