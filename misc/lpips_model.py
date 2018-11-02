from __future__ import absolute_import

import sys
# sys.path.append('..')
# sys.path.append('.')
import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
from misc.lpips.base_model import BaseModel
from scipy.ndimage import zoom
import fractions
import functools
import skimage.transform
# from IPython import embed

# import torch._utils
# try:
#   torch._utils._rebuild_tensor_v2
# except AttributeError:
#   def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
#     tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
#     tensor.requires_grad = requires_grad
#     tensor._backward_hooks = backward_hooks
#     return tensor
#   torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

from misc.lpips import networks_basic as networks

class DistModel(BaseModel):
  def name(self):
    return self.model_name

  def initialize(self, model='net-lin', net='alex', pnet_rand=False, pnet_tune=False, model_path=None, colorspace='Lab', use_gpu=True, printNet=False, spatial=False, spatial_shape=None, spatial_order=1, spatial_factor=None, is_train=False, lr=.0001, beta1=0.5, version='0.1'):
    '''
    INPUTS
      model - ['net-lin'] for linearly calibrated network
          ['net'] for off-the-shelf network
          ['L2'] for L2 distance in Lab colorspace
          ['SSIM'] for ssim in RGB colorspace
      net - ['squeeze','alex','vgg']
      model_path - if None, will look in weights/[NET_NAME].pth
      colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
      use_gpu - bool - whether or not to use a GPU
      printNet - bool - whether or not to print network architecture out
      spatial - bool - whether to output an array containing varying distances across spatial dimensions
      spatial_shape - if given, output spatial shape. if None then spatial shape is determined automatically via spatial_factor (see below).
      spatial_factor - if given, specifies upsampling factor relative to the largest spatial extent of a convolutional layer. if None then resized to size of input images.
      spatial_order - spline order of filter for upsampling in spatial mode, by default 1 (bilinear).
      is_train - bool - [True] for training mode
      lr - float - initial learning rate
      beta1 - float - initial momentum term for adam
      version - 0.1 for latest, 0.0 was original
    '''
    BaseModel.initialize(self, use_gpu=use_gpu)

    self.model = model
    self.net = net
    self.use_gpu = use_gpu
    self.is_train = is_train
    self.spatial = spatial
    self.spatial_shape = spatial_shape
    self.spatial_order = spatial_order
    self.spatial_factor = spatial_factor

    self.model_name = '%s [%s]'%(model,net)
    if(self.model == 'net-lin'): # pretrained net + linear layer
      self.net = networks.PNetLin(use_gpu=use_gpu,pnet_rand=pnet_rand, pnet_tune=pnet_tune, pnet_type=net,use_dropout=True,spatial=spatial,version=version)
      kw = {}
      if not use_gpu:
        kw['map_location'] = 'cpu'
      if(model_path is None):
        import inspect
        # model_path = './PerceptualSimilarity/weights/v%s/%s.pth'%(version,net)
        model_path = os.path.abspath(os.path.join(inspect.getfile(self.initialize),  '..', 'lpips','weights/v%s/%s.pth'%(version,net)))

      if(not is_train):
        print('Loading model from: %s'%model_path)
        self.net.load_state_dict(torch.load(model_path, **kw))

    elif(self.model=='net'): # pretrained network
      assert not self.spatial, 'spatial argument not supported yet for uncalibrated networks'
      self.net = networks.PNet(use_gpu=use_gpu,pnet_type=net)
      self.is_fake_net = True
    elif(self.model in ['L2','l2']):
      self.net = networks.L2(use_gpu=use_gpu,colorspace=colorspace) # not really a network, only for testing
      self.model_name = 'L2'
    elif(self.model in ['DSSIM','dssim','SSIM','ssim']):
      self.net = networks.DSSIM(use_gpu=use_gpu,colorspace=colorspace)
      self.model_name = 'SSIM'
    else:
      raise ValueError("Model [%s] not recognized." % self.model)

    self.parameters = list(self.net.parameters())

    if self.is_train: # training mode
      # extra network on top to go from distances (d0,d1) => predicted human judgment (h*)
      self.rankLoss = networks.BCERankingLoss(use_gpu=use_gpu)
      self.parameters+=self.rankLoss.parameters
      self.lr = lr
      self.old_lr = lr
      self.optimizer_net = torch.optim.Adam(self.parameters, lr=lr, betas=(beta1, 0.999))
    else: # test mode
      self.net.eval()

    if(printNet):
      print('---------- Networks initialized -------------')
      networks.print_network(self.net)
      print('-----------------------------------------------')

  def forward_pair(self,in1,in2,retPerLayer=False):
    if(retPerLayer):
      return self.net.forward(in1,in2, retPerLayer=True)
    else:
      return self.net.forward(in1,in2)

  def forward(self, in0, in1, retNumpy=True):
    ''' Function computes the distance between image patches in0 and in1
    INPUTS
      in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
      retNumpy - [False] to return as torch.Tensor, [True] to return as numpy array
    OUTPUT
      computed distances between in0 and in1
    '''

    self.input_ref = in0
    self.input_p0 = in1

    if(self.use_gpu):
      self.input_ref = self.input_ref.cuda()
      self.input_p0 = self.input_p0.cuda()

    self.var_ref = Variable(self.input_ref,requires_grad=True)
    self.var_p0 = Variable(self.input_p0,requires_grad=True)

    self.d0 = self.forward_pair(self.var_ref, self.var_p0)
    self.loss_total = self.d0

    def convert_output(d0):
      if(retNumpy):
        ans = d0.cpu().data.numpy()
        if not self.spatial:
          ans = ans.flatten()
        else:
          assert(ans.shape[0] == 1 and len(ans.shape) == 4)
          return ans[0,...].transpose([1, 2, 0])          # Reshape to usual numpy image format: (height, width, channels)
        return ans
      else:
        return d0

    if self.spatial:
      L = [convert_output(x) for x in self.d0]
      spatial_shape = self.spatial_shape
      if spatial_shape is None:
        if(self.spatial_factor is None):
          spatial_shape = (in0.size()[2],in0.size()[3])
        else:
          spatial_shape = (max([x.shape[0] for x in L])*self.spatial_factor, max([x.shape[1] for x in L])*self.spatial_factor)
      
      L = [skimage.transform.resize(x, spatial_shape, order=self.spatial_order, mode='edge') for x in L]
      
      L = np.mean(np.concatenate(L, 2) * len(L), 2)
      return L
    else:
      return convert_output(self.d0)

