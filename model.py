import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.utils import get_SN
from models.utils import print_debug as _print_debug
from models.spectral import SpectralNorm as SpectralNormalization
import ipdb
import math
from misc.utils import PRINT, to_cuda, to_var,to_parallel
from misc.blocks import AdaptiveInstanceNorm2d, ResidualBlock, LinearBlock, Conv2dBlock, LayerNorm

#===============================================================================================#
#===============================================================================================#
class MultiDiscriminator(nn.Module):
  # Multi-scale discriminator architecture
  def __init__(self, config, debug=False):
    super(MultiDiscriminator, self).__init__()

    self.image_size = config.image_size
    self.conv_dim = config.d_conv_dim
    self.repeat_num = config.d_repeat_num    
    self.c_dim = config.c_dim
    self.color_dim = config.color_dim
    self.config = config
    SN = True
    self.SpectralNorm = get_SN(SN)

    print_debug = lambda x,v: _print_debug(x, v, file=config.log)

    self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
    self.cnns_main = nn.ModuleList()
    self.cnns_src = nn.ModuleList()
    self.cnns_aux = nn.ModuleList()
    for idx in range(config.MultiDis):
      cnns_main, cnns_src, cnns_aux = self._make_net(idx) 
      self.cnns_main.append(cnns_main)
      self.cnns_src.append(cnns_src)
      self.cnns_aux.append(cnns_aux)

    if debug:
      feed = to_var(torch.ones(1, self.color_dim, self.image_size, self.image_size), volatile=True, no_cuda=True)
      modelList = zip(self.cnns_main, self.cnns_src, self.cnns_aux)
      for idx, outs in enumerate(modelList):
        PRINT(config.log, '-- MultiDiscriminator ({}):'.format(idx))
        features = print_debug(feed, outs[-3])
        # ipdb.set_trace()
        _ = print_debug(features, outs[-2])
        _ = print_debug(features, outs[-1]).view(feed.size(0), -1)     
        feed = self.downsample(feed)      
        
  def _make_net(self, idx=0):
    image_size = self.image_size/(2**(idx))   
    self.repeat_num = int(math.log(image_size,2)-1)
    self.conv_dim = self.conv_dim if image_size<256 else self.conv_dim//2
    self.conv_dim = self.conv_dim if image_size<512 else self.conv_dim//2    
    k_size = int(image_size / np.power(2, self.repeat_num))
    layers = [] 
    layers.append(self.SpectralNorm(nn.Conv2d(self.color_dim, self.conv_dim, kernel_size=4, stride=2, padding=1)))
    layers.append(nn.LeakyReLU(0.01, inplace=True))
    curr_dim = self.conv_dim
    for i in range(1, self.repeat_num):
      layers.append(self.SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
      layers.append(nn.LeakyReLU(0.01, inplace=True))
      curr_dim *= 2     
    main = nn.Sequential(*layers)
    src = nn.Sequential(*[nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)])
    aux = nn.Sequential(*[nn.Conv2d(curr_dim, self.c_dim, kernel_size=k_size, bias=False)])

    return main, src, aux

  def forward(self, x):
    outs_src = []; outs_aux = []
    modelList = zip(self.cnns_main, self.cnns_src, self.cnns_aux)
    for outs in modelList:
      main = to_parallel(outs[0], x, self.config.GPU)
      _src = to_parallel(outs[1], main, self.config.GPU)
      _aux = to_parallel(outs[2], main, self.config.GPU).view(main.size(0), -1)
      outs_src.append(_src)
      outs_aux.append(_aux)

      x = self.downsample(x)

    return outs_src, outs_aux, 


#===============================================================================================#
#===============================================================================================#
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
    AdaIn_res = 0 if self.Deterministic else 1

    conv_dim = config.g_conv_dim
    if config.Slim_Generator: conv_dim = conv_dim if config.image_size<256 else conv_dim//2

    layers.append(nn.Conv2d(self.color_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
    layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
    layers.append(nn.ReLU(inplace=True))     

    # Down-Sampling
    if config.Slim_Generator:
      conv_repeat = int(math.log(self.image_size, 2))-5 if self.image_size>64 else 2
    else:
      conv_repeat = 2  
    curr_dim = conv_dim
    for i in range(conv_repeat):
      layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
      layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
      layers.append(nn.ReLU(inplace=True))
      curr_dim = curr_dim * 2

    # Bottleneck
    for i in range(repeat_num):
      layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, AdaIn=AdaIn_res))

    # Up-Sampling
    for i in range(conv_repeat):
      layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
      if not self.Deterministic: layers.append(LayerNorm(curr_dim//2))
      else: layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True)) #undesirable to generate images in vastly different styles
      layers.append(nn.ReLU(inplace=True))        
      curr_dim = curr_dim // 2

    self.main = nn.Sequential(*layers)

    layers0 = []
    layers0.append(nn.Conv2d(curr_dim, self.color_dim, kernel_size=7, stride=1, padding=3, bias=False))
    layers0.append(nn.Tanh())
    self.img_reg = nn.Sequential(*layers0)
    
    layers1 = []       
    layers1.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
    layers1.append(nn.Sigmoid())
    self.attn_reg = nn.Sequential(*layers1)           

    if debug and self.Deterministic:
      self.debug()

  def debug(self):
    print_debug = lambda x,v: _print_debug(x, v, file=self.config.log)
    PRINT(self.config.log, '-- Generator:')
    feed = to_var(torch.ones(1,self.color_dim,self.image_size,self.image_size), volatile=True, no_cuda=True)
    features = print_debug(feed, self.main)
    _ = print_debug(features, self.img_reg)
    _ = print_debug(features, self.attn_reg)      
      
  def forward(self, x):
    features = self.main(x)
    fake_img = to_parallel(self.img_reg, features, self.config.GPU)
    mask_img = to_parallel(self.attn_reg, features, self.config.GPU)
    fake_img = mask_img * x + (1 - mask_img) * fake_img
    output = [fake_img, mask_img]
    return output

#===============================================================================================#
#===============================================================================================#
class AdaInGEN(nn.Module):
  def __init__(self, config, debug=False):
    super(AdaInGEN, self).__init__()

    conv_dim = config.g_conv_dim
    dc_dim = config.dc_dim
    self.config = config
    self.color_dim = config.color_dim
    self.image_size = config.image_size
    self.style_dim = config.style_dim    
    self.c_dim = config.c_dim
    self.Deterministic = config.Deterministic
    print_debug = lambda x,v: _print_debug(x, v, file=config.log)

    self.generator = Generator(config, debug=False)
    if self.Deterministic:
      in_dim = self.c_dim
    else:
      in_dim = self.style_dim+self.c_dim
    self.adain_net = DC(config, in_dim, self.get_num_adain_params(self.generator), dc_dim, 3, norm='none', activ='relu', debug=debug)
    if debug: self.debug()

  def debug(self):
    feed = to_var(torch.ones(1,self.color_dim,self.image_size,self.image_size), volatile=True, no_cuda=True)
    label = to_var(torch.ones(1,self.c_dim), volatile=True, no_cuda=True)
    style = to_var(self.random_style(feed), volatile=True, no_cuda=True) 
    self.apply_style(feed, label, style)
    self.generator.debug()
    
  def forward(self, x, c, stochastic):     
    self.apply_style(x, c, stochastic)
    return self.generator(x)

  def random_style(self, x):
    if type(x)==int: number = x
    else: number = x.size(0)
    z = torch.randn(number, self.style_dim)
    return z

  def apply_style(self, image, label, style):
    label = label.view(label.size(0),-1)
    style = style.view(style.size(0),-1)
    if self.Deterministic:
      input_adain = label
    else:
      input_adain = torch.cat([style, label], dim=-1)
    adain_params = self.adain_net(input_adain)
    self.assign_adain_params(adain_params, self.generator)


  def assign_adain_params(self, adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
      if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
        mean = adain_params[:, :m.num_features]
        std = adain_params[:, m.num_features:2*m.num_features]
        m.bias = mean.contiguous().view(-1)
        m.weight = std.contiguous().view(-1)
        if adain_params.size(1) > 2*m.num_features:
          adain_params = adain_params[:, 2*m.num_features:]

  def get_num_adain_params(self, model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
      if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
        num_adain_params += 2*m.num_features
    return num_adain_params    
    
#===============================================================================================#
#===============================================================================================#
class DC(nn.Module):
  def __init__(self, config, input_dim, output_dim, dim, n_blk, norm='none', activ='relu', debug=False):

    super(DC, self).__init__()
    self.config = config
    self._model = []
    self._model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
    for i in range(n_blk - 2):
      self._model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
    self._model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
    self.model = nn.Sequential(*self._model)

    print_debug = lambda x,v: _print_debug(x, v, file=config.log)

    if debug:
      feed = to_var(torch.ones(1,input_dim), volatile=True, no_cuda=True)
      PRINT(config.log, '-- DC:')
      _ = print_debug(feed, self._model)    

  def forward(self, x):
    return to_parallel(self.model, x.view(x.size(0), -1), self.config.GPU)