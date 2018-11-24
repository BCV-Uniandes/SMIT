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
class Discriminator(nn.Module):
  def __init__(self, config, debug=False):
    super(Discriminator, self).__init__()
    
    layers = []
    image_size = config.image_size
    conv_dim = config.d_conv_dim
    if not config.FIXED_D_CONV:
      conv_dim = conv_dim if image_size<256 else conv_dim//2
      conv_dim = conv_dim if image_size<512 else conv_dim//2      

    repeat_num = config.d_repeat_num      
    self.c_dim = config.c_dim
    self.config = config
    self.RafD = self.config.dataset_fake=='RafD'
    self.GRAY_DISC = 'GRAY_DISC' in config.GAN_options
    self.STYLE_DISC = 'STYLE_DISC' in config.GAN_options
    color_dim = config.color_dim if not self.GRAY_DISC else 1
    SN = 'SpectralNorm' in config.GAN_options
    self.DILATE = config.DISC_DILATE

    print_debug = lambda x,v: _print_debug(x, v, file=config.log)

    SpectralNorm = get_SN(SN)
    layers.append(SpectralNorm(nn.Conv2d(color_dim, conv_dim, kernel_size=4, stride=2, padding=1)))
    layers.append(nn.LeakyReLU(0.01, inplace=True))

    curr_dim = conv_dim
    _curr_dim = curr_dim
    for i in range(1, repeat_num):
      layers.append(SpectralNorm(nn.Conv2d(_curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
      layers.append(nn.LeakyReLU(0.01, inplace=True))
      curr_dim = curr_dim * 2 
      if self.DILATE and i==repeat_num-5:
        self.main0 = nn.Sequential(*layers)
        layers = []
        layers_dilate = []
        for k in range(3):
          layers_dilate.append(nn.Conv2d(curr_dim, curr_dim, dilation=2**(k+1), kernel_size=3, padding=2**(k+1), bias=False))
          layers_dilate.append(nn.InstanceNorm2d(curr_dim, affine=True))
          layers_dilate.append(nn.LeakyReLU(0.01, inplace=True))    
        _curr_dim = curr_dim * 2 
      else: _curr_dim = curr_dim


    k_size = int(image_size / np.power(2, repeat_num))
    layers_debug = layers
    if self.DILATE: 
      self.dilate = nn.Sequential(*layers_dilate)
    self.main = nn.Sequential(*layers)
    self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
    self.conv2 = nn.Conv2d(curr_dim, self.c_dim, kernel_size=k_size, bias=False)
    # if self.RafD: self.pose = nn.Conv2d(curr_dim, 1, kernel_size=k_size, bias=True)

    if self.STYLE_DISC:
      layers = []
      layers.append(nn.Linear(curr_dim*k_size*k_size, 256, bias=True))  
      layers.append(nn.Dropout(0.5))  
      layers.append(nn.Linear(256, 256, bias=True))  
      layers.append(nn.Dropout(0.5))
      layers.append(nn.Linear(256, config.style_dim))

      self.style = nn.Sequential(*layers)

    if debug:
      feed = to_var(torch.ones(1,color_dim,image_size,image_size), volatile=True, no_cuda=True)
      PRINT(config.log, '-- Discriminator:')
      if self.DILATE:
        # ipdb.set_trace()
        features0 = print_debug(feed, self.main0)
        features_dilated = print_debug(features0, self.dilate)
        # ipdb.set_trace()
        feed = torch.cat([features0, features_dilated], dim=1)
      features = print_debug(feed, self.main)
      _ = print_debug(features, [self.conv1])
      _ = print_debug(features, [self.conv2])
      if self.STYLE_DISC: 
        features = features.view(features.size(0), -1)
        _ = print_debug(features, self.style)


  def forward(self, x):
    if self.GRAY_DISC: x = x.mean(dim=1).unsqueeze(1)
    if self.DILATE:
      features0 = to_parallel(self.main0, x, self.config.GPU)
      features_dilated = to_parallel(self.dilate, features0, self.config.GPU)
      x = torch.cat([features0, features_dilated], dim=1)  
    features = to_parallel(self.main, x, self.config.GPU)
    out_real = to_parallel(self.conv1, features, self.config.GPU).squeeze()
    out_real = out_real.view(x.size(0), out_real.size(-2), out_real.size(-1))

    out_aux = to_parallel(self.conv2, features, self.config.GPU).squeeze()
    out_aux = out_aux.view(x.size(0), out_aux.size(-1))

    if self.STYLE_DISC: 
      h = h.view(h.size(0), -1)
      out_style = to_parallel(self.style, features, self.config.GPU).squeeze()
      out_style = out_style.view(x.size(0), out_style.size(-1))      
    else:
      out_style = None

    return [out_real], [out_aux], [out_style]



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
    SN = 'SpectralNorm' in config.GAN_options
    self.SpectralNorm = get_SN(SN)
    self.DILATE = config.DISC_DILATE

    print_debug = lambda x,v: _print_debug(x, v, file=config.log)

    self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
    self.cnns_main = nn.ModuleList()
    if self.DILATE:
      self.cnns_main0 = nn.ModuleList()
      self.cnns_dilate = nn.ModuleList()
    self.cnns_src = nn.ModuleList()
    self.cnns_aux = nn.ModuleList()
    for idx in range(config.MultiDis):
      if self.DILATE:
        cnns_main0, cnns_dilate, cnns_main, cnns_src, cnns_aux = self._make_net(idx)
        self.cnns_main0.append(cnns_main0)
        self.cnns_dilate.append(cnns_dilate)  
      else:
        cnns_main, cnns_src, cnns_aux = self._make_net(idx) 
      self.cnns_main.append(cnns_main[-1])
      self.cnns_src.append(cnns_src)
      self.cnns_aux.append(cnns_aux)

    if debug:
      feed = to_var(torch.ones(1, self.color_dim, self.image_size, self.image_size), volatile=True, no_cuda=True)
      if self.DILATE: modelList = zip(self.cnns_main0, self.cnns_dilate, self.cnns_main, self.cnns_src, self.cnns_aux)
      else: modelList = zip(self.cnns_main, self.cnns_src, self.cnns_aux)
      for idx, outs in enumerate(modelList):
        PRINT(config.log, '-- MultiDiscriminator ({}):'.format(idx))
        if self.DILATE:
          features0 = print_debug(feed, outs[0])
          features_dilated = print_debug(features0, outs[1])
          features = torch.cat([features0, features_dilated], dim=1)     
          features = print_debug(features, outs[-3])   
        else:
          features = print_debug(feed, outs[-3])
        # ipdb.set_trace()
        _ = print_debug(features, outs[-2])
        _ = print_debug(features, outs[-1]).view(feed.size(0), -1)     
        feed = self.downsample(feed)      
        
  def _make_net(self, idx=0):
    image_size = self.image_size/(2**(idx))   
    if self.config.FIXED_D_CONV:
      self.repeat_num = 6
    else:
      self.repeat_num = int(math.log(image_size,2)-1)
    self.conv_dim = self.conv_dim if image_size<256 else self.conv_dim//2
    self.conv_dim = self.conv_dim if image_size<512 else self.conv_dim//2    
    k_size = int(image_size / np.power(2, self.repeat_num))
    layers = [] 
    layers.append(self.SpectralNorm(nn.Conv2d(self.color_dim, self.conv_dim, kernel_size=4, stride=2, padding=1)))
    layers.append(nn.LeakyReLU(0.01, inplace=True))
    curr_dim = self.conv_dim
    _curr_dim = curr_dim
    for i in range(1, self.repeat_num):
      layers.append(self.SpectralNorm(nn.Conv2d(_curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
      layers.append(nn.LeakyReLU(0.01, inplace=True))
      curr_dim *= 2     
      # conv_size /= 2
      if self.DILATE and i==self.repeat_num-5:
        main0 = [nn.Sequential(*layers)]
        layers = []
        layers_dilate = []
        for k in range(3):
          layers_dilate.append(nn.Conv2d(curr_dim, curr_dim, dilation=2**(k+1), kernel_size=3, padding=2**(k+1), bias=False))
          layers_dilate.append(nn.InstanceNorm2d(curr_dim, affine=True))
          layers_dilate.append(nn.LeakyReLU(0.01, inplace=True))    
        _curr_dim = curr_dim * 2 
      else: _curr_dim = curr_dim      
    main = [nn.Sequential(*layers)]
    if self.DILATE: 
      dilate = [nn.Sequential(*layers_dilate)]
      main = main0+dilate+main
    src = nn.Sequential(*[nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)])
    aux = nn.Sequential(*[nn.Conv2d(curr_dim, self.c_dim, kernel_size=k_size, bias=False)])
    # ipdb.set_trace()

    return main, src, aux

  def forward(self, x):
    outs_src = []; outs_aux = []
    if self.DILATE: modelList = zip(self.cnns_main0, self.cnns_dilate, self.cnns_main, self.cnns_src, self.cnns_aux)
    else: modelList = zip(self.cnns_main, self.cnns_src, self.cnns_aux)
    for outs in modelList:
      if self.DILATE:
        features0 = to_parallel(outs[0], x, self.config.GPU)
        features_dilated = to_parallel(outs[1], features0, self.config.GPU)
        features = torch.cat([features0, features_dilated], dim=1)     
        main = to_parallel(outs[-3], features, self.config.GPU)     
      else:
        main = to_parallel(outs[-3], x, self.config.GPU)
      _src = to_parallel(outs[-2], main, self.config.GPU)
      _aux = to_parallel(outs[-1], main, self.config.GPU).view(main.size(0), -1)
      outs_src.append(_src)
      outs_aux.append(_aux)

      x = self.downsample(x)

    return outs_src, outs_aux, [None]*len(outs_aux)


#===============================================================================================#
#===============================================================================================#
class Generator(nn.Module):
  """Generator. Encoder-Decoder Architecture."""
  def __init__(self, config, debug=False, **kwargs):
    super(Generator, self).__init__()
    layers = []
    conv_dim = config.g_conv_dim
    repeat_num = config.g_repeat_num
    self.config = config
    self.image_size = config.image_size
    self.c_dim = config.c_dim    
    self.color_dim = config.color_dim
    self.style_dim = config.style_dim
    self.LayerNorm = 'LayerNorm' in config.GAN_options    
    self.InstanceNorm = 'InstanceNorm' in config.GAN_options    
    self.content_loss = 'content_loss' in config.GAN_options
    self.Attention = 'Attention' in config.GAN_options
    self.Attention2 = 'Attention2' in config.GAN_options #Attention before residual
    self.Attention3 = 'Attention3' in config.GAN_options #Attention after residual
    self.AttentionStyle = 'AttentionStyle' in config.GAN_options
    self.InterLabels = 'InterLabels' in config.GAN_options
    self.InterStyleLabels = 'InterStyleLabels' in config.GAN_options
    self.InterStyleConcatLabels = 'InterStyleConcatLabels' in config.GAN_options
    self.style_gen = 'style_gen' in config.GAN_options
    self.DRIT = 'DRIT' in config.GAN_options and not self.InterStyleLabels
    self.DRITZ = 'DRITZ' in config.GAN_options and not self.InterStyleLabels
    self.AdaIn = 'AdaIn' in config.GAN_options and not 'DRIT' in config.GAN_options
    self.AdaIn2 = 'AdaIn2' in config.GAN_options and not 'DRIT' in config.GAN_options 
    self.AdaIn3 = 'AdaIn3' in config.GAN_options
    if self.AdaIn2: AdaIn_res=1
    elif self.AdaIn3: AdaIn_res=2
    else: AdaIn_res = 0
    if self.InterLabels or self.InterStyleConcatLabels: in_dim=self.color_dim
    elif self.DRITZ: in_dim=self.color_dim+self.c_dim+self.style_dim
    else: in_dim=self.color_dim+self.c_dim

    layers.append(nn.Conv2d(in_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
    layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
    layers.append(nn.ReLU(inplace=True))     

    # Down-Sampling
    if self.config.FIXED_G_CONV:
      conv_repeat = 2  
    else:
      conv_repeat = int(math.log(self.image_size, 2))-5 if self.image_size>64 else 2
    curr_dim = conv_dim
    for i in range(conv_repeat):
      layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
      layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
      layers.append(nn.ReLU(inplace=True))
      curr_dim = curr_dim * 2

    if (self.InterStyleConcatLabels and self.AdaIn2 and self.content_loss) or self.Attention2:
      self.content = nn.Sequential(*layers)
      layers = []
      # ipdb.set_trace()

    # Bottleneck
    for i in range(repeat_num):
      # ipdb.set_trace()
      layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, AdaIn=AdaIn_res))
      # if i==int(repeat_num/2)-1 and not self.AdaIn2 and self.AdaIn: AdaIn_res=1
      if i==int(repeat_num/2)-1 and not self.AdaIn2 and (self.InterLabels or self.DRIT or self.InterStyleLabels):
        # if self.AdaIn: AdaIn_res=1
        self.content = nn.Sequential(*layers)
        layers = []
        curr_dim = curr_dim+self.c_dim    
        if self.InterLabels and self.DRIT and not self.InterStyleLabels: curr_dim += self.c_dim   

    # Up-Sampling
    for i in range(conv_repeat):
      layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
      if not self.AdaIn or self.InstanceNorm: layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True)) #undesirable to generate images in vastly different styles
      elif self.LayerNorm: layers.append(LayerNorm(curr_dim//2))
      # layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True)) #undesirable to generate images in vastly different styles
      layers.append(nn.ReLU(inplace=True))        

      curr_dim = curr_dim // 2

    if self.Attention:
      self.main = nn.Sequential(*layers)
      self.layers = layers

      layers0 = []
      layers0.append(nn.Conv2d(curr_dim, self.color_dim, kernel_size=7, stride=1, padding=3, bias=False))
      layers0.append(nn.Tanh())
      self.layers0 = layers0
      self.img_reg = nn.Sequential(*layers0)
      
      layers1 = []       
      layers1.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
      layers1.append(nn.Sigmoid())
      self.layers1 = layers1
      self.attetion_reg = nn.Sequential(*layers1)

    elif self.AdaIn:
      self.main = nn.Sequential(*layers)
      self.layers = layers

      if self.Attention2:
        layers1.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers1.append(nn.Sigmoid())
        self.layers1 = layers1
        self.attetion_reg = nn.Sequential(*layers1)

        layers.append(nn.Conv2d(curr_dim, self.color_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.layers = layers
        self.img_reg = nn.Sequential(*layers)  

      elif self.Attention3:
        layers1.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers1.append(nn.Sigmoid())
        self.layers1 = layers1
        self.attetion_reg = nn.Sequential(*layers1)

        layers0.append(nn.Conv2d(curr_dim, self.color_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers0.append(nn.Tanh())
        self.layers0 = layers0
        self.img_reg = nn.Sequential(*layers0)          


      else:
        layers0 = []
        layers0.append(nn.Conv2d(curr_dim, self.color_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers0.append(nn.Tanh())
        self.layers0 = layers0
        self.img_reg = nn.Sequential(*layers0)              

    else:
      layers.append(nn.Conv2d(curr_dim, self.color_dim, kernel_size=7, stride=1, padding=3, bias=False))
      layers.append(nn.Tanh())
      self.layers = layers
      self.main = nn.Sequential(*layers)

    if debug and not self.AdaIn:
      self.debug()

  def debug(self):

      print_debug = lambda x,v: _print_debug(x, v, file=self.config.log)

      PRINT(self.config.log, '-- Generator:')
      if self.InterLabels and self.DRIT: 
        c_dim = self.c_dim*2
        in_dim = self.color_dim
      elif self.InterLabels or self.InterStyleConcatLabels:
        c_dim = self.c_dim
        in_dim = self.color_dim
      elif self.DRITZ:
        c_dim = self.c_dim
        in_dim = self.color_dim+self.c_dim +self.style_dim     
      elif self.DRIT:
        c_dim = self.c_dim
        in_dim = self.color_dim+self.c_dim
      else:
        in_dim = self.color_dim+self.c_dim
        c_dim = 0

      if self.InterLabels or self.DRIT: 
        feed = to_var(torch.ones(1,in_dim, self.image_size,self.image_size), volatile=True, no_cuda=True)
        feed = print_debug(feed, self.content)
        c = to_var(torch.ones(1, c_dim, feed.size(2), feed.size(3)), volatile=True, no_cuda=True)
        feed = torch.cat([feed, c], dim=1)
      else: 
        feed = to_var(torch.ones(1,in_dim,self.image_size,self.image_size), volatile=True, no_cuda=True)
      

      if self.content_loss or self.Attention2:
        feed = print_debug(feed, self.content)        
 
      if not self.Attention2: features = print_debug(feed, self.layers)
      else: features=feed
      if self.Attention or self.Attention3: 
        _ = print_debug(features, self.layers0)
        _ = print_debug(features, self.layers1)      
      elif self.Attention2: 
        _ = print_debug(features, self.layers)
        _ = print_debug(features, self.layers1)              
      elif self.AdaIn: 
        _ = print_debug(features, self.layers0)        

  def forward(self, x, c, stochastic=None, CONTENT=False, JUST_CONTENT=False):
    # replicate spatially and concatenate domain information
    if self.InterLabels or self.InterStyleConcatLabels:
      x_cat = x
    elif self.DRITZ:
      c = c.unsqueeze(2).unsqueeze(3)
      c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
      stochastic = stochastic.unsqueeze(2).unsqueeze(3)
      stochastic = stochastic.expand(stochastic.size(0), stochastic.size(1), x.size(2), x.size(3))      
      x_cat = torch.cat([x, c, stochastic], dim=1)      
    else:
      c = c.unsqueeze(2).unsqueeze(3)
      c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
      # ipdb.set_trace()
      x_cat = torch.cat([x, c], dim=1)

    if self.InterStyleLabels or self.InterLabels:
      content = self.content(x_cat)
      # content = to_parallel(self.content, x_cat, self.config.GPU)
      if self.InterStyleLabels:
        if stochastic.size(-1)!=content.size(-1):
          stochastic = stochastic.repeat(1,1,content.size(-1)//stochastic.size(-1))        
        c = stochastic*c.unsqueeze(2)
      else: 
        c = c.unsqueeze(2)
      c = c.unsqueeze(3)
      c = c.expand(c.size(0), c.size(1), content.size(2), content.size(3))
      x_cat = torch.cat([content, c], dim=1)

      if self.DRIT and not self.InterStyleLabels:
        if stochastic.size(-1)!=content.size(-1):
          stochastic = stochastic.repeat(1,1,content.size(-1)//stochastic.size(-1))
        stochastic = stochastic.unsqueeze(3)
        stochastic = stochastic.expand(stochastic.size(0), stochastic.size(1), stochastic.size(2), stochastic.size(2))
        x_cat = torch.cat([x_cat, stochastic], dim=1)    

    elif self.DRIT:
      content = self.content(x_cat)
      # content = to_parallel(self.content, x_cat, self.config.GPU)
      if self.InterLabels:
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), content.size(2), content.size(3))        
        if stochastic.size(-1)!=content.size(-1):
          stochastic = stochastic.repeat(1,1,content.size(-1)//stochastic.size(-1))        
        stochastic = stochastic.unsqueeze(3)
        stochastic = stochastic.expand(stochastic.size(0), stochastic.size(1), stochastic.size(2), stochastic.size(2))
        x_cat = torch.cat([content, c, stochastic], dim=1)        
      else:
        content = self.content(x_cat)
        # content = to_parallel(self.content, x_cat, self.config.GPU)
        stochastic = stochastic.unsqueeze(3)
        stochastic = stochastic.expand(stochastic.size(0), stochastic.size(1), stochastic.size(2), stochastic.size(2))
        x_cat = torch.cat([content, stochastic], dim=1)                

    elif self.content_loss or self.Attention2:
      content = self.content(x_cat)
      x_cat = content

    if JUST_CONTENT:
      return content

    if self.Attention or self.Attention2 or self.Attention3: 
      if not self.Attention2: features = self.main(x_cat)
      else: features = x_cat
      fake_img = to_parallel(self.img_reg, features, self.config.GPU)
      mask_img = to_parallel(self.attetion_reg, features, self.config.GPU)
      fake_img = mask_img * x + (1 - mask_img) * fake_img
      output = [fake_img, mask_img]

    elif self.AdaIn:  
      features = self.main(x_cat)
      # output = [self.img_reg(features)]
      output = [to_parallel(self.img_reg, features, self.config.GPU)]

    else: 
      # output = [self.main(x_cat)]
      output = [to_parallel(self.main, x_cat, self.config.GPU)]

    if CONTENT:
      output += [content]

    return output

#===============================================================================================#
#===============================================================================================#
class AdaInGEN(nn.Module):
  def __init__(self, config, STYLE_ENC = None, debug=False):
    super(AdaInGEN, self).__init__()

    conv_dim = config.g_conv_dim
    mlp_dim = config.mlp_dim
    self.config = config
    self.color_dim = config.color_dim
    self.image_size = config.image_size
    self.style_dim = config.style_dim    
    self.c_dim = config.c_dim
    self.s_dim = config.c_dim if config.style_dim!=8 and config.style_dim!=20 and config.style_dim!=16 else 1
    self.STYLE_DISC = 'STYLE_DISC' in config.GAN_options
    self.InterStyleConcatLabels= 'InterStyleConcatLabels' in config.GAN_options
    self.InterStyleMulLabels= 'InterStyleMulLabels' in config.GAN_options
    if STYLE_ENC is None and self.config.lambda_style!=0: self.enc_style = StyleEncoder(config, debug=debug)
    else: self.enc_style = STYLE_ENC

    print_debug = lambda x,v: _print_debug(x, v, file=config.log)

    self.generator = Generator(config, debug=False)
    if self.style_dim==0: 
      in_dim=self.c_dim
    elif config.style_dim!=8 and config.style_dim!=20 and config.style_dim!=16:
      in_dim = self.style_dim*self.c_dim
      if self.InterStyleConcatLabels: in_dim *=2
    elif self.InterStyleConcatLabels:
      in_dim = self.style_dim+self.c_dim
    else:
      in_dim = self.style_dim
    self.adain_net = MLP(config, in_dim, self.get_num_adain_params(self.generator), mlp_dim, 3, norm='none', activ='relu', debug=debug)
    if debug: self.debug()

  def debug(self):
    feed = to_var(torch.ones(1,self.color_dim,self.image_size,self.image_size), volatile=True, no_cuda=True)
    label = to_var(torch.ones(1,self.c_dim), volatile=True, no_cuda=True)
    style = to_var(self.random_style(feed), volatile=True, no_cuda=True) 
    self.apply_style(feed, style, label=label)
    self.generator.debug()
    
  def forward(self, x, c, stochastic=None, CONTENT=False, JUST_CONTENT=False):
    if stochastic is None:
      style = self.get_style(x)
    else:
      style = stochastic
      
    self.apply_style(x, style, label=c)
    return self.generator(x, c, stochastic=style, CONTENT=CONTENT, JUST_CONTENT=JUST_CONTENT)

  def random_style(self, x, interp=False):
    if type(x)==int: number = x
    else: number = x.size(0)
    if self.style_dim!=8 and self.style_dim!=20 and self.style_dim!=16: 
      z = torch.randn(number, self.s_dim, self.style_dim)

    else:
      z = torch.randn(number, self.style_dim)

    return z

  def get_style(self, x, volatile=False):
    if self.config.lambda_style==0: return None
    style = self.enc_style(x)
    if self.STYLE_DISC: style = style[-1]
    style = style.view(style.size(0), self.s_dim, -1) if self.s_dim!=1 else style.view(style.size(0), -1)
    return style

  def apply_style(self, image, style, label=None):
    if self.style_dim==0:
      style = label.view(label.size(0),-1)
    elif self.InterStyleConcatLabels:
      # ipdb.set_trace()
      if self.style_dim==8 or self.style_dim==20 or self.style_dim==16: 
        label = label.view(label.size(0),-1)
        style = style.view(style.size(0),-1)
        # label = label.unsqueeze(1) #label.view(label.size(0),-1)
        # style = style.unsqueeze(-1).repeat(1,1,label.size(-1)) #style.view(style.size(0),-1)
        # style = torch.cat([style, label], dim=1)        

      else:
        label = label.unsqueeze(-1).expand_as(style)
      style = torch.cat([style, label], dim=-1)
    if self.InterStyleMulLabels:
      label = label.unsqueeze(-1).expand_as(style)
      style = style*label
    adain_params = self.adain_net(style)
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
class StyleEncoder(nn.Module):
  """Generator. Encoder-Decoder Architecture."""
  def __init__(self, config, debug=False):
    super(StyleEncoder, self).__init__()
    image_size = config.image_size
    conv_dim = config.g_conv_dim//2
    color_dim = config.color_dim if not 'AttentionStyle' in config.GAN_options else 1
    self.GRAY_STYLE = 'GRAY_STYLE' in config.GAN_options
    color_dim = color_dim if not self.GRAY_STYLE else 1    
    style_dim = config.style_dim
    self.DRITZ = 'DRITZ' in config.GAN_options    
    self.c_dim = config.c_dim
    self.s_dim = style_dim if style_dim==8 or style_dim==20 or style_dim==16  or self.DRITZ else config.c_dim*style_dim
    self.FC = 'FC' in config.GAN_options
    self.config = config

    print_debug = lambda x,v: _print_debug(x, v, file=config.log)

    layers = []
    norm = 'none'
    activ = 'relu'
    pad_type = 'reflect'
    layers.append(Conv2dBlock(color_dim, conv_dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type))
    # layers.append(nn.Conv2d(color_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
    # layers.append(nn.ReLU(inplace=True))

    # Down-Sampling
    if style_dim==4 or style_dim==20 or style_dim==16:
      down = [2,1]
    else:
      down =[3,3]
    conv_repeat = int(math.log(image_size, 2))-down[0] #1 until 2x2, 2 for 4x4
    curr_dim = conv_dim
    for i in range(conv_repeat):
      layers.append(Conv2dBlock(curr_dim, curr_dim*2, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type))
      # layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
      # layers.append(nn.ReLU(inplace=True))      
      curr_dim = curr_dim * 2    

    layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
    curr_dim = curr_dim * 2 
    
    if style_dim==1 or style_dim==8: 
      layers.append(nn.AdaptiveAvgPool2d(1)) # global average pooling

    if self.FC:
      # ipdb.set_trace()
      conv_repeat = conv_repeat+down[1]
      k_size = int(image_size / np.power(2, conv_repeat))
      layers0=[]
      layers0.append(nn.Linear(curr_dim*k_size*k_size, 256, bias=True))  
      layers0.append(nn.Dropout(0.5))  
      layers0.append(nn.Linear(256, 256, bias=True))  
      layers0.append(nn.Dropout(0.5))
      self.fc = nn.Sequential(*layers0)
      self.style_mu = nn.Linear(256, self.s_dim)
    else:
      self.style_mu = nn.Conv2d(curr_dim, self.c_dim, kernel_size=1, stride=1, padding=0)
    self.main = nn.Sequential(*layers)

    if debug:
      feed = to_var(torch.ones(1,color_dim,image_size,image_size), volatile=True, no_cuda=True)
      PRINT(config.log, '-- StyleEncoder:')
      features = print_debug(feed, layers)
      if self.FC:
        fc_in = features.view(features.size(0), -1)
        features = print_debug(fc_in, layers0)
      _ = print_debug(features, [self.style_mu]) 

  def forward(self, x):
    if self.GRAY_STYLE: x = x.mean(dim=1).unsqueeze(1)
    features = to_parallel(self.main, x, self.config.GPU)
    if self.FC:
      fc_input = features.view(features.size(0), -1)
      features = to_parallel(self.fc, fc_input, self.config.GPU)

    style_mu = to_parallel(self.style_mu, features, self.config.GPU)
    return style_mu
    
#===============================================================================================#
#===============================================================================================#
class MLP(nn.Module):
  def __init__(self, config, input_dim, output_dim, dim, n_blk, norm='none', activ='relu', debug=False):

    super(MLP, self).__init__()
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
      PRINT(config.log, '-- MLP:')
      _ = print_debug(feed, self._model)    

  def forward(self, x):
    return to_parallel(self.model, x.view(x.size(0), -1), self.config.GPU)