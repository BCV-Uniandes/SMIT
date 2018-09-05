import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
import numpy as np
from models.spectral import SpectralNorm as SpectralNormalization
import ipdb
import math
from misc.utils import to_cuda, to_var

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

class ResidualBlock(nn.Module):
  """Residual Block."""
  def __init__(self, dim_in, dim_out, AdaIn=False):
    super(ResidualBlock, self).__init__()
    norm1 = AdaptiveInstanceNorm2d(dim_out) if AdaIn else nn.InstanceNorm2d(dim_out, affine=True)
    norm2 = AdaptiveInstanceNorm2d(dim_out) if AdaIn else nn.InstanceNorm2d(dim_out, affine=True)
    self.main = nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
      norm1,
      nn.ReLU(inplace=True),
      nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
      norm2)

  def forward(self, x):
    return x + self.main(x)

#===============================================================================================#
#===============================================================================================#
class Discriminator(nn.Module):
  """Discriminator. PatchGAN."""
  def __init__(self, config, debug=False):
    super(Discriminator, self).__init__()
    
    layers = []
    image_size = config.image_size
    conv_dim = config.d_conv_dim
    repeat_num = config.d_repeat_num    
    self.c_dim = config.c_dim
    self.s_dim = config.c_dim if config.style_dim!=8 else 1
    color_dim = config.color_dim
    SN = 'SpectralNorm' in config.GAN_options
    self.StyleDisc = 'StyleDisc' in config.GAN_options
    if self.StyleDisc: 
      style_dim = config.style_dim
      self.FC = 'FC' in config.GAN_options
    SpectralNorm = get_SN(SN)
    layers.append(SpectralNorm(nn.Conv2d(color_dim, conv_dim, kernel_size=4, stride=2, padding=1)))
    layers.append(nn.LeakyReLU(0.01, inplace=True))

    curr_dim = conv_dim
    for i in range(1, repeat_num):
      layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
      layers.append(nn.LeakyReLU(0.01, inplace=True))
      curr_dim = curr_dim * 2     

    k_size = int(image_size / np.power(2, repeat_num))
    layers_debug = layers
    self.main = nn.Sequential(*layers)
    self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
    self.conv2 = nn.Conv2d(curr_dim, self.c_dim, kernel_size=k_size, bias=False)

    if self.StyleDisc: 
      if style_dim==1 or style_dim==8: 
        layers.append(nn.AdaptiveAvgPool2d(1)) # global average pooling
      if self.FC:
        layers0=[]
        layers0.append(nn.Linear(curr_dim*k_size*k_size, 256, bias=True))  
        layers0.append(nn.Dropout(0.5))  
        layers0.append(nn.Linear(256, 256, bias=True))  
        layers0.append(nn.Dropout(0.5))
        self.fc = nn.Sequential(*layers0)
        out_dim = self.s_dim*style_dim  
        self.style_mu = nn.Linear(256, out_dim)   
      else:
        self.style_mu = nn.Conv2d(curr_dim, self.s_dim, kernel_size=3, stride=1, padding=1, bias=False)
    if debug:
      feed = to_var(torch.ones(1,color_dim,image_size,image_size), volatile=True, no_cuda=True)
      print('-- Discriminator:')
      features = print_debug(feed, layers_debug)
      _ = print_debug(features, [self.conv1])
      _ = print_debug(features, [self.conv2])    

      if self.StyleDisc:
        if self.FC:
          fc_in = features.view(features.size(0), -1)
          features = print_debug(fc_in, layers0)
        _ = print_debug(features, [self.style_mu])         

  def forward(self, x, get_style=False):
    h = self.main(x)
    out_real = self.conv1(h).squeeze()
    out_real = out_real.view(x.size(0), out_real.size(-2), out_real.size(-1))

    out_aux = self.conv2(h).squeeze()
    out_aux = out_aux.view(x.size(0), out_aux.size(-1))

    if self.StyleDisc:
      if self.FC:
        fc_input = h.view(h.size(0), -1)
        h = self.fc(fc_input)
      style_mu = self.style_mu(h)   
      style = style_mu
      if get_style: 
        style = style.view(style[0].size(0), self.c_dim, -1)
        return style
    else:
      style = None

    return [out_real], [out_aux], [style]



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
    self.DRITZ = 'DRITZ' in config.GAN_options
    self.s_dim = 1 if config.style_dim==8 or self.DRITZ else config.c_dim
    self.color_dim = config.color_dim
    SN = 'SpectralNorm' in config.GAN_options
    self.StyleDisc = 'StyleDisc' in config.GAN_options
    if self.StyleDisc: 
      self.style_dim = config.style_dim
      self.FC = 'FC' in config.GAN_options
    self.SpectralNorm = get_SN(SN)
    
    self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
    self.cnns_main = nn.ModuleList() 
    self.cnns_src = nn.ModuleList()
    self.cnns_aux = nn.ModuleList()
    if self.StyleDisc: self.cnns_sty = nn.ModuleList()
    else: self.cnns_sty = []
    # ipdb.set_trace()
    for idx in range(config.MultiDis):
      self.cnns_main.append(self._make_net(idx)[0])
      self.cnns_src.append(self._make_net(idx)[1])
      self.cnns_aux.append(self._make_net(idx)[2])
      self.cnns_sty.append(self._make_net(idx)[3])

    if debug:
      feed = to_var(torch.ones(1, self.color_dim, self.image_size, self.image_size), volatile=True, no_cuda=True)
      # ipdb.set_trace()
      for idx, (model, src, aux, sty) in enumerate(zip(self.cnns_main, self.cnns_src, self.cnns_aux, self.cnns_sty)):
        # ipdb.set_trace()
        print('-- MultiDiscriminator ({}):'.format(idx))
        features = print_debug(feed, model)
        _ = print_debug(features, src)
        _ = print_debug(features, aux).view(feed.size(0), -1)     
        if self.StyleDisc: _ = print_debug(features.view(feed.size(0), -1), sty)
        feed = self.downsample(feed)      
        
  def _make_net(self, idx=0):
    conv_size = self.image_size/(2**(idx))   
    layers = [] 
    layers.append(self.SpectralNorm(nn.Conv2d(self.color_dim, self.conv_dim, kernel_size=4, stride=2, padding=1)))
    layers.append(nn.LeakyReLU(0.01, inplace=True))
    curr_dim = self.conv_dim
    for i in range(1, self.repeat_num-1):
      layers.append(self.SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
      layers.append(nn.LeakyReLU(0.01, inplace=True))
      curr_dim *= 2     
      conv_size /= 2

    main = nn.Sequential(*layers)
    src = nn.Sequential(*[nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)])
    aux = nn.Sequential(*[nn.Conv2d(curr_dim, self.c_dim, kernel_size=conv_size//2, bias=False)])
    if self.StyleDisc: 
      layers = []
      if self.style_dim==1 or self.style_dim==8: 
        layers.append(nn.AdaptiveAvgPool2d(1)) # global average pooling
      if self.FC:
        # ipdb.set_trace()
        k_size = int(self.image_size / (np.power(2, self.repeat_num-1)*(2**idx)))
        layers.append(nn.Linear(curr_dim*k_size*k_size, 256, bias=True))  
        layers.append(nn.Dropout(0.5))  
        layers.append(nn.Linear(256, 256, bias=True))  
        layers.append(nn.Dropout(0.5))
        # self.fc = nn.Sequential(*layers0)
        if self.style_dim==8:
          out_dim = self.style_dim
        else:
          out_dim = self.s_dim*self.style_dim 
        layers.append(nn.Linear(256, out_dim))
      else:
        layers.append(nn.Conv2d(curr_dim, self.s_dim, kernel_size=3, stride=1, padding=1, bias=False))
      sty =  nn.Sequential(*layers)
    else:
      sty = None

    return main, src, aux, sty

  def forward(self, x, get_style=False):
    outs_src = []; outs_aux = []; outs_sty = []
    # ipdb.set_trace()
    for model, src, aux, sty in zip(self.cnns_main, self.cnns_src, self.cnns_aux, self.cnns_sty):
      # ipdb.set_trace()
      main = model(x)
      _src = src(main)
      _aux = aux(main).view(main.size(0), -1)
      outs_src.append(_src)
      outs_aux.append(_aux)
      if self.StyleDisc:
        if self.FC:
          main = main.view(main.size(0), -1) 
        _sty = sty(main)
        outs_sty.append(_sty)
      else:
        outs_sty.append(None)

      x = self.downsample(x)

    if get_style and self.StyleDisc: 
      outs_sty = [out.unsqueeze(0) for out in outs_sty]
      outs_sty = torch.cat(outs_sty,dim=0).mean(dim=0)
      outs_sty = [outs_sty.view(outs_sty[0].size(0), self.c_dim, -1)]
      return outs_sty
    return outs_src, outs_aux, outs_sty


#===============================================================================================#
#===============================================================================================#
class Generator(nn.Module):
  """Generator. Encoder-Decoder Architecture."""
  def __init__(self, config, debug=False):
    super(Generator, self).__init__()
    layers = []
    conv_dim = config.g_conv_dim
    repeat_num = config.g_repeat_num
    self.image_size = config.image_size
    self.c_dim = config.c_dim    
    self.color_dim = config.color_dim
    self.style_dim = config.style_dim
    self.Attention = 'Attention' in config.GAN_options
    self.InterLabels = 'InterLabels' in config.GAN_options
    self.InterStyleLabels = 'InterStyleLabels' in config.GAN_options
    self.InterStyleConcatLabels = 'InterStyleConcatLabels' in config.GAN_options
    self.style_gen = 'style_gen' in config.GAN_options
    self.DRIT = 'DRIT' in config.GAN_options and not self.InterStyleLabels
    self.DRITZ = 'DRITZ' in config.GAN_options and not self.InterStyleLabels
    self.AdaIn = 'AdaIn' in config.GAN_options and not 'DRIT' in config.GAN_options
    self.AdaIn2 = 'AdaIn2' in config.GAN_options and not 'DRIT' in config.GAN_options 
    self.AdaIn3 = 'AdaIn3' in config.GAN_options and not 'DRIT' in config.GAN_options 
    if self.AdaIn2: AdaIn_res=True
    else: AdaIn_res = False
    if self.InterLabels or self.InterStyleConcatLabels: in_dim=self.color_dim
    elif self.DRITZ: in_dim=self.color_dim+self.c_dim+self.style_dim
    else: in_dim=self.color_dim+self.c_dim
    layers.append(nn.Conv2d(in_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
    layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
    layers.append(nn.ReLU(inplace=True))     

    # Down-Sampling
    conv_repeat = int(math.log(self.image_size, 2))-5 if self.image_size>64 else 2
    curr_dim = conv_dim
    for i in range(conv_repeat):
      layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
      layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
      layers.append(nn.ReLU(inplace=True))
      curr_dim = curr_dim * 2

    # Bottleneck
    for i in range(repeat_num):
      layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, AdaIn=AdaIn_res))
      if i==int(repeat_num/2)-1 and self.AdaIn and not self.AdaIn3: AdaIn_res=True
      if i==int(repeat_num/2)-1 and (self.InterLabels or self.DRIT or self.InterStyleLabels):
        if self.AdaIn and not self.AdaIn3: AdaIn_res=True
        self.content = nn.Sequential(*layers)
        layers = []
        curr_dim = curr_dim+self.c_dim    
        if self.InterLabels and self.DRIT and not self.InterStyleLabels: curr_dim += self.c_dim   

    # Up-Sampling
    for i in range(conv_repeat):
      layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
      if not self.AdaIn: layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True)) #undesirable to generate images in vastly different styles
      layers.append(nn.ReLU(inplace=True))
      curr_dim = curr_dim // 2

    if self.Attention:
      self.main = nn.Sequential(*layers)
      self.layers = layers

      layers0 = []
      if self.AdaIn3:
        layers0.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, AdaIn=True))       
      layers0.append(nn.Conv2d(curr_dim, self.color_dim, kernel_size=7, stride=1, padding=3, bias=False))
      layers0.append(nn.Tanh())
      self.layers0 = layers0
      self.img_reg = nn.Sequential(*layers0)
      
      layers1 = []   
      # if self.AdaIn3:
      #   layers1.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, AdaIn=True))         
      layers1.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
      layers1.append(nn.Sigmoid())
      self.layers1 = layers1
      self.attetion_reg = nn.Sequential(*layers1)

    elif self.AdaIn:
      self.main = nn.Sequential(*layers)
      self.layers = layers

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
      print('-- Generator:')
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
      
      features = print_debug(feed, self.layers)
      if self.Attention: 
        _ = print_debug(features, self.layers0)
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
      x_cat = torch.cat([x, c], dim=1)

    if self.InterStyleLabels or self.InterLabels:
      content = self.content(x_cat)
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
        # ipdb.set_trace()
        if stochastic.size(-1)!=content.size(-1):
          stochastic = stochastic.repeat(1,1,content.size(-1)//stochastic.size(-1))
        stochastic = stochastic.unsqueeze(3)
        stochastic = stochastic.expand(stochastic.size(0), stochastic.size(1), stochastic.size(2), stochastic.size(2))
        x_cat = torch.cat([x_cat, stochastic], dim=1)    

    elif self.DRIT:
      content = self.content(x_cat)
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
        stochastic = stochastic.unsqueeze(3)
        stochastic = stochastic.expand(stochastic.size(0), stochastic.size(1), stochastic.size(2), stochastic.size(2))
        x_cat = torch.cat([content, stochastic], dim=1)                

    if JUST_CONTENT:
      return content

    if self.Attention: 
      features = self.main(x_cat)
      fake_img = self.img_reg(features)
      mask_img = self.attetion_reg(features)
      fake_img = mask_img * x + (1 - mask_img) * fake_img
      output = [fake_img, mask_img]

    elif self.AdaIn:  
      features = self.main(x_cat)
      output = [self.img_reg(features)]

    else: 
      output = [self.main(x_cat)]

    if CONTENT:
      output += [content]

    return output

#===============================================================================================#
#===============================================================================================#
class DRITZGEN(nn.Module):
  def __init__(self, config, debug=False):
    super(DRITZGEN, self).__init__()

    self.image_size = config.image_size    
    self.color_dim = config.color_dim
    self.style_dim = config.style_dim
    self.c_dim = config.c_dim

    self.enc_style = StyleEncoder(config, debug=True)
    self.generator = Generator(config, debug=False)
    self.debug()

  def debug(self):
    # feed = to_var(torch.ones(1, self.color_dim+self.c_dim+self.style_dim, self.image_size, self.image_size), volatile=True, no_cuda=True)
    # style = to_var(self.random_style(feed), volatile=True)
    self.generator.debug()
    
  def forward(self, x, c, stochastic=None, CONTENT=False, JUST_CONTENT=False):
    if stochastic is None:
      style = self.get_style(x)
    else:
      style = stochastic
    return self.generator(x, c, stochastic=style, CONTENT=CONTENT, JUST_CONTENT=JUST_CONTENT)

  def random_style(self,x):
    z = torch.randn(x.size(0), self.style_dim)
    return z

  def get_style(self, x, volatile=False):
    style = self.enc_style(x)
    style = style[0].view(style[0].size(0), -1)
    return [style]

#===============================================================================================#
#===============================================================================================#
class DRITGEN(nn.Module):
  def __init__(self, config, debug=False):
    super(DRITGEN, self).__init__()

    self.image_size = config.image_size    
    self.color_dim = config.color_dim
    self.style_dim = config.style_dim
    self.mono_style = 'mono_style' in config.GAN_options
    self.c_dim = config.c_dim if not self.mono_style else 1

    self.enc_style = StyleEncoder(config, debug=True)
    self.generator = Generator(config, debug=False)
    self.debug()

  def debug(self):
    feed = to_var(torch.ones(1, self.color_dim, self.image_size, self.image_size), volatile=True, no_cuda=True)
    style = self.get_style(feed, volatile=True)
    self.generator.debug()
    
  def forward(self, x, c, stochastic=None, CONTENT=False, JUST_CONTENT=False):
    if stochastic is None:
      style = self.get_style(x)
    else:
      style = stochastic
    return self.generator(x, c, stochastic=style, CONTENT=CONTENT, JUST_CONTENT=JUST_CONTENT)

  def random_style(self,x):
    if not self.mono_style: 
      z = torch.randn(x.size(0), self.c_dim, self.style_dim)
    else:
      z = torch.randn(x.size(0), self.style_dim)
    return z

  def get_style(self, x, volatile=False):
    style = self.enc_style(x)
    if len(style)==1:
      style = [style[0].view(style[0].size(0), self.c_dim, -1)]
    else:
      style[0] = style[0].view(style[0].size(0), self.c_dim, -1)
      style[1] = style[1].view(style[1].size(0), self.c_dim, -1)      
      std = style[1].data.mul(0.5).exp_()
      eps = self.random_style(style[0].data)
      if style[1].data.is_cuda: eps = to_cuda(eps)
      style = [to_var(eps.mul(std).add_(style[0].data), volatile=volatile)] + style
    return style

#===============================================================================================#
#===============================================================================================#
class AdaInGEN_Label(nn.Module):
  def __init__(self, config, debug=False):
    super(AdaInGEN_Label, self).__init__()

    conv_dim = config.g_conv_dim
    mlp_dim = config.mlp_dim
    self.color_dim = config.color_dim
    self.image_size = config.image_size
    self.c_dim = config.c_dim
    self.Stochastic = 'Stochastic' in config.GAN_options
    self.InterStyleConcatLabels= 'InterStyleConcatLabels' in config.GAN_options
    self.generator = Generator(config, debug=False)
    self.adain_net = MLP(self.c_dim, self.get_num_adain_params(self.generator), mlp_dim, 3, norm='none', activ='relu', debug=True)
    self.debug()

  def debug(self):
    feed  = to_var(torch.ones(1,self.color_dim,self.image_size,self.image_size), volatile=True, no_cuda=True)
    label = to_var(torch.ones(1,self.c_dim), volatile=True, no_cuda=True)
    self.apply_style(feed, label)
    self.generator.debug()
    
  def forward(self, x, c, stochastic=None, CONTENT=False, JUST_CONTENT=False):
    self.apply_style(x, c)
    return self.generator(x, c, stochastic=None, CONTENT=CONTENT, JUST_CONTENT=JUST_CONTENT)

  def apply_style(self, image, label):
    adain_params = self.adain_net(label)
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
class AdaInGEN(nn.Module):
  def __init__(self, config, debug=False):
    super(AdaInGEN, self).__init__()

    conv_dim = config.g_conv_dim
    mlp_dim = config.mlp_dim
    self.color_dim = config.color_dim
    self.image_size = config.image_size
    self.style_dim = config.style_dim    
    self.c_dim = config.c_dim if config.style_dim!=8 else 1
    self.StyleDisc = 'StyleDisc' in config.GAN_options
    self.InterStyleConcatLabels= 'InterStyleConcatLabels' in config.GAN_options
    if not self.StyleDisc: self.enc_style = StyleEncoder(config, debug=True)
    self.generator = Generator(config, debug=False)
    in_dim = self.style_dim*self.c_dim
    if self.InterStyleConcatLabels: in_dim *=2
    self.adain_net = MLP(in_dim, self.get_num_adain_params(self.generator), mlp_dim, 3, norm='none', activ='relu', debug=True)
    self.debug()

  def debug(self):
    feed = to_var(torch.ones(1,self.color_dim,self.image_size,self.image_size), volatile=True, no_cuda=True)
    label = to_var(torch.ones(1,self.c_dim), volatile=True, no_cuda=True)
    style = to_var(self.random_style(feed), volatile=True, no_cuda=True) #self.get_style(feed)
    self.apply_style(feed, style, label=label)
    self.generator.debug()
    
  def forward(self, x, c, stochastic=None, CONTENT=False, JUST_CONTENT=False):
    if stochastic is None:
      style = self.get_style(x)
      style = style[0]
    else:
      style = stochastic
    self.apply_style(x, style, label=c)
    return self.generator(x, c, stochastic=style, CONTENT=CONTENT, JUST_CONTENT=JUST_CONTENT)

  def random_style(self, x):
    if self.style_dim!=8: 
      z = torch.randn(x.size(0), self.c_dim, self.style_dim)
    else:
      z = torch.randn(x.size(0), self.style_dim)
    return z

  def get_style(self, x, volatile=False):
    style = self.enc_style(x)
    style = [style[0].view(style[0].size(0), self.c_dim, -1)]
    return style

  def apply_style(self, image, style, label=None):
    # apply style code to an image
    if self.InterStyleConcatLabels:
      # ipdb.set_trace()
      label = label.unsqueeze(2).expand_as(style)
      style = torch.cat([style, label], dim=2)
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
    color_dim = config.color_dim
    style_dim = config.style_dim
    self.DRITZ = 'DRITZ' in config.GAN_options    
    self.c_dim = config.c_dim
    self.s_dim = 1 if style_dim==8 or self.DRITZ else config.c_dim*style_dim
    self.FC = 'FC' in config.GAN_options
    layers = []
    norm = 'none'
    activ = 'relu'
    pad_type = 'reflect'
    layers.append(Conv2dBlock(color_dim, conv_dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type))
    # layers.append(nn.Conv2d(color_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
    # layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
    # layers.append(nn.ReLU(inplace=True))

    # Down-Sampling
    if style_dim==4:
      down = [2,1]
    else:
      down =[3,3]
    conv_repeat = int(math.log(image_size, 2))-down[0] #1 until 2x2, 2 for 4x4
    curr_dim = conv_dim
    for i in range(conv_repeat):
      layers.append(Conv2dBlock(curr_dim, curr_dim*2, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type))
      # layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
      # layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
      # layers.append(nn.ReLU(inplace=True))

      # Bottleneck
      # for i in range(2):
      #   layers.append(ResidualBlock(dim_in=curr_dim*2, dim_out=curr_dim*2))          

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
      out_dim = self.s_dim*style_dim
      self.style_mu = nn.Linear(256, out_dim)
    else:
      self.style_mu = nn.Conv2d(curr_dim, self.c_dim, kernel_size=1, stride=1, padding=0)
    self.main = nn.Sequential(*layers)

    if debug:
      feed = to_var(torch.ones(1,color_dim,image_size,image_size), volatile=True, no_cuda=True)
      print('-- StyleEncoder:')
      features = print_debug(feed, layers)
      if self.FC:
        fc_in = features.view(features.size(0), -1)
        features = print_debug(fc_in, layers0)
      _ = print_debug(features, [self.style_mu]) 

  def forward(self, x):
    features = self.main(x)
    if self.FC:
      fc_input = features.view(features.size(0), -1)
      features = self.fc(fc_input)

    style_mu = self.style_mu(features)
    style = [style_mu]
    return style
    
#===============================================================================================#
#===============================================================================================#
class MLP(nn.Module):
  def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu', debug=False):

    super(MLP, self).__init__()
    self._model = []
    self._model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
    for i in range(n_blk - 2):
      self._model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
    self._model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
    self.model = nn.Sequential(*self._model)

    if debug:
      feed = to_var(torch.ones(1,input_dim), volatile=True, no_cuda=True)
      print('-- MLP:')
      _ = print_debug(feed, self._model)    

  def forward(self, x):
    return self.model(x.view(x.size(0), -1))  

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
  def __init__(self, num_features, eps=1e-5, momentum=0.1):
    super(AdaptiveInstanceNorm2d, self).__init__()
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    # weight and bias are dynamically assigned
    self.weight = None
    self.bias = None
    # just dummy buffers, not used
    self.register_buffer('running_mean', torch.zeros(num_features))
    self.register_buffer('running_var', torch.ones(num_features))

  def forward(self, x):
    assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
    b, c = x.size(0), x.size(1)
    running_mean = self.running_mean.repeat(b)
    running_var = self.running_var.repeat(b)

    # Apply instance norm
    x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

    out = F.batch_norm(
      x_reshaped, running_mean, running_var, self.weight, self.bias,
      True, self.momentum, self.eps)
    return out.view(b, c, *x.size()[2:])

  def __repr__(self):
    return self.__class__.__name__ + '(' + str(self.num_features) + ')'   

#===============================================================================================#
#===============================================================================================#
class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out    

#===============================================================================================#
#===============================================================================================#
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
