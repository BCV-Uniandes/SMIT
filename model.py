import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable
import numpy as np
from models.spectral import SpectralNorm as SpectralNormalization
import ipdb
import math

if int(torch.__version__.split('.')[1])>3:
  def to_var(x, volatile=False):
    return to_cuda(x)
  def to_cuda(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(x, nn.Module):
      x.to(device)
    else:
      return x.to(device)
else:
  from torch.autograd import Variable
  def to_var(x, volatile=False):
    return Variable(x, volatile=volatile)
  def to_cuda(x):
    if torch.cuda.is_available():
      if isinstance(x, nn.Module):
        x.cuda()
      else:
        return x.cuda()
    else:
      return x

def get_SN(bool):
  if bool:
    return SpectralNormalization
  else:
    return lambda x:x

def print_debug(feed, layers):
  print(feed.size())
  for layer in layers:
    feed = layer(feed)
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d) \
                                    or isinstance(layer, nn.Linear) \
                                    or isinstance(layer, ResidualBlock) \
                                    or isinstance(layer, LinearBlock) \
                                    or isinstance(layer, SpectralNormalization):
      print(str(layer).split('(')[0], feed.size())
  print(' ')
  return feed

class ResidualBlock(nn.Module):
  """Residual Block."""
  def __init__(self, dim_in, dim_out):
    super(ResidualBlock, self).__init__()
    self.main = nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
      nn.InstanceNorm2d(dim_out, affine=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
      nn.InstanceNorm2d(dim_out, affine=True))

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
    c_dim = config.c_dim
    color_dim = config.color_dim
    SN = 'SpectralNorm' in config.GAN_options
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
    self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)
    if debug:
      feed = to_var(torch.ones(1,color_dim,image_size,image_size), volatile=True)
      print('-- Discriminator:')
      features = print_debug(feed, layers_debug)
      _ = print_debug(features, [self.conv1])
      _ = print_debug(features, [self.conv2])      

  def forward(self, x):
    h = self.main(x)
    out_real = self.conv1(h).squeeze()
    out_real = out_real.view(x.size(0), out_real.size(-2), out_real.size(-1))

    out_aux = self.conv2(h).squeeze()
    out_aux = out_aux.view(x.size(0), out_aux.size(-1))

    return out_real, out_aux


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
    self.Attention = 'Attention' in config.GAN_options
    self.InterLabels = 'InterLabels' in config.GAN_options
    self.InterStyleLabels = 'InterStyleLabels' in config.GAN_options
    self.style_gen = 'style_gen' in config.GAN_options
    self.DRIT = 'DRIT' in config.GAN_options and not self.InterStyleLabels
    self.AdaIn = 'AdaIn' in config.GAN_options and not 'DRIT' in config.GAN_options 
    if self.InterLabels: layers.append(nn.Conv2d(self.color_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
    else: layers.append(nn.Conv2d(self.color_dim+self.c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
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
      layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
      if i==int(repeat_num/2)-1 and (self.InterLabels or self.DRIT or self.InterStyleLabels):
        self.content = nn.Sequential(*layers)
        layers = []
        curr_dim = curr_dim+self.c_dim    
        if self.InterLabels and self.DRIT and not self.InterStyleLabels: curr_dim += self.c_dim   

    # Up-Sampling
    for i in range(conv_repeat):
      layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
      if self.AdaIn:
        layers.append(AdaptiveInstanceNorm2d(curr_dim//2))
      else:
        layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
      layers.append(nn.ReLU(inplace=True))
      curr_dim = curr_dim // 2

    # self.main = nn.Sequential(*layers)
    # self.layers = layers

    # layers0 = []
    # layers0.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
    # layers0.append(nn.Tanh())
    # self.layers0 = layers0
    # self.img_reg = nn.Sequential(*layers0)

    if self.Attention:
      ##ADDED
      self.main = nn.Sequential(*layers)
      self.layers = layers

      layers0 = []
      layers0.append(nn.Conv2d(curr_dim, self.color_dim, kernel_size=7, stride=1, padding=3, bias=False))
      layers0.append(nn.Tanh())
      self.layers0 = layers0
      self.img_reg = nn.Sequential(*layers0)
      #######

      layers1 = []
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
      elif self.InterLabels:
        c_dim = self.c_dim
        in_dim = self.color_dim
      elif self.DRIT:
        c_dim = self.c_dim
        in_dim = self.color_dim+self.c_dim
      else:
        in_dim = self.color_dim+self.c_dim
        c_dim = 0

      if self.InterLabels or self.DRIT: 
        feed = to_var(torch.ones(1,in_dim, self.image_size,self.image_size), volatile=True)
        feed = print_debug(feed, self.content)
        c = to_var(torch.ones(1, c_dim, feed.size(2), feed.size(3)), volatile=True)
        feed = torch.cat([feed, c], dim=1)
      else: 
        feed = to_var(torch.ones(1,in_dim,self.image_size,self.image_size), volatile=True)
      
      features = print_debug(feed, self.layers)
      if self.Attention: 
        _ = print_debug(features, self.layers0)
        _ = print_debug(features, self.layers1)
      elif self.AdaIn: 
        _ = print_debug(features, self.layers0)        

  def forward(self, x, c, stochastic=None, CONTENT=False):
    # replicate spatially and concatenate domain information
    if not self.InterLabels:
      c = c.unsqueeze(2).unsqueeze(3)
      c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
      x_cat = torch.cat([x, c], dim=1)
    else:
      x_cat = x

    if self.InterStyleLabels:
      content = self.content(x_cat)
      c = stochastic*c.unsqueeze(2)
      c = c.unsqueeze(3)
      c = c.expand(c.size(0), c.size(1), content.size(2), content.size(3))
      x_cat = torch.cat([content, c], dim=1)

    elif self.DRIT:
      content = self.content(x_cat)
      if self.InterLabels:
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), content.size(2), content.size(3))        
        stochastic = stochastic.unsqueeze(3)
        stochastic = stochastic.expand(stochastic.size(0), stochastic.size(1), stochastic.size(2), stochastic.size(2))
        x_cat = torch.cat([content, c, stochastic], dim=1)        
      else:
        content = self.content(x_cat)
        stochastic = stochastic.unsqueeze(3)
        stochastic = stochastic.expand(stochastic.size(0), stochastic.size(1), stochastic.size(2), stochastic.size(2))
        x_cat = torch.cat([content, stochastic], dim=1)                

    if self.Attention: 
      features = self.main(x_cat)
      fake_img = self.img_reg(features)
      mask_img = self.attetion_reg(features)
      fake_img = (mask_img * x) + ((1 - mask_img) * fake_img)
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
    feed = to_var(torch.ones(1, self.color_dim, self.image_size, self.image_size), volatile=True)
    style = self.get_style(feed, volatile=True)
    self.generator.debug()
    
  def forward(self, x, c, stochastic=None, CONTENT=False):
    if stochastic is None:
      style = self.get_style(x)
    else:
      style = stochastic
    return self.generator(x, c, stochastic=style, CONTENT=CONTENT)

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
class AdaInGEN(nn.Module):
  def __init__(self, config, debug=False):
    super(AdaInGEN, self).__init__()

    conv_dim = config.g_conv_dim
    mlp_dim = config.mlp_dim
    self.color_dim = config.color_dim
    self.image_size = config.image_size
    self.style_dim = config.style_dim    
    self.mono_style = 'mono_style' in config.GAN_options
    self.c_dim = config.c_dim if not self.mono_style else 1
    
    self.enc_style = StyleEncoder(config, debug=True)
    self.generator = Generator(config, debug=False)
    self.adain_net = MLP(self.style_dim*self.c_dim, self.get_num_adain_params(self.generator), mlp_dim, 3, norm='none', activ='relu', debug=True)
    self.debug()

  def debug(self):
    feed = to_var(torch.ones(1,self.color_dim,self.image_size,self.image_size), volatile=True)
    style = self.get_style(feed)
    self.apply_style(feed, style[0])
    self.generator.debug()
    
  def forward(self, x, c, stochastic=None, CONTENT=False):
    if stochastic is None:
      style = self.get_style(x)
      style = style[0]
    else:
      style = stochastic
    self.apply_style(x, style)
    return self.generator(x, c, stochastic=style, CONTENT=CONTENT)

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

  def apply_style(self, image, style):
    # apply style code to an image
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
    self.c_dim = config.c_dim
    self.mono_style = 'mono_style' in config.GAN_options
    self.FC = 'FC' in config.GAN_options
    self.lognet = 'LOGVAR' in config.GAN_options
    layers = []
    layers.append(nn.Conv2d(color_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
    layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
    layers.append(nn.ReLU(inplace=True))

    # Down-Sampling
    conv_repeat = int(math.log(image_size, 2))-2
    curr_dim = conv_dim
    for i in range(conv_repeat):
      layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
      layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
      layers.append(nn.ReLU(inplace=True))

      # Bottleneck
      # for i in range(2):
      #   layers.append(ResidualBlock(dim_in=curr_dim*2, dim_out=curr_dim*2))          

      curr_dim = curr_dim * 2    

    if self.mono_style: 
      layers.append(nn.AdaptiveAvgPool2d(1)) # global average pooling
      self.style_mu = nn.Conv2d(curr_dim, style_dim, kernel_size=1, stride=1, padding=0, bias=False)
      if self.lognet:
        self.style_var = nn.Conv2d(curr_dim, style_dim, kernel_size=1, stride=1, padding=0, bias=False)
    elif self.FC:
      k_size = int(image_size / np.power(2, conv_repeat))
      layers0=[]
      layers0.append(nn.Linear(curr_dim*k_size*k_size, 256, bias=True))  
      layers0.append(nn.Dropout(0.5))  
      layers0.append(nn.Linear(256, 256, bias=True))  
      layers0.append(nn.Dropout(0.5))
      self.fc = nn.Sequential(*layers0)
      self.style_mu = nn.Linear(256, self.c_dim*k_size*k_size, bias=False)
      if self.lognet:
        self.style_var = nn.Linear(256, self.c_dim*k_size*k_size, bias=False)
    else:
      self.style_mu = nn.Conv2d(curr_dim, self.c_dim, kernel_size=1, stride=1, padding=0, bias=False)
      if self.lognet:
        self.style_var = nn.Conv2d(curr_dim, self.c_dim, kernel_size=1, stride=1, padding=0, bias=False)
    self.main = nn.Sequential(*layers)

    if debug:
      feed = to_var(torch.ones(1,color_dim,image_size,image_size), volatile=True)
      print('-- StyleEncoder:')
      features = print_debug(feed, layers)
      if self.FC:
        fc_in = features.view(features.size(0), -1)
        features = print_debug(fc_in, layers0)
      _ = print_debug(features, [self.style_mu]) 
      if self.lognet:
        _ = print_debug(features, [self.style_var]) 

  def forward(self, x):
    features = self.main(x)
    if self.FC:
      fc_input = features.view(features.size(0), -1)
      features = self.fc(fc_input)

    style_mu = self.style_mu(features)
    if self.lognet:
      style_var = self.style_var(features)
      style = [style_mu, style_var]
    else:
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
      feed = to_var(torch.ones(1,input_dim), volatile=True)
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
