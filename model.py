import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from models.spectral import SpectralNorm as SpectralNormalization
from models.sagan import Self_Attn
import ipdb
import math

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
                                    or isinstance(layer, ResidualBlock) \
                                    or isinstance(layer, Self_Attn) \
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


class Discriminator(nn.Module):
  """Discriminator. PatchGAN."""
  def __init__(self, image_size=256, conv_dim=64, c_dim=5, repeat_num=6, SN=False, SAGAN=False, debug=False):
    super(Discriminator, self).__init__()
    SpectralNorm = get_SN(SN)
    layers = []
    layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
    layers.append(nn.LeakyReLU(0.01, inplace=True))

    curr_dim = conv_dim
    for i in range(1, repeat_num):
      layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
      layers.append(nn.LeakyReLU(0.01, inplace=True))
      curr_dim = curr_dim * 2
      # if SAGAN and i<repeat_num-3:
      #   layers.append(Self_Attn(64*(i+1), curr_dim))        

    k_size = int(image_size / np.power(2, repeat_num))
    layers_debug = layers
    self.main = nn.Sequential(*layers)
    # ipdb.set_trace()
    self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
    self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)
    if debug:
      feed = Variable(torch.ones(1,3,image_size,image_size), volatile=True)
      print('-- Discriminator:')
      features = print_debug(feed, layers_debug)
      _ = print_debug(features, [self.conv1])
      _ = print_debug(features, [self.conv2])      

  def forward(self, x):
    h = self.main(x)
    # ipdb.set_trace()
    out_real = self.conv1(h).squeeze()
    out_aux = self.conv2(h).squeeze()

    return out_real.view(x.size(0), out_real.size(-2), out_real.size(-1)), out_aux.view(x.size(0), out_aux.size(-1))


class Generator(nn.Module):
  """Generator. Encoder-Decoder Architecture."""
  def __init__(self, image_size = 128, conv_dim=64, c_dim=5, repeat_num=6, Attention=False, AdaIn=False, SAGAN=False, style_label_net=False, vae_like=False, debug=False):
    super(Generator, self).__init__()
    layers = []
    self.Attention = Attention
    self.AdaIn = AdaIn
    self.image_size = image_size
    self.c_dim = c_dim
    layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
    layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
    layers.append(nn.ReLU(inplace=True))

    # if SAGAN:
    #   attn1 = Self_Attn(int(self.imsize/4), 128, 'relu')
    #   attn2 = Self_Attn(int(self.imsize/2), 64, 'relu')      

    # Down-Sampling
    conv_repeat = int(math.log(image_size, 2))-5
    curr_dim = conv_dim
    for i in range(conv_repeat):
      layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
      layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
      layers.append(nn.ReLU(inplace=True))
      curr_dim = curr_dim * 2

    # Bottleneck
    for i in range(repeat_num):
      layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

    # Up-Sampling
    if SAGAN: self.scores = []
    for i in range(conv_repeat):
      layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
      if AdaIn:
        layers.append(AdaptiveInstanceNorm2d(curr_dim//2))
      else:
        layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
      layers.append(nn.ReLU(inplace=True))
      curr_dim = curr_dim // 2
      if SAGAN and i>0:
        layers.append(Self_Attn(64*(i+1), curr_dim))  

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
      layers0.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
      layers0.append(nn.Tanh())
      self.layers0 = layers0
      self.img_reg = nn.Sequential(*layers0)
      #######

      layers1 = []
      layers1.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
      layers1.append(nn.Sigmoid())
      self.layers1 = layers1
      self.attetion_reg = nn.Sequential(*layers1)

    elif self.AdaIn is not None:
      self.main = nn.Sequential(*layers)
      self.layers = layers

      layers0 = []
      layers0.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
      layers0.append(nn.Tanh())
      self.layers0 = layers0
      self.img_reg = nn.Sequential(*layers0)      

    else:
      layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
      layers.append(nn.Tanh())
      self.layers = layers
      self.main = nn.Sequential(*layers)

    if debug and not AdaIn:
      feed = Variable(torch.ones(1,3+c_dim,image_size,image_size), volatile=True)
      print('-- Generator:')
      features = print_debug(feed, layers)
      if self.Attention: 
        _ = print_debug(features, layers0)
        _ = print_debug(features, layers1)

      elif self.AdaIn is not None: 
        _ = print_debug(features, layers0)

  def debug(self):
      feed = Variable(torch.ones(1,3+self.c_dim,self.image_size,self.image_size), volatile=True)
      print('-- Generator:')
      features = print_debug(feed, self.layers)
      if self.Attention: 
        _ = print_debug(features, self.layers0)
        _ = print_debug(features, self.layers1)
      elif self.AdaIn is not None: 
        _ = print_debug(features, self.layers0)        

  def forward(self, x, c, stochastic=None):
    # replicate spatially and concatenate domain information
    c = c.unsqueeze(2).unsqueeze(3)
    c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
    # ipdb.set_trace()
    x_cat = torch.cat([x, c], dim=1)
    if self.Attention: 
      features = self.main(x_cat)
      fake_img = self.img_reg(features)
      mask_img = self.attetion_reg(features)
      fake_img = mask_img * x + (1 - mask_img) * fake_img            
      return fake_img, mask_img

    elif self.AdaIn is not None:  
      features = self.main(x_cat)
      return self.img_reg(features)         

    else: 
      return self.main(x)

class StyleEncoder(nn.Module):
  """Generator. Encoder-Decoder Architecture."""
  def __init__(self, image_size=128, mlp_dim=256, style_dim=8, c_dim=12, conv_dim=64, style_label_net=False, debug=False):
    super(StyleEncoder, self).__init__()
    self.c_dim = c_dim
    self.style_label_net = style_label_net
    layers = []
    layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
    layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
    layers.append(nn.ReLU(inplace=True))

    # if SAGAN:
    #   attn1 = Self_Attn(int(self.imsize/4), 128, 'relu')
    #   attn2 = Self_Attn(int(self.imsize/2), 64, 'relu')      

    # Down-Sampling
    conv_repeat = int(math.log(image_size, 2))-2
    curr_dim = conv_dim
    for i in range(conv_repeat):
      layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
      layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
      layers.append(nn.ReLU(inplace=True))
      curr_dim = curr_dim * 2

    self.main = nn.Sequential(*layers)
    self.style = nn.Conv2d(curr_dim, c_dim, kernel_size=7, stride=1, padding=3, bias=False)
    if self.style_label_net: 
      k_size = int(image_size / np.power(2, conv_repeat))
      self.cls = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    if debug:
      feed = Variable(torch.ones(1,3,image_size,image_size), volatile=True)
      print('-- StyleEncoder:')
      features = print_debug(feed, layers)
      _ = print_debug(features, [self.style]) 
      if self.style_label_net: 
        _ = print_debug(features, [self.cls]) 

  def forward(self, x):
    features = self.main(x)
    style = self.style(features).view(features.size(0), self.c_dim, -1)
    cls = self.cls(features).view(features.size(0), self.c_dim) if self.style_label_net else None
    return style, cls
    # style = style*cls.unsqueeze(2)

class StyleDecoder(nn.Module):
  def __init__(self, image_size=128, c_dim=12, conv_dim=64, debug=False):
    super(StyleDecoder, self).__init__()

    conv_repeat = int(math.log(image_size, 2))-2
    layers = []
    conv_dim = conv_dim*int(math.pow(2,conv_repeat))
    layers.append(nn.ConvTranspose2d(c_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=False))
    layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
    layers.append(nn.ReLU(inplace=True))

    for i in range(conv_repeat):
      layers.append(nn.ConvTranspose2d(conv_dim, conv_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
      layers.append(nn.InstanceNorm2d(conv_dim//2, affine=True))
      layers.append(nn.ReLU(inplace=True))
      conv_dim = conv_dim // 2
    
    layers.append(nn.Conv2d(conv_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
    layers.append(nn.Tanh())
    self.main = nn.Sequential(*layers)
    self.style_size = int(image_size//math.pow(2,int(math.log(image_size, 2))-2))
    self.c_dim = c_dim

    if debug:
      feed = Variable(torch.ones(1, self.c_dim, self.style_size, self.style_size), volatile=True)
      print('-- StyleDecoder:')
      _ = print_debug(feed, layers)

  def forward(self, x):
    return self.main(x)

  def random_noise(self, batchSize):
    z = torch.cuda.FloatTensor(batchSize, self.c_dim, self.style_size, self.style_size)
    z.copy_(torch.randn(batchSize, self.c_dim, self.style_size, self.style_size))
    return z    


class AdaInGEN(nn.Module):
  def __init__(self, image_size = 128, conv_dim=64, c_dim=12, repeat_num=6, mlp_dim=256, style_dim=8, Attention=False, style_label_net=False, vae_like=False, debug=False):
    super(AdaInGEN, self).__init__()

    self.image_size = image_size
    self.style_dim = style_dim
    self.c_dim = c_dim
    self.vae_like = vae_like
    # style encoder
    self.enc_style = StyleEncoder(image_size, mlp_dim, style_dim, c_dim, conv_dim, style_label_net=style_label_net, debug=True)
    if vae_like: self.dec_style = StyleDecoder(image_size, c_dim, conv_dim, debug=True)

    self.generator = Generator(image_size, conv_dim, c_dim, repeat_num, Attention, AdaIn=True, debug=False)

    self.adain_net = MLP(style_dim*c_dim, self.get_num_adain_params(self.generator), mlp_dim, 3, norm='none', activ='relu', debug=True)
    self.debug()

  def debug(self):
    feed = Variable(torch.ones(1,3,self.image_size,self.image_size), volatile=True)
    # print('-- Generator:')    
    style, _ = self.get_style(feed)
    # if self.vae_like:
      # _ = self.dec_style(style)
    self.apply_style(feed, style)
    self.generator.debug()
    
  def forward(self, x, c, stochastic=None):
    if stochastic is None:
      style, _ = self.get_style(x)
    else:
      style = stochastic
    self.apply_style(x, style)
    return self.generator(x, c)

  def random_style(self,x):
    return torch.randn(x.size(0), self.c_dim, self.style_dim)

  def get_style(self, x):
    style = self.enc_style(x)
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
        # ipdb.set_trace()
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

class MLP(nn.Module):
  def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu', debug=False):

    super(MLP, self).__init__()
    self._model = []
    # ipdb.set_trace()
    self._model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
    for i in range(n_blk - 2):
      self._model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
    self._model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
    self.model = nn.Sequential(*self._model)


    if debug:
      feed = Variable(torch.ones(1,input_dim), volatile=True)
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
    # ipdb.set_trace()
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