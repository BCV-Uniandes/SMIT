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
                                    or isinstance(layer, SpectralNormalization):
      print(str(layer).split('(')[0], feed.size())
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
    layers_debug.append(self.conv1)
    if debug:
      feed = Variable(torch.ones(1,3,image_size,image_size), volatile=True)
      print('-- Discriminator:')
      _ = print_debug(feed, layers_debug)


  def forward(self, x):
    h = self.main(x)
    # ipdb.set_trace()
    out_real = self.conv1(h).squeeze()
    out_aux = self.conv2(h).squeeze()

    return out_real.view(x.size(0), out_real.size(-2), out_real.size(-1)), out_aux.view(x.size(0), out_aux.size(-1))


class Generator(nn.Module):
  """Generator. Encoder-Decoder Architecture."""
  def __init__(self, image_size = 128, conv_dim=64, c_dim=5, repeat_num=6, Attention=False, AdaIn=False, debug=False):
    super(Generator, self).__init__()
    layers = []
    self.Attention = Attention
    self.AdaIn = AdaIn
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
        layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
      layers.append(nn.ReLU(inplace=True))
      curr_dim = curr_dim // 2
      if SAGAN and i>0:
        layers.append(Self_Attn(64*(i+1), curr_dim))  

    self.main = nn.Sequential(*layers)

    layers0 = []
    layers0.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
    layers0.append(nn.Tanh())
    self.img_reg = nn.Sequential(*layers0)

    if self.Attention:
      layers1 = []
      layers1.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
      layers1.append(nn.Sigmoid())
      self.attetion_reg = nn.Sequential(*layers1)

    if debug and not AdaIn:
      feed = Variable(torch.ones(1,3+c_dim,image_size,image_size), volatile=True)
      print('-- Generator:')
      features = print_debug(feed, layers)
      _ = print_debug(features, layers0)
      if self.Attention: _ = print_debug(features, layers1)

  def forward(self, x, c, style=None):
    # replicate spatially and concatenate domain information
    c = c.unsqueeze(2).unsqueeze(3)
    c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
    # ipdb.set_trace()
    if style is not None:
      style.
    x = torch.cat([x, c], dim=1)
    features = self.main(x)
    if self.Attention: return self.img_reg(features), self.attetion_reg(features)
    else: return self.img_reg(features)

class StyleEncoder(nn.Module):
  """Generator. Encoder-Decoder Architecture."""
  def __init__(self, image_size=128, mlp_dim=256, style_dim=8, c_dim=12, conv_dim=64, debug=False):
    super(StyleEncoder, self).__init__()
    self.c_dim = c_dim
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

    layers.append(nn.Conv2d(curr_dim, c_dim, kernel_size=7, stride=1, padding=3, bias=False))
    self.main = nn.Sequential(*layers)

    if debug:
      feed = Variable(torch.ones(1,3,image_size,image_size), volatile=True)
      print('-- StyleEncoder:')
      _ = print_debug(features, layers)

  def forward(self, x):
    x = self.main(x)
    x = x.view(x.size(0), self.c_dim, -1)
    return x

class AdaInGEN(nn.Module):
  def __init__(self, image_size = 128, conv_dim=64, c_dim=12, repeat_num=6, mlp_dim=256, style_dim=8, Attention=False, debug=False):
    super(AdaInGEN, self).__init__()

    self.generator = Generator(image_size, g_conv_dim, c_dim, g_repeat_num, Attention, debug=False)

    # style encoder
    self.enc_style = StyleEncoder(image_size, mlp_dim, style_dim, c_dim, conv_dim)


    self.adain_net = MLP(style_dim, self.get_num_adain_params(self.generator), mlp_dim, 3, norm='none', activ=activ)


  def forward(self, x):
    style = self.get_style(x)
    self.apply_style(style)
    return self.generator(x)

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
  def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

    super(MLP, self).__init__()
    self.model = []
    self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
    for i in range(n_blk - 2):
      self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
    self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
    self.model = nn.Sequential(*self.model)

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
    ipdb.set_trace()
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