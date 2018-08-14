# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import sys
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
from model import Generator
from model import Discriminator
from PIL import Image
import ipdb
import config as cfg
import glob
import pickle
from tqdm import tqdm
from misc.utils import f1_score, f1_score_max, F1_TEST
import imageio
import skimage.transform
import math
from scipy.ndimage import filters
import warnings
import pytz

warnings.filterwarnings('ignore')

class Solver(object):

  def __init__(self, data_loader, config):
    # Data loader
    self.data_loader = data_loader[0]
    self.config = config

    self.blurrandom = 0

    # Build tensorboard if use
    self.build_model()
    if self.config.use_tensorboard:
      self.build_tensorboard()

    # Start with trained model
    if self.config.pretrained_model:
      self.load_pretrained_model()


  #=======================================================================================#
  #=======================================================================================#
  def display_net(self, name='discriminator'):
    #pip install git+https://github.com/szagoruyko/pytorchviz
    from graphviz import Digraph
    from torchviz import make_dot, make_dot_from_trace
    
    if name=='Discriminator':
      y1,y2 = self.D(self.to_var(torch.ones(1,3,self.config.image_size,self.config.image_size)))
      g=make_dot(y1, params=dict(self.D.named_parameters()))
    elif name=='Generator':
      y, _ = self.G(self.to_var(torch.ones(1,3,self.config.image_size,self.config.image_size)), self.to_var(torch.zeros(1,12)))
      g=make_dot(y, params=dict(self.G.named_parameters()))
    g.filename=name
    g.render()
    os.remove(name)

    from utils import pdf2png
    pdf2png(name)
    self.PRINT('Network saved at {}.png'.format(name))

  #=======================================================================================#
  #=======================================================================================#
  def build_model(self):
    # Define a generator and a discriminator
    from model import Discriminator
    if 'AdaIn' in self.config.GAN_options:
      from model import AdaInGEN
      self.G = AdaInGEN(self.config.image_size, self.config.g_conv_dim, self.config.c_dim, \
                   self.config.g_repeat_num, mlp_dim=self.config.mlp_dim, \
                   style_dim=self.config.style_dim, Attention='Attention' in self.config.GAN_options, \
                   style_label_net= 'style_label_net' in self.config.GAN_options, 
                   vae_like= 'vae_like' in self.config.GAN_options, debug=True)
    else: 
      from model import Generator
      self.G = Generator(self.config.image_size, self.config.g_conv_dim, self.config.c_dim, \
                       self.config.g_repeat_num, Attention='Attention' in self.config.GAN_options, debug=True)
    if not 'GOOGLE' in self.config.GAN_options: 
      self.D = Discriminator(self.config.image_size, self.config.d_conv_dim, self.config.c_dim, 
                       self.config.d_repeat_num, SN='SpectralNorm' in self.config.GAN_options, SAGAN='SAGAN' in self.config.GAN_options,
                       debug=True) 

    G_parameters = filter(lambda p: p.requires_grad, self.G.parameters())
    self.g_optimizer = torch.optim.Adam(G_parameters, self.config.g_lr, [self.config.beta1, self.config.beta2])

    if not 'GOOGLE' in self.config.GAN_options: 
      D_parameters = filter(lambda p: p.requires_grad, self.D.parameters())
      self.d_optimizer = torch.optim.Adam(D_parameters, self.config.d_lr, [self.config.beta1, self.config.beta2])

    if torch.cuda.is_available():
      self.G.cuda()
      if not 'GOOGLE' in self.config.GAN_options: self.D.cuda()

    # self.PRINT networks
    self.print_network(self.G, 'Generator')
    if not 'GOOGLE' in self.config.GAN_options: self.print_network(self.D, 'Discriminator')

  #=======================================================================================#
  #=======================================================================================#

  def print_network(self, model, name):
    num_params = 0
    for p in model.parameters():
      num_params += p.numel()
    # self.PRINT(name)
    # self.PRINT(model)
    self.PRINT("{} number of parameters: {}".format(name, num_params))
    # self.display_net(name)

  #=======================================================================================#
  #=======================================================================================#

  def load_pretrained_model(self):
    # ipdb.set_trace()
    self.G.load_state_dict(torch.load(os.path.join(
      self.config.model_save_path, '{}_G.pth'.format(self.config.pretrained_model))))
    if not 'GOOGLE' in self.config.GAN_options: self.D.load_state_dict(torch.load(os.path.join(
      self.config.model_save_path, '{}_D.pth'.format(self.config.pretrained_model))))
    self.PRINT('loaded trained models (step: {})..!'.format(self.config.pretrained_model))

  #=======================================================================================#
  #=======================================================================================#

  def build_tensorboard(self):
    # ipdb.set_trace()
    from misc.logger import Logger
    self.logger = Logger(self.config.log_path)

  #=======================================================================================#
  #=======================================================================================#

  def update_lr(self, g_lr, d_lr):
    for param_group in self.g_optimizer.param_groups:
      param_group['lr'] = g_lr
    for param_group in self.d_optimizer.param_groups:
      param_group['lr'] = d_lr

  #=======================================================================================#
  #=======================================================================================#

  def reset_grad(self):
    self.g_optimizer.zero_grad()
    self.d_optimizer.zero_grad()

  #=======================================================================================#
  #=======================================================================================#

  def to_cuda(self, x):
    if torch.cuda.is_available():
      x = x.cuda()
    return x

  #=======================================================================================#
  #=======================================================================================#

  def to_var(self, x, volatile=False, requires_grad=False):
    return Variable(self.to_cuda(x), volatile=volatile, requires_grad=requires_grad)

  #=======================================================================================#
  #=======================================================================================#

  def denorm(self, x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

  #=======================================================================================#
  #=======================================================================================#

  def one_hot(self, labels, dim):
    """Convert label indices to one-hot vector"""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    # ipdb.set_trace()
    out[np.arange(batch_size), labels.long()] = 1
    return out

  #=======================================================================================#
  #=======================================================================================#

  def get_aus(self):
    resize = lambda x: skimage.transform.resize(imageio.imread(line), (self.config.image_size,self.config.image_size))
    imgs = [resize(line).transpose(2,0,1) for line in sorted(glob.glob('data/{}/aus_flat/*.jpeg'.format(self.config.dataset_fake)))]
    imgs = torch.from_numpy(np.concatenate(imgs, axis=2).astype(np.float32)).unsqueeze(0)
    return imgs

  #=======================================================================================#
  #=======================================================================================#

  def imgShow(self, img):
    try:save_image(self.denorm(img).cpu(), 'dummy.jpg')
    except: save_image(self.denorm(img.data).cpu(), 'dummy.jpg')
    #os.system('eog dummy.jpg')  
    #os.remove('dummy.jpg')

  #=======================================================================================#
  #=======================================================================================#

  def blur(self, img, window=5, sigma=1, gray=False):
    from scipy.ndimage import filters
    trunc = (((window-1)/2.)-0.5)/sigma
    conv_img = torch.zeros_like(img.clone())
    for i in range(conv_img.size(0)):   
      if not gray:
        for j in range(conv_img.size(1)):
          conv_img[i,j] = torch.from_numpy(filters.gaussian_filter(img[i,j], sigma=sigma, truncate=trunc))
      else:
        conv_img[i] = torch.from_numpy(filters.gaussian_filter(img[i], sigma=sigma, truncate=trunc))
        # conv_img[i,j] = self.to_var(torch.from_numpy(cv.GaussianBlur(img[i,j].data.cpu().numpy(), sigmaX=sigma, sigmaY=sigma, ksize=window)), volatile=True)
    return conv_img


  #=======================================================================================#
  #=======================================================================================#

  def compute_kl(self, mu):
    # def _compute_kl(self, mu, sd):
    # mu_2 = torch.pow(mu, 2)
    # sd_2 = torch.pow(sd, 2)
    # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
    # return encoding_loss
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  #=======================================================================================#
  #=======================================================================================#

  def update_loss(self, loss, value):
    try:
      self.LOSS[loss].append(value)
    else:
      self.LOSS[loss] = []
      self.LOSS[loss].append(value)

  #=======================================================================================#
  #=======================================================================================#

  def blurRANDOM(self, img,):
    self.blurrandom +=1
    np.random.seed(self.blurrandom) 
    gray = np.random.randint(0,2,img.size(0))
    np.random.seed(self.blurrandom)
    sigma = np.random.randint(2,9,img.size(0))
    np.random.seed(self.blurrandom)
    window = np.random.randint(7,29,img.size(0))

    trunc = (((window-1)/2.)-0.5)/sigma
    # ipdb.set_trace()
    conv_img = torch.zeros_like(img.clone())
    for i in range(img.size(0)):    
      # ipdb.set_trace()
      if gray[i] and 'GRAY' in self.config.GAN_options:
        conv_img[i] = torch.from_numpy(filters.gaussian_filter(img[i], sigma=sigma[i], truncate=trunc[i]))
      else:
        for j in range(img.size(1)):
          conv_img[i,j] = torch.from_numpy(filters.gaussian_filter(img[i,j], sigma=sigma[i], truncate=trunc[i]))

    return conv_img

  #=======================================================================================#
  #=======================================================================================#

  def show_img(self, img, real_label, fake_label, ppt=False):         

    AUS_SHOW = self.get_aus()

    fake_image_list=[img]
    fake_attn_list = [self.to_var(self.denorm(img.data), volatile=True)]
    flag_time=True
    self.G.eval()

    for fl in fake_label:
      # ipdb.set_trace()
      if flag_time:
        start=time.time()
      fake_img, fake_attn = self.G(img, self.to_var(fl.data, volatile=True))
      fake_image_list.append(fake_img)
      fake_attn_list.append(fake_attn.repeat(1,3,1,1))
      if flag_time:
        elapsed = time.time() - start
        elapsed = str(datetime.timedelta(seconds=elapsed))        
        self.PRINT("Time elapsed for transforming one batch: "+elapsed)
        flag_time=False

    fake_images = torch.cat(fake_image_list, dim=3)
    fake_attn = torch.cat(fake_attn_list, dim=3)   
    shape0 = min(20, fake_images.data.cpu().shape[0])

    fake_images = fake_images.data.cpu()
    fake_attn = fake_attn.data.cpu()
    fake_images = torch.cat((AUS_SHOW, self.denorm(fake_images[:shape0])),dim=0)
    fake_attn = torch.cat((AUS_SHOW, fake_attn[:shape0]),dim=0)
    if ppt:
      name_folder = len(glob.glob('show/ppt*'))
      name_folder = 'show/ppt'+str(name_folder)
      self.PRINT("Saving outputs at: "+name_folder)
      if not os.path.isdir(name_folder): os.makedirs(name_folder)
      for n in range(1,14):
        new_fake_images = fake_images.clone()
        file_name = os.path.join(name_folder, 'tmp_all_'+str(n-1))
        new_fake_images[:,:,:,256*n:] = 0
        save_image(new_fake_images, file_name+'.jpg',nrow=1, padding=0)
      file_name = os.path.join(name_folder, 'tmp_all_'+str(n+1))
      save_image(fake_images, file_name+'.jpg',nrow=1, padding=0)       
    else:
      file_name = os.path.join(self.config.sample_path, self.config.pretrained_model+'_google.jpg')
      print('Saved at '+file_name)
      save_image(fake_images, file_name,nrow=1, padding=0)
      save_image(fake_attn, file_name.replace('.jpg', '_attn.jpg'),nrow=1, padding=0)
      #os.system('eog tmp_all.jpg')    
      #os.remove('tmp_all.jpg')

  #=======================================================================================#
  #=======================================================================================#
  @property
  def TimeNow(self):
    return str(datetime.datetime.now(pytz.timezone('Europe/Amsterdam'))).split('.')[0]

  #=======================================================================================#
  #=======================================================================================#
  @property
  def TimeNow_str(self):
    import re
    return re.sub('\D','_',self.TimeNow)

  #=======================================================================================#
  #=======================================================================================#

  def show_img_single(self, img):  
    img_ = self.denorm(img.data.cpu())
    save_image(img_.cpu(), 'show/tmp0.jpg',nrow=int(math.sqrt(img_.size(0))), padding=0)
    os.system('eog show/tmp0.jpg')    
    os.remove('show/tmp0.jpg')    

  #=======================================================================================#
  #=======================================================================================#

  def PRINT(self, str):  
    if not 'GOOGLE' in self.config.GAN_options:
      print >> self.config.log, str
      self.config.log.flush()
    print(str)

  #=======================================================================================#
  #=======================================================================================#
  def save(self, Epoch, iter):
    name = os.path.join(self.config.model_save_path, '{}_{}_{}.pth'.format(Epoch, iter, '{}'))
    torch.save(self.G.state_dict(), name.format('G'))
    torch.save(self.D.state_dict(), name.format('D'))

  #=======================================================================================#
  #=======================================================================================#
  def compute_loss_smooth(self, mat):
    return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
          torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))    

  #=======================================================================================#
  #=======================================================================================#
  def _GAN_LOSS(self, real_x, fake_x, label):
    src_real, cls_real = self.D(real_x)
    src_fake, cls_fake = self.D(fake_x)

    if 'HINGE' in self.config.GAN_options:
      loss_src = (torch.mean(torch.nn.ReLU()(1.0 - (src_real - torch.mean(src_fake)))) \
                      + torch.mean(torch.nn.ReLU()(1.0 + (src_fake - torch.mean(src_real)))))/2         
    else:
      loss_src = (torch.mean(- (src_real - torch.mean(src_fake))) \
                    + torch.mean(src_fake - torch.mean(src_real)))/2          

    # ipdb.set_trace()
    loss_cls = F.binary_cross_entropy_with_logits(cls_real, label, size_average=False) / real_x.size(0)  

    return  loss_src, loss_cls

  #=======================================================================================#
  #=======================================================================================#

  def train(self):
    # The number of iterations per epoch
    iters_per_epoch = len(self.data_loader)

    fixed_x = []
    for i, (images, labels, files) in enumerate(self.data_loader):
      # ipdb.set_trace()
      if 'BLUR' in self.config.GAN_options: images = self.blurRANDOM(images)
      fixed_x.append(images)
      if i == 1:
        break

    # Fixed inputs and target domain labels for debugging
    # ipdb.set_trace()
    fixed_x = torch.cat(fixed_x, dim=0)
    # fixed_x = self.to_var(fixed_x, volatile=True)
    
    # lr cache for decaying
    g_lr = self.config.g_lr
    d_lr = self.config.d_lr

    # Start with trained model if exists
    if self.config.pretrained_model:
      start = int(self.config.pretrained_model.split('_')[0])
      for i in range(start):
        # if (i+1) > (self.config.num_epochs - self.config.num_epochs_decay):
        if (i+1) %self.config.num_epochs_decay==0:
          g_lr = (g_lr / 10.)
          d_lr = (d_lr / 10.)
          self.update_lr(g_lr, d_lr)
          self.PRINT ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))     
    else:
      start = 0

    last_model_step = len(self.data_loader)

    # Start training

    self.PRINT("Log path: "+self.config.log_path)

    Log = "---> batch size: {}, fold: {}, img: {}, GPU: {}, !{}, [{}]\n-> GAN_options:".format(\
        self.config.batch_size, self.config.fold, self.config.image_size, \
        self.config.GPU, self.config.mode_data, self.config.PLACE) 

    for item in sorted(self.config.GAN_options):
      Log += ' [*{}]'.format(item.upper())
    Log += ' [*{}]'.format(self.config.dataset_fake)
    self.PRINT(Log)
    start_time = time.time()

    for e in range(start, self.config.num_epochs):
      E = str(e+1).zfill(3)
      self.D.train()
      self.G.train()
      self.LOSS = {}
      desc_bar = 'Epoch: %d/%d'%(e,self.config.num_epochs)
      progress_bar = tqdm(enumerate(self.data_loader), \
          total=len(self.data_loader), desc=desc_bar, ncols=10)
      for i, (real_x, real_label, files) in progress_bar:
        # ipdb.set_trace()   
        if real_x.size(0)==self.config.batch_size:

          # save_image(self.denorm(real_x.cpu()), 'dm1.png',nrow=1, padding=0)
          #=======================================================================================#
          #========================================== BLUR =======================================#
          #=======================================================================================#
          np.random.seed(i+(e*len(self.data_loader)))
          if 'BLUR' in self.config.GAN_options and np.random.randint(0,2,1)[0]:
            real_x = self.blurRANDOM(real_x)
          loss = {}

          #=======================================================================================#
          #====================================== DATA2VAR =======================================#
          #=======================================================================================#
          # Generat fake labels randomly (target domain labels)
          rand_idx = torch.randperm(real_label.size(0))
          fake_label = real_label[rand_idx]
          real_c = real_label.clone()
          fake_c = fake_label.clone()

          # Convert tensor to variable
          real_x = self.to_var(real_x)
          real_c = self.to_var(real_c)       # input for the generator
          fake_c = self.to_var(fake_c)
          real_label = self.to_var(real_label)   # this is same as real_c if dataset == 'CelebA'
          fake_label = self.to_var(fake_label)

          # split = lambda x: (x[:x.size(0)//2], x[x.size(0)//2:])
          split = lambda x: (x, x)
          real_x0, real_x1 = split(real_x)
          real_label0, real_label1 = split(real_label)
          real_c0, real_c1 = split(real_c)
          fake_c0, fake_c1 = split(fake_c)
          fake_label0, fake_label1 = split(fake_label)
          # ipdb.set_trace()  

          if 'Attention' in self.config.GAN_options:
            fake_x0, fake_mask = self.G(real_x0, fake_c0)    
            # fake_x0 = fake_mask * real_x0 + (1 - fake_mask) * fake_x0            
          else:
            fake_x0 = self.G(real_x0, fake_c0)

          #=======================================================================================#
          #======================================== Train D ======================================#
          #=======================================================================================#
          d_loss_src, d_loss_cls = self._GAN_LOSS(real_x0, fake_x0, real_label0)

          d_loss_cls = self.config.lambda_cls * d_loss_cls  

          # Backward + Optimize       
          d_loss = d_loss_src + d_loss_cls

          self.reset_grad()
          d_loss.backward()
          self.d_optimizer.step()

          loss['D/src'] = d_loss_src.data[0]
          loss['D/cls'] = d_loss_cls.data[0]          
          self.update_loss('D/src', loss['D/src'])
          self.update_loss('D/cls', loss['D/cls'])
                    
          #=======================================================================================#
          #=================================== Gradient Penalty ==================================#
          #=======================================================================================#
          # Compute gradient penalty
          if not 'HINGE' in self.config.GAN_options:
            alpha = torch.rand(real_x0.size(0), 1, 1, 1).cuda().expand_as(real_x0)
            # ipdb.set_trace()
            interpolated = Variable(alpha * real_x0.data + (1 - alpha) * fake_x0.data, requires_grad=True)
            out, out_cls = self.D(interpolated)

            grad = torch.autograd.grad(outputs=out,
                           inputs=interpolated,
                           grad_outputs=torch.ones(out.size()).cuda(),
                           retain_graph=True,
                           create_graph=True,
                           only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1)**2)

            # Backward + Optimize
            d_loss_gp = self.config.lambda_gp * d_loss_gp
            loss['D/gp'] = d_loss_gp.data[0]
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
            self.update_loss('D/gp', loss['D/gp'])
          
          #=======================================================================================#
          #======================================= Train G =======================================#
          #=======================================================================================#
          if (i+1) % self.config.d_train_repeat == 0:

            # Original-to-target and target-to-original domain

            ############################## Stochastic Part ##################################
            if 'Stochastic' in self.config.GAN_options:
              style_real, _ = self.G.get_style(real_x1)
              if 'style_pseudo_random' in self.config.GAN_options:
                style_fake = style_real[self.to_cuda(rand_idx)]

              else:
                style_fake = self.to_var(self.G.random_style(real_x1))
              # if 'style_labels' in self.config.GAN_options and 'style_pseudo_random' not in self.config.GAN_options:
                # if 'style_label_net' not in self.config.GAN_options: 
                #   style_real *= real_c1.unsqueeze(2)
                # style_fake *= fake_c1.unsqueeze(2)
            else:
              style_real = style_fake = None

            ############################## Attention Part ###################################
            if 'Attention' in self.config.GAN_options:
              fake_x1, fake_mask1 = self.G(real_x1, fake_c1, stochastic = style_fake)
              # fake_x1 = fake_mask1 * real_x1 + (1 - fake_mask1) * fake_x1          
              rec_x1, rec_real_mask1 = self.G(fake_x1, real_c1, stochastic = style_real)  
              # rec_x1 = rec_real_mask1 * fake_x1 + (1 - rec_real_mask1) * rec_x1   

              g_loss_mask = self.config.lambda_mask * (torch.mean(rec_real_mask1) + torch.mean(fake_mask1))
              g_loss_mask_smooth = self.config.lambda_mask_smooth * (self.compute_loss_smooth(fake_mask1) + self.compute_loss_smooth(rec_real_mask1))
              loss['G/mask'] = g_loss_mask.data[0]
              loss['G/mask_sm'] = g_loss_mask_smooth.data[0]     
              self.update_loss('G/mask', loss['G/mask'])
              self.update_loss('G/mask_sm', loss['G/mask_sm'])
     
              g_loss = g_loss_mask + g_loss_mask_smooth
            else:
              fake_x1 = self.G(real_x1, fake_c1, stochastic = style_fake)
              rec_x1  = self.G(fake_x1, real_c1, stochastic = style_real) 
              g_loss = 0
            
            ##############################       GAN        #################################

            g_loss_src, g_loss_cls = self._GAN_LOSS(fake_x1, real_x1, fake_label1)

            if 'L1_LOSS' in self.config.GAN_options:
              g_loss_rec = F.l1_loss(real_x1, fake_x1) + \
                           F.l1_loss(fake_x1, rec_x1)
            else:
              g_loss_rec = F.l1_loss(real_x1, rec_x1)

            g_loss_rec = g_loss_rec*self.config.lambda_rec
            g_loss_cls = g_loss_cls*self.config.lambda_cls

            loss['G/src'] = g_loss_src.data[0]
            loss['G/rec'] = g_loss_rec.data[0]
            loss['G/cls'] = g_loss_cls.data[0]

            self.update_loss('G/src', loss['G/src'])
            self.update_loss('G/rec', loss['G/rec'])
            self.update_loss('G/cls', loss['G/cls'])

            # Backward + Optimize
            g_loss += g_loss_src + g_loss_rec + g_loss_cls 

            # loss KL style
            ############################## KL Part ###################################
            if 'AdaIn' in self.config.GAN_options and 'kl_loss' in self.config.GAN_options:
              _style_fake, _style_cls_fake = self.G.get_style(fake_x1)
              _style_rec, _style_cls_real = self.G.get_style(rec_x1)
              # if 'style_labels' in self.config.GAN_options and 'style_label_net' not in  self.config.GAN_options  and 'style_pseudo_random' not in self.config.GAN_options:
              #   # ipdb.set_trace()
              #   style_real *= real_c1.unsqueeze(2)
              #   _style_fake *= fake_c1.unsqueeze(2)
              #   _style_rec *= real_c1.unsqueeze(2)              
              g_loss_kl = self.config.lambda_kl * (#self.compute_kl(style_real) +\
                                                   self.compute_kl(_style_fake) +
                                                   self.compute_kl(_style_rec))
              loss['G/kl'] = g_loss_kl.data[0]
              self.update_loss('G/kl', loss['G/kl'])

              g_loss += g_loss_kl

              ############################## VAE Part ###################################
              if 'vae_like' in self.config.GAN_options:
                rand = self.to_var(self.G.dec_style.random_noise(style_real.size(0)))
                rec_real_vae = self.G.dec_style(style_real.view(*rand.shape) + rand)

                rand = self.to_var(self.G.dec_style.random_noise(_style_fake.size(0)))
                rec_fake_vae = self.G.dec_style(_style_fake.view(*rand.shape) + rand)                
                
                g_loss_vae = self.config.lambda_kl * (#self.compute_kl(style_real) +\
                                                     F.l1_loss(real_x1, rec_real_vae) +
                                                     F.l1_loss(fake_x1, rec_fake_vae)
                                                     )
                loss['G/vae'] = g_loss_vae.data[0]
                self.update_loss('G/vae', loss['G/vae'])

                g_loss += g_loss_vae              

            ############################## Stochastic Part ###################################
            if 'Stochastic' in self.config.GAN_options:              
              g_loss_style = self.config.lambda_style * (F.l1_loss(style_real, _style_rec) +
                                                         F.l1_loss(_style_fake, style_fake))
              loss['G/sty'] = g_loss_style.data[0]
              self.update_loss('G/sty', loss['G/sty'])
              g_loss += g_loss_style

              if 'style_label_net' in self.config.GAN_options:
                g_style_cls_fake = F.binary_cross_entropy_with_logits(
                  _style_cls_fake, fake_label1, size_average=False) / fake_x1.size(0)
                g_style_cls_real = F.binary_cross_entropy_with_logits(
                  _style_cls_real, real_label1, size_average=False) / real_x1.size(0)
                g_loss_style_cls = self.config.lambda_style * (g_style_cls_fake + g_style_cls_real)              
                loss['G/sty_cls'] = g_loss_style_cls.data[0]
                self.update_loss('G/sty_cls', loss['G/sty_cls'])

                g_loss += g_loss_style_cls                

            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

        #=======================================================================================#
        #========================================MISCELANEOUS===================================#
        #=======================================================================================#

        # self.PRINT out log info
        if (i+1) % self.config.log_step == 0 or (i+1)==last_model_step:
          # progress_bar.set_postfix(G_loss_rec=np.array(self.LOSS['G/rec']).mean())
          progress_bar.set_postfix(**loss)
          if (i+1)==last_model_step: progress_bar.set_postfix('')
          if self.config.use_tensorboard:
            for tag, value in loss.items():
              self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)
          name = os.path.join(self.config.sample_path, 'current_fake.jpg')
          self.save_fake_output(fixed_x, name)


    # Translate fixed images for debugging
        if (i+1) % self.config.sample_step == 0 or (i+1)==last_model_step or i+e==0:
          name = os.path.join(self.config.sample_path, '{}_{}_fake.jpg'.format(E, i+1))
          # self.save_debug(fixed_x, fixed_c_list, name)
          self.save_fake_output(fixed_x, name)
          # self.PRINT('Translated images and saved into {}..!'.format(self.sample_path))

      self.save(E, i+1)
                 
      #Stats per epoch
      elapsed = time.time() - start_time
      elapsed = str(datetime.timedelta(seconds=elapsed))
      # log = '!Elapsed: %s | [F1_VAL: %0.3f LOSS_VAL: %0.3f]\nTrain'%(elapsed, np.array(f1).mean(), np.array(loss).mean())
      log = '--> %s | Elapsed (%d/%d) : %s | %s\nTrain'%(self.TimeNow, e, self.config.num_epochs, elapsed, Log)
      for tag, value in sorted(self.LOSS.items()):
        log += ", {}: {:.4f}".format(tag, np.array(value).mean())   

      self.PRINT(log)
      self.data_loader.dataset.shuffle(e) #Shuffling dataset after each epoch

      # Decay learning rate     
      # if (e+1) > (self.config.num_epochs - self.config.num_epochs_decay):
      if (e+1) % self.config.num_epochs_decay ==0:
        g_lr = (g_lr / 10)
        d_lr = (d_lr / 10)
        self.update_lr(g_lr, d_lr)
        self.PRINT ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

  #=======================================================================================#
  #=======================================================================================#

  def save_fake_output(self, real_x, save_path, label=None):
    self.G.eval()
    Attention = 'Attention' in self.config.GAN_options
    Stochastic = 'Stochastic' in self.config.GAN_options
    real_x = self.to_var(real_x, volatile=True)

    # target_c_list = []
    target_c= torch.from_numpy(np.zeros((real_x.size(0), self.config.c_dim), dtype=np.float32))

    target_c_list = [self.to_var(target_c, volatile=True)]

    for j in range(self.config.c_dim):
      if label is None:
        target_c[:]=0 
        target_c[:,j]=1       
      else:
        # ipdb.set_trace()
        print(j+1, sorted(list(set(map(int, (self.config.AUs['EMOTIONNET']*label[j].cpu().numpy()).tolist())))))
        target_c=label[j].repeat(label.size(0),1)      
      target_c_list.append(self.to_var(target_c, volatile=True))      

    # Start translations
    fake_image_list = [real_x]
    if Stochastic: 
      n_rep = 5
      n_img = 5
      real_x0 = real_x[n_img].repeat(n_rep,1,1,1)
      fake_rand_list = [real_x0]
      if Attention: 
        rand_attn_list = [self.to_var(self.denorm(real_x0.data), volatile=True)]
        # ipdb.set_trace()
    if Attention: 
      fake_attn_list = [self.to_var(self.denorm(real_x.data), volatile=True)]
    for target_c in target_c_list:   
      if Stochastic: 
        style, _ = self.G.get_style(real_x)
        style_rand=self.to_var(self.G.random_style(real_x0), volatile=True)
      else: 
        style = None   

      if Attention and Stochastic: 
        fake_x, fake_attn = self.G(real_x, target_c, stochastic=style)
        # fake_x = fake_attn * real_x + (1 - fake_attn) * fake_x 
        target_c0 = target_c[:n_rep]
        # ipdb.set_trace()
        style_rand[0] = style[n_img] #Compare with the original style
        style_rand[1] = style[6]
        style_rand[2] = style[9]
        # fake_x0, fake_attn0 = self.G(real_x0, target_c0, stochastic=style_rand*target_c0.unsqueeze(2))
        fake_x0, fake_attn0 = self.G(real_x0, target_c0, stochastic=style_rand)
        # fake_x0 = fake_attn0 * real_x0 + (1 - fake_attn0) * fake_x0 
        
        rand_idx = torch.randperm(real_label.size(0))        
        rand_attn_list.append(fake_attn0.repeat(1,3,1,1))
      elif Stochastic:
        fake_x = self.G(real_x, target_c, stochastic=style)
        target_c0 = target_c[:n_rep]
        # ipdb.set_trace()
        style_rand[0] = style[n_img] #Compare with the original style
        style_rand[1] = style[6]
        style_rand[2] = style[9]     
        # fake_x0 = self.G(real_x0, target_c0, stochastic=style_rand*target_c0.unsqueeze(2))   
        fake_x0 = self.G(real_x0, target_c0, stochastic=style_rand)   
        fake_rand_list.append(fake_x0)    
      elif Attention:
        fake_x, fake_attn = self.G(real_x, target_c, stochastic=style)
        # fake_x = fake_attn * real_x + (1 - fake_attn) * fake_x 
      else:
        fake_x = self.G(real_x, target_c, stochastic=style)
      # ipdb.set_trace()
      fake_image_list.append(fake_x)
      if Attention: fake_attn_list.append(fake_attn.repeat(1,3,1,1))

    # ipdb.set_trace()
    fake_images = self.denorm(torch.cat(fake_image_list, dim=3).data.cpu())
    fake_images = torch.cat((self.get_aus(), fake_images), dim=0)
    save_image(fake_images, save_path, nrow=1, padding=0)
    if Attention: 
      fake_attn = torch.cat(fake_attn_list, dim=3).data.cpu()
      fake_attn = torch.cat((self.get_aus(), fake_attn), dim=0)
      save_image(fake_attn, save_path.replace('fake', 'attn'), nrow=1, padding=0)
    if Stochastic:
      fake_images = self.denorm(torch.cat(fake_rand_list, dim=3).data.cpu())
      fake_images = torch.cat((self.get_aus(), fake_images), dim=0)
      save_image(fake_images, save_path.replace('fake', 'style'), nrow=1, padding=0)
      if Attention:
        fake_attn = torch.cat(rand_attn_list, dim=3).data.cpu()
        fake_attn = torch.cat((self.get_aus(), fake_attn), dim=0)
        save_image(fake_attn, save_path.replace('fake', 'attn'), nrow=1, padding=0)        
    self.G.train()

  #=======================================================================================#
  #=======================================================================================#

  def test(self, dataset='', load=False):
    import re
    from data_loader import get_loader
    if dataset=='': dataset = 'BP4D'
    if self.config.pretrained_model in ['',None] or load:
      last_file = sorted(glob.glob(os.path.join(self.config.model_save_path, '*_D.pth')))[-1]
      last_name = '_'.join(os.path.basename(last_file).split('_')[:2])
    else:
      last_name = self.config.pretrained_model

    G_path = os.path.join(self.config.model_save_path, '{}_G.pth'.format(last_name))
    self.G.load_state_dict(torch.load(G_path))
    self.G.eval()
    # ipdb.set_trace()

    data_loader_val = get_loader(self.config.metadata_path, self.config.image_size,
                 self.config.image_size, self.config.batch_size, shuffling = True,
                 dataset=[dataset], mode='test', AU=self.config.AUs)[0]  

    for i, (real_x, org_c, files) in enumerate(data_loader_val):
      # ipdb.set_trace()
      save_path = os.path.join(self.config.sample_path, '{}_fake_val_{}_{}_{}.jpg'.format(last_name, dataset, i+1, '{}'))
      name = save_path.format(self.TimeNow_str)
      if 'REAL_LABELS' in self.config.GAN_options: 
        self.save_fake_output(real_x, name, label=org_c)
      else: 
        self.save_fake_output(real_x, name)
      self.PRINT('Translated test images and saved into "{}"..!'.format(name))
      if i==1: break
    # if 'Stochastic' in self.config.GAN_options:
    #   # ipdb.set_trace()
    #   self.save_fake_output(real_x[0].repeat(3,1,1,1), save_path.format(self.TimeNow_str))      

  #=======================================================================================#
  #=======================================================================================#

  def DEMO(self, path):
    from data_loader import get_loader
    import re
    if self.config.pretrained_model in ['',None]:
      last_file = sorted(glob.glob(os.path.join(self.config.model_save_path,  '*_D.pth')))[-1]
      last_name = '_'.join(os.path.basename(last_file).split('_')[:2])
    else:
      last_name = self.config.pretrained_model

    G_path = os.path.join(self.config.model_save_path, '{}_G.pth'.format(last_name))
    self.G.load_state_dict(torch.load(G_path))
    self.G.eval()
    # ipdb.set_trace()

    data_loader = get_loader(path, self.config.image_size,
                 self.config.image_size, self.config.batch_size, shuffling = True,
                 dataset=['DEMO'], mode='test', AU=self.config.AUs)[0]  

    for real_x in data_loader:
      # ipdb.set_trace()
      save_path = os.path.join(self.config.sample_path, '{}_fake_val_DEMO_{}.jpg'.format(last_name, self.TimeNow_str))
      self.save_fake_output(real_x, save_path)
      self.PRINT('Translated test images and saved into "{}"..!'.format(save_path))