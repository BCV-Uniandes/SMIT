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
from utils import f1_score, f1_score_max, F1_TEST
import imageio
import skimage.transform
import math
from scipy.ndimage import filters
import warnings
warnings.filterwarnings('ignore')

class Solver(object):

  def __init__(self, data_loader, config):
    # Data loader
    self.data_loader = data_loader
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
    y = self.C(self.to_var(torch.randn(1,3,self.config.image_size,self.config.image_size)))
    if name=='discriminator':
      g=make_dot(y, params=dict(self.D.named_parameters()))
    elif name=='generator':
      g=make_dot(y, params=dict(self.G.named_parameters()))
    filename=name
    g.filename=filename
    g.render()
    os.remove(filename)

    from utils import pdf2png
    pdf2png(filename)
    print('Network saved at {}.png'.format(filename))

  #=======================================================================================#
  #=======================================================================================#
  def build_model(self):
    # Define a generator and a discriminator
    if self.config.DENSENET:
      from models.densenet import Generator, densenet121 as Discriminator
      self.G = Generator(self.config.g_conv_dim, self.config.c_dim, self.config.g_repeat_num)
      self.D = Discriminator(num_classes = self.config.c_dim) 
    else:
      from model import Generator, Discriminator
      self.G = Generator(self.config.g_conv_dim, self.config.c_dim, self.config.g_repeat_num)
      self.D = Discriminator(self.config.image_size, self.config.d_conv_dim, self.config.c_dim, self.config.d_repeat_num) 

    # Optimizers
    self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.config.g_lr, [self.config.beta1, self.config.beta2])
    self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.config.d_lr, [self.config.beta1, self.config.beta2])

    # Print networks
    self.print_network(self.G, 'G')
    self.print_network(self.D, 'D')

    if torch.cuda.is_available():
      self.G.cuda()
      self.D.cuda()

  #=======================================================================================#
  #=======================================================================================#

  def print_network(self, model, name):
    num_params = 0
    for p in model.parameters():
      num_params += p.numel()
    # print(name)
    # print(model)
    print("The number of parameters: {}".format(num_params))

  #=======================================================================================#
  #=======================================================================================#

  def load_pretrained_model(self):
    # ipdb.set_trace()
    self.G.load_state_dict(torch.load(os.path.join(
      self.config.model_save_path, '{}_G.pth'.format(self.config.pretrained_model))))
    self.D.load_state_dict(torch.load(os.path.join(
      self.config.model_save_path, '{}_D.pth'.format(self.config.pretrained_model))))
    print('loaded trained models (step: {})..!'.format(self.config.pretrained_model))

  #=======================================================================================#
  #=======================================================================================#

  def build_tensorboard(self):
    # ipdb.set_trace()
    from logger import Logger
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

  def threshold(self, x):
    x = x.clone()
    x = (x >= 0.5).float()
    return x

  #=======================================================================================#
  #=======================================================================================#

  def compute_accuracy(self, x, y, dataset):
    x = F.sigmoid(x)
    predicted = self.threshold(x)
    correct = (predicted == y).float()
    accuracy = torch.mean(correct, dim=0) * 100.0

    return accuracy

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

  def focal_loss(self, out, label):
    alpha=0.5
    gamma=0.5
    # sigmoid = out.clone().sigmoid()
    # ipdb.set_trace()
    max_val = (-out).clamp(min=0)
    # pt = -out + out * label - max_val - ((-max_val).exp() + (out + max_val).exp())
    pt = out - out * label + max_val + ((-max_val).exp() + (-out - max_val).exp()).log()

    # pt = sigmoid*label + (1-sigmoid)*(1-label)
    FL = alpha*torch.pow(1-(-pt).exp(),gamma)*pt
    FL = FL.sum()
    # ipdb.set_trace()
    # FL = F.binary_cross_entropy_with_logits(out, label, size_average=False)
    return FL      

  #=======================================================================================#
  #=======================================================================================#

  def get_aus(self):
    resize = lambda x: skimage.transform.resize(imageio.imread(line), (self.config.image_size,self.config.image_size))
    imgs = [resize(line).transpose(2,0,1) for line in sorted(glob.glob('/home/afromero/datos2/aus_flat/*.jpeg'))]
    imgs = torch.from_numpy(np.concatenate(imgs, axis=2).astype(np.float32)).unsqueeze(0)
    return imgs

  #=======================================================================================#
  #=======================================================================================#

  def imgShow(self, img):
    try:save_image(self.denorm(img).cpu(), 'dummy.jpg')
    except: save_image(self.denorm(img.data).cpu(), 'dummy.jpg')
    os.system('eog dummy.jpg')  
    os.remove('dummy.jpg')

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
      if gray[i] and self.config.GRAY:
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
    flag_time=True
    self.G.eval()

    for fl in fake_label:
      # ipdb.set_trace()
      if flag_time:
        start=time.time()
      fake_image_list.append(self.G(img, self.to_var(fl.data, volatile=True)))
      if flag_time:
        elapsed = time.time() - start
        elapsed = str(datetime.timedelta(seconds=elapsed))        
        print("Time elapsed for transforming one batch: "+elapsed)
        flag_time=False

    fake_images = torch.cat(fake_image_list, dim=3)   
    shape0 = min(10, fake_images.data.cpu().shape[0])

    fake_images = fake_images.data.cpu()
    fake_images = torch.cat((AUS_SHOW, self.denorm(fake_images[:shape0])),dim=0)
    if ppt:
      name_folder = len(glob.glob('show/ppt*'))
      name_folder = 'show/ppt'+str(name_folder)
      print("Saving outputs at: "+name_folder)
      if not os.path.isdir(name_folder): os.makedirs(name_folder)
      for n in range(1,14):
        new_fake_images = fake_images.clone()
        file_name = os.path.join(name_folder, 'tmp_all_'+str(n-1))
        new_fake_images[:,:,:,256*n:] = 0
        save_image(new_fake_images, file_name+'.jpg',nrow=1, padding=0)
      file_name = os.path.join(name_folder, 'tmp_all_'+str(n+1))
      save_image(fake_images, file_name+'.jpg',nrow=1, padding=0)       
    else:
      file_name = 'tmp_all'
      save_image(fake_images, file_name+'.jpg',nrow=1, padding=0)
      os.system('eog tmp_all.jpg')    
      os.remove('tmp_all.jpg')

  #=======================================================================================#
  #=======================================================================================#

  def show_img_single(self, img):  
    img_ = self.denorm(img.data.cpu())
    save_image(img_.cpu(), 'show/tmp0.jpg',nrow=int(math.sqrt(img_.size(0))), padding=0)
    os.system('eog show/tmp0.jpg')    
    os.remove('show/tmp0.jpg')    

  #=======================================================================================#
  #=======================================================================================#

  def train(self):
    # The number of iterations per epoch
    iters_per_epoch = len(self.data_loader)

    fixed_x = []
    real_c = []
    for i, (images, labels, files) in enumerate(self.data_loader):
      # ipdb.set_trace()
      if self.config.BLUR: images = self.blurRANDOM(images)
      fixed_x.append(images)
      real_c.append(labels)
      if i == 1:
        break

    # Fixed inputs and target domain labels for debugging
    # ipdb.set_trace()
    fixed_x = torch.cat(fixed_x, dim=0)
    fixed_x = self.to_var(fixed_x, volatile=True)
    real_c = torch.cat(real_c, dim=0)

    fixed_c_list = [self.to_var(torch.zeros(fixed_x.size(0), self.config.c_dim), volatile=True)]
    for i in range(self.config.c_dim):
      # ipdb.set_trace()
      fixed_c = self.one_hot(torch.ones(fixed_x.size(0)) * i, self.config.c_dim)
      fixed_c_list.append(self.to_var(fixed_c, volatile=True))

    
    # lr cache for decaying
    g_lr = self.config.g_lr
    d_lr = self.config.d_lr

    # Start with trained model if exists
    if self.config.pretrained_model:
      start = int(self.config.pretrained_model.split('_')[0])
      for i in range(start):
        if (i+1) > (self.config.num_epochs - self.config.num_epochs_decay):
          g_lr -= (self.config.g_lr / float(self.config.num_epochs_decay))
          d_lr -= (self.config.d_lr / float(self.config.num_epochs_decay))
          self.update_lr(g_lr, d_lr)
          print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))     
    else:
      start = 0

    last_model_step = len(self.data_loader)

    # Start training

    print("Log path: "+self.config.log_path)

    Log = "---> batch size: {}, fold: {}, img: {}, GPU: {}, !{}".format(\
        self.config.batch_size, self.config.fold, self.config.image_size, \
        self.config.GPU, self.config.mode_data) 

    if self.config.COLOR_JITTER: Log += ' [*COLOR_JITTER]'
    if self.config.BLUR: Log += ' [*BLUR]'
    if self.config.GRAY: Log += ' [*GRAY]'
    if self.config.LSGAN: Log += ' [*LSGAN]'
    if self.config.L1_LOSS: Log += ' [*L1_LOSS]'
    if self.config.L2_LOSS: Log += ' [*L2_LOSS]'
    print(Log)
    loss_cum = {}
    start_time = time.time()
    flag_init=True
    for e in range(start, self.config.num_epochs):
      E = str(e+1).zfill(3)
      self.D.train()
      self.G.train()

      # if flag_init:
      #   f1, loss, f1_1 = self.val_cls(init=True)   
      #   log = '[F1_VAL: %0.3f (F1_1: %0.3f) LOSS_VAL: %0.3f]'%(np.mean(f1), np.mean(f1_1), np.mean(loss))
      #   print(log)
      #   flag_init = False

      desc_bar = 'Epoch: %d/%d'%(e,self.config.num_epochs)
      progress_bar = tqdm(enumerate(self.data_loader), \
          total=len(self.data_loader), desc=desc_bar, ncols=10)
      for i, (real_x, real_label, files) in progress_bar:
        
        # save_image(self.denorm(real_x.cpu()), 'dm1.png',nrow=1, padding=0)
        #=======================================================================================#
        #========================================== BLUR =======================================#
        #=======================================================================================#
        np.random.seed(i+(e*len(self.data_loader)))
        if self.config.BLUR and np.random.randint(0,2,1)[0]:
          # for i in range(3,27,2):
          #   for j in range(1,10):
          #     save_image(self.denorm(self.blur(real_x,i,j)), 'dummy%s_%s_color.jpg'%(i,j))      
          # ipdb.set_trace()    
          # save_image(self.denorm(self.blurRANDOM(real_x)), 'dummy.jpg')
          real_x = self.blurRANDOM(real_x)
          

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
        
        #=======================================================================================#
        #======================================== Train D ======================================#
        #=======================================================================================#
        out_src, out_cls = self.D(real_x)

        if self.config.LSGAN:
          d_loss_real = F.mse_loss(out_src, torch.ones_like(out_src))
        else:
          d_loss_real = - torch.mean(out_src)

        d_loss_cls = F.binary_cross_entropy_with_logits(
          out_cls, real_label, size_average=False) / real_x.size(0)

        # Compute loss with fake images    
        fake_x = self.G(real_x, fake_c)
        fake_x_D = Variable(fake_x.data)
        out_src, out_cls = self.D(fake_x)

        if self.config.LSGAN:
          d_loss_fake = F.mse_loss(out_src, torch.zeros_like(out_src))
        else:
          d_loss_fake = torch.mean(out_src)

        # Backward + Optimize
        
        d_loss = d_loss_real + d_loss_fake + self.config.lambda_cls * d_loss_cls
        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

        #=======================================================================================#
        #=================================== Gradient Penalty ==================================#
        #=======================================================================================#
        # Compute gradient penalty
        if not self.config.LSGAN:
          alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
          # ipdb.set_trace()
          interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
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
          d_loss = self.config.lambda_gp * d_loss_gp
          self.reset_grad()
          d_loss.backward()
          self.d_optimizer.step()
        else:
          d_loss_gp = Variable(torch.FloatTensor([0]), volatile=True)

        # Logging
        loss = {}
        loss['D/real'] = d_loss_real.data[0]
        loss['D/fake'] = d_loss_fake.data[0]
        loss['D/cls'] = d_loss_cls.data[0]*self.config.lambda_cls
        loss['D/gp'] = d_loss_gp.data[0]*self.config.lambda_gp
        if len(loss_cum.keys())==0: 
          loss_cum['D/real'] = []; loss_cum['D/fake'] = []
          loss_cum['D/cls'] = []; loss_cum['G/cls'] = []
          loss_cum['G/fake'] = []; loss_cum['G/rec'] = []
          loss_cum['D/gp'] = []; loss_cum['G/l1'] = []
        loss_cum['D/real'].append(d_loss_real.data[0])
        loss_cum['D/fake'].append(d_loss_fake.data[0])
        loss_cum['D/cls'].append(d_loss_cls.data[0]*self.config.lambda_cls)
        loss_cum['D/gp'].append(d_loss_gp.data[0]*self.config.lambda_gp)

        #=======================================================================================#
        #======================================= Train G =======================================#
        #=======================================================================================#
        if (i+1) % self.config.d_train_repeat == 0:

          # Original-to-target and target-to-original domain
          fake_x = self.G(real_x, fake_c)
          rec_x = self.G(fake_x, real_c)
          out_src, out_cls = self.D(fake_x)
          
          if self.config.LSGAN:
            g_loss_fake = F.mse_loss(out_src, torch.ones_like(out_src))
          else:          
            g_loss_fake = - torch.mean(out_src)

          g_loss_rec = torch.mean(torch.abs(real_x - rec_x))

          g_loss_cls = F.binary_cross_entropy_with_logits(
            out_cls, fake_label, size_average=False) / fake_x.size(0)

          if self.config.L1_LOSS:
            g_l1 = F.l1_loss(fake_x, real_x) + \
                 F.l1_loss(rec_x, fake_x)
            # ipdb.set_trace()
          else:
            g_l1 = 0.0


          # Backward + Optimize
          g_loss = g_loss_fake + \
                   self.config.lambda_rec * g_loss_rec + \
                   self.config.lambda_cls * g_loss_cls + \
                   self.config.lambda_l1 * g_l1
          self.reset_grad()
          g_loss.backward()
          self.g_optimizer.step()

          # Logging
          loss['G/fake'] = g_loss_fake.data[0]
          loss['G/rec'] = g_loss_rec.data[0]*self.config.lambda_rec
          loss['G/cls'] = g_loss_cls.data[0]*self.config.lambda_cls
          loss['G/l1'] = g_l1.data[0]*self.config.lambda_l1


          loss_cum['G/fake'].append(g_loss_fake.data[0])
          loss_cum['G/rec'].append(g_loss_rec.data[0]*self.config.lambda_rec)
          loss_cum['G/cls'].append(g_loss_cls.data[0]*self.config.lambda_cls)
          loss_cum['G/l1'].append(g_l1.data[0]*self.config.lambda_l1)


        #=======================================================================================#
        #========================================MISCELANEOUS===================================#
        #=======================================================================================#

        # Print out log info
        if (i+1) % self.config.log_step == 0 or (i+1)==last_model_step:
          # progress_bar.set_postfix(G_loss_rec=np.array(loss_cum['G/loss_rec']).mean())
          # progress_bar.set_postfix(**loss)
          # if (i+1)==last_model_step: progress_bar.set_postfix('')
          if self.config.use_tensorboard:
            for tag, value in loss.items():
              self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

        # Translate fixed images for debugging
        if (i+1) % self.config.sample_step == 0 or (i+1)==last_model_step or i+e==0:
          self.G.eval()
          fake_image_list = [fixed_x]
          # ipdb.set_trace()
          for fixed_c in fixed_c_list:
            fake_image_list.append(self.G(fixed_x, fixed_c))
          fake_images = torch.cat(fake_image_list, dim=3)
          # ipdb.set_trace()
          shape0 = min(64, fake_images.data.cpu().shape[0])
          img_denorm = self.denorm(fake_images.data.cpu()[:shape0])
          save_image(img_denorm,
            os.path.join(self.config.sample_path, '{}_{}_fake.jpg'.format(E, i+1)),nrow=1, padding=0)
          # print('Translated images and saved into {}..!'.format(self.sample_path))

      torch.save(self.G.state_dict(),
        os.path.join(self.config.model_save_path, '{}_{}_G.pth'.format(E, i+1)))
      torch.save(self.D.state_dict(),
        os.path.join(self.config.model_save_path, '{}_{}_D.pth'.format(E, i+1)))


      #=======================================================================================#
      #=========================================METRICS=======================================#
      #=======================================================================================#
      #F1 val
      # f1, loss,_ = self.val_cls()
      # if self.use_tensorboard:
      #   # print("Log path: "+self.log_path)
      #   for idx, au in enumerate(cfg.AUs):
      #     self.logger.scalar_summary('F1_val_'+str(au).zfill(2), f1[idx], e * iters_per_epoch + i + 1)      
      #     self.logger.scalar_summary('Loss_val_'+str(au).zfill(2), loss[idx], e * iters_per_epoch + i + 1)      
      #   self.logger.scalar_summary('F1_val_mean', np.array(f1).mean(), e * iters_per_epoch + i + 1)     
      #   self.logger.scalar_summary('Loss_val_mean', np.array(loss).mean(), e * iters_per_epoch + i + 1)     

      #   for tag, value in loss_cum.items():
      #     self.logger.scalar_summary(tag, np.array(value).mean(), e * iters_per_epoch + i + 1)   
                 
      #Stats per epoch
      elapsed = time.time() - start_time
      elapsed = str(datetime.timedelta(seconds=elapsed))
      # log = '!Elapsed: %s | [F1_VAL: %0.3f LOSS_VAL: %0.3f]\nTrain'%(elapsed, np.array(f1).mean(), np.array(loss).mean())
      log = '!Elapsed: %s\nTrain'%(elapsed)
      for tag, value in sorted(loss_cum.items()):
        log += ", {}: {:.4f}".format(tag, np.array(value).mean())   

      print(log)

      # Decay learning rate     
      if (e+1) > (self.config.num_epochs - self.config.num_epochs_decay):
        g_lr -= (self.config.g_lr / float(self.config.num_epochs_decay))
        d_lr -= (self.config.d_lr / float(self.config.num_epochs_decay))
        self.update_lr(g_lr, d_lr)
        # print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

  #=======================================================================================#
  #=======================================================================================#

  def save_fake_output(self, real_x, save_path):
    real_x = self.to_var(real_x, volatile=True)

    # target_c_list = []
    target_c= torch.from_numpy(np.zeros((real_x.size(0), self.config.c_dim), dtype=np.float32))
    target_c_list = [self.to_var(target_c, volatile=True)]
    for j in range(self.c_dim):
      target_c[:,j]=1       
      target_c_list.append(self.to_var(target_c, volatile=True))
      # target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
      # target_c_list.append(self.to_var(target_c, volatile=True))

    # Start translations
    fake_image_list = [real_x]

    for target_c in target_c_list:       
      fake_x = self.G(real_x, target_c)
      fake_image_list.append(fake_x)
    fake_images = torch.cat(fake_image_list, dim=3)
    save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)

  #=======================================================================================#
  #=======================================================================================#

  def test(self):
    """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
    # Load trained parameters
    from data_loader import get_loader
    if self.config.pretrained_model=='':
      last_file = sorted(glob.glob(os.path.join(self.config.model_save_path,  '*_D.pth')))[-1]
      last_name = '_'.join(last_file.split('/')[-1].split('_')[:2])
    else:
      last_name = self.config.pretrained_model

    G_path = os.path.join(self.config.model_save_path, '{}_G.pth'.format(last_name))
    self.G.load_state_dict(torch.load(G_path))
    self.G.eval()
    # ipdb.set_trace()

    data_loader_val = get_loader(self.config.metadata_path, self.config.image_size,
                 self.config.image_size, self.config.batch_size, 'MultiLabelAU', 'val')   

    for i, (real_x, org_c, files) in enumerate(data_loader_val):
      save_path = os.path.join(self.config.sample_path, '{}_fake_val_{}.jpg'.format(last_name, i+1))
      self.save_fake_output(real_x, save_path)
      print('Translated test images and saved into "{}"..!'.format(save_path))
      if i==3: break

  #=======================================================================================#
  #=======================================================================================#

  def val_cls(self, init=False, load=False):
    """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
    # Load trained parameters
    if init:

      from data_loader import get_loader
      # ipdb.set_trace()
      self.data_loader_val = get_loader(self.config.metadata_path, self.config.image_size,
                   self.config.image_size, self.config.batch_size, 'MultiLabelAU', 'val', shuffling=True)

      txt_path = os.path.join(self.config.model_save_path, 'init_val.txt')
    else:
      last_file = sorted(glob.glob(os.path.join(self.config.model_save_path,  '*_D.pth')))[-1]
      last_name = '_'.join(last_file.split('/')[-1].split('_')[:2])
      txt_path = os.path.join(self.config.model_save_path, '{}_{}_val.txt'.format(last_name,'{}'))
      try:
        output_txt  = sorted(glob.glob(txt_path.format('*')))[-1]
        number_file = len(glob.glob(output_txt))
      except:
        number_file = 0
      txt_path = txt_path.format(str(number_file).zfill(2)) 
    
    if load:
      D_path = os.path.join(self.config.model_save_path, '{}_D.pth'.format(last_name))
      self.D.load_state_dict(torch.load(D_path))

    self.D.eval()

    self.config.f=open(txt_path, 'a')   
    self.config.thresh = np.linspace(0.01,0.99,200).astype(np.float32)
    # ipdb.set_trace()
    # F1_real, F1_max, max_thresh_train  = self.F1_TEST(data_loader_train, mode = 'TRAIN')
    # _ = self.F1_TEST(data_loader_test, thresh = max_thresh_train)
    f1,_,_, loss, f1_1 = F1_TEST(self, self.data_loader_val, thresh = [0.5]*12, mode='VAL', verbose=load)
    # f1,_,_ = self.F1_TEST(self.data_loader_val, thresh = [0.5]*12, mode='VAL', verbose=load)
    self.f.close()
    return f1, loss, f1_1

  #=======================================================================================#
  #=======================================================================================#

  def test_cls(self):
    """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
    # Load trained parameters
    from data_loader import get_loader
    if self.config.pretrained_model=='':
      last_file = sorted(glob.glob(os.path.join(self.config.model_save_path,  '*_D.pth')))[-1]
      last_name = '_'.join(last_file.split('/')[-1].split('_')[:2])
    else:
      last_name = self.config.pretrained_model

    G_path = os.path.join(self.config.model_save_path, '{}_G.pth'.format(last_name))
    D_path = os.path.join(self.config.model_save_path, '{}_D.pth'.format(last_name))
    txt_path = os.path.join(self.config.model_save_path, '{}_{}_test.txt'.format(last_name,'{}'))
    show_fake = os.path.join(self.config.sample_path, '{}_fake_{}_{}.jpg'.format(last_name,'{}', '{}'))
    self.config.pkl_data = os.path.join(self.config.model_save_path, '{}_{}.pkl'.format(last_name, '{}'))
    self.config.lstm_path = os.path.join(self.config.model_save_path, '{}_lstm'.format(last_name))
    if not os.path.isdir(self.config.lstm_path): os.makedirs(self.config.lstm_path)
    print(" [!!] {} model loaded...".format(D_path))
    # ipdb.set_trace()
    self.G.load_state_dict(torch.load(G_path))
    self.D.load_state_dict(torch.load(D_path))
    self.G.eval()
    self.D.eval()
    # ipdb.set_trace()
    if self.config.dataset == 'MultiLabelAU' and not self.config.GOOGLE:
      data_loader_val = get_loader(self.config.metadata_path, self.config.image_size,
                self.config.image_size, self.config.batch_size, 'MultiLabelAU', 'val', shuffling=True)
      data_loader_test = get_loader(self.config.metadata_path, self.config.image_size,
                self.config.image_size, self.config.batch_size, 'MultiLabelAU', 'test', shuffling=True)

    elif self.config.GOOGLE: 
      # data_loader_google = get_loader('', self.config.image_size, self.config.image_size, \
      #                 self.config.batch_size, 'Google',  mode=self.config.mode_data, shuffling=True)
      data_loader_google = get_loader(self.config.metadata_path, self.config.image_size, \
                self.config.image_size, self.config.batch_size, 'MultiLabelAU', 'train', shuffling=True)     

    if not hasattr(self.config, 'output_txt'):
      # ipdb.set_trace()
      self.config.output_txt = txt_path
      try:
        self.config.output_txt  = sorted(glob.glob(self.config.output_txt.format('*')))[-1]
        number_file = len(glob.glob(self.config.output_txt))
      except:
        number_file = 0
      self.config.output_txt = self.config.output_txt.format(str(number_file).zfill(2)) 
    
    self.config.f=open(self.config.output_txt, 'a')  
    self.config.thresh = np.linspace(0.01,0.99,200).astype(np.float32)
    # ipdb.set_trace()
    # F1_real, F1_max, max_thresh_train  = self.F1_TEST(data_loader_train, mode = 'TRAIN')
    # _ = self.F1_TEST(data_loader_test, thresh = max_thresh_train)
    if self.config.GOOGLE: 
      _ = F1_TEST(self, data_loader_google)
    else: 
      F1_real, F1_max, max_thresh_val, _, _  = F1_TEST(self, data_loader_val, mode = 'VAL')
      _ = F1_TEST(self, data_loader_test, thresh = max_thresh_val, show_fake = show_fake) 
      # _ = self.F1_TEST(data_loader_test)
    self.config.f.close()

  #=======================================================================================#
  #=======================================================================================#

  def save_lstm(self, data, files):
    assert data.shape[0]==len(files)
    for i in range(len(files)):
      name = os.path.join(self.config.lstm_path, '/'.join(files[i].split('/')[-6:]))
      name = name.replace('jpg', 'npy')
      folder = os.path.dirname(name)
      if not os.path.isdir(folder): os.makedirs(folder)
      np.save(name, data[i])

