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

from PIL import Image
import ipdb
import config as cfg
import glob
import pickle
from tqdm import tqdm
from utils import f1_score, f1_score_max, F1_TEST


class Solver(object):

  def __init__(self, data_loader, config):
    # Data loader
    self.data_loader = data_loader

    self.config = config
    self.config.lr = self.config.d_lr

    # Build tensorboard if use
    self.build_model()
    if self.config.use_tensorboard:
      self.build_tensorboard()

    # Start with trained model
    if self.config.pretrained_model:
      self.load_pretrained_model()

  #=======================================================================================#
  #=======================================================================================#
  def display_net(self, name='Claffifier'):
    #pip install git+https://github.com/szagoruyko/pytorchviz
    from graphviz import Digraph
    from torchviz import make_dot, make_dot_from_trace
    
    y1,y2 = self.D(self.to_var(torch.ones(1,3,self.config.image_size,self.config.image_size)))
    g=make_dot(y1, params=dict(self.D.named_parameters()))

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

    if not 'JUST_REAL' in self.config.CLS_options: 
      from model import Generator
      self.G = Generator(self.config.g_conv_dim, self.config.c_dim, self.config.g_repeat_num)
      print("Loading Generator from {}".format(self.config.Generator_path))
      self.G.load_state_dict(torch.load(self.config.Generator_path))
      self.G.eval()

    if 'DENSENET' in self.config.CLS_options:
      from model import Generator
      from models.densenet import densenet121 as Classifier
      self.C = Classifier(num_classes = self.config.c_dim) 
    else:
      from model import Discriminator as Classifier
      self.C = Classifier(self.config.mage_size, self.config.d_conv_dim, self.config.c_dim, self.config.d_repeat_num) 


    # Optimizers
    self.d_optimizer = torch.optim.Adam(self.C.parameters(), self.config.lr, [self.config.beta1, self.config.beta2])

    # Print networks
    if not 'JUST_REAL' in self.config.CLS_options: self.print_network(self.G, 'Generator')
    self.print_network(self.C, 'Classifier')

    if torch.cuda.is_available():
      if not 'JUST_REAL' in self.config.CLS_options: self.G.cuda()
      self.C.cuda()

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
    model = os.path.join(
      self.config.model_save_path, '{}.pth'.format(self.config.pretrained_model))
    self.C.load_state_dict(torch.load(model))
    print('loaded CLS trained model: {}!'.format(model))

  #=======================================================================================#
  #=======================================================================================#
  def build_tensorboard(self):
    from logger import Logger
    self.logger = Logger(self.config.log_path)

  #=======================================================================================#
  #=======================================================================================#
  def update_lr(self, c_lr):
    for param_group in self.d_optimizer.param_groups:
      param_group['lr'] = c_lr

  #=======================================================================================#
  #=======================================================================================#
  def reset_grad(self):
    self.optimizer.zero_grad()

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
  def denorm(self, x, img_org=None):   
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
  def show_img(self, img, real_label, fake_label):          
    import matplotlib.pyplot as plt
    fake_image_list=[img]

    for fl in fake_label:
      # ipdb.set_trace()
      fake_image_list.append(self.G(img, self.to_var(fl.data, volatile=True)))
    fake_images = torch.cat(fake_image_list, dim=3)    
    shape0 = min(8, fake_images.data.cpu().shape[0])
    # ipdb.set_trace()
    save_image(self.denorm(fake_images.data.cpu()[:shape0]), 'tmp_all.jpg',nrow=1, padding=0)
    print("Real Label: \n"+str(real_label.data.cpu()[:shape0].numpy()))
    for fl in fake_label:
      print("Fake Label: \n"+str(fl.data.cpu()[:shape0].numpy()))    
    os.system('eog tmp_all.jpg')    
    os.remove('tmp_all.jpg')

  #=======================================================================================#
  #=======================================================================================#
  def PRINT(self, str):  
    if not 'GOOGLE' in self.config.GAN_options:
      print >> self.config.log, str
      self.config.log.flush()
    print(str)

  #=======================================================================================#
  #=======================================================================================#
  #                                           TRAIN                                       #
  #=======================================================================================#
  #=======================================================================================#  
  def train(self):
    # lr cache for decaying
    lr = self.config.lr

    # Start with trained model if exists
    if self.config.pretrained_model:
      start = int(self.config.pretrained_model.split('_')[0])
      for i in range(start):
        # if (i+1) > (self.config.num_epochs - self.config.num_epochs_decay):
        if (i+1) %10==0:
          lr = (self.config.lr / 10.)
          self.update_lr(lr)
          self.PRINT ('Decay learning rate to lr: {}.'.format(lr))     
    else:
      start = 0

    last_model_step = len(self.data_loader)

    # Start training
    self.PRINT("Log path: "+self.config.log_path)
    Log = "---> batch size: {}, fold: {}, img: {}, GPU: {}, !{}\n-> GAN_options:".format(\
        self.config.batch_size, self.config.fold, self.config.image_size, \
        self.config.GPU, self.config.mode_data) 

    for item in self.config.GAN_options:
      Log += ' [*{}]'.format(item.upper())
    Log +='\n-> CLS_options:'
    for item in self.config.CLS_options:
      Log += ' [*{}]'.format(item.upper())

    self.PRINT(Log)
    loss_cum = {}
    start_time = time.time()
    flag_init=True


    for e in range(start, self.config.num_epochs):
      E = str(e+1).zfill(3)
      self.C.train()
      
      if flag_init:
        f1, loss, f1_1 = self.val(init=True)   
        log = '[F1_VAL: %0.3f (F1_1: %0.3f) LOSS_VAL: %0.3f]'%(np.mean(f1), np.mean(f1_1), np.mean(loss))
        self.PRINT(log)
        flag_init = False

      desc_bar = 'Epoch: %d/%d'%(e,self.config.num_epochs)
      progress_bar = tqdm(enumerate(self.data_loader), \
          total=len(self.data_loader), desc=desc_bar, ncols=10)
      for i, (real_x, real_label, files) in progress_bar:      

        real_x = self.to_var(real_x)
        real_label = self.to_var(real_label) 

        if luck==0 and real_label.size(0)==real_x_full.size(0):
          rand_idx = self.to_cuda(torch.randperm(real_x_full.size(0)))
          fake_label = real_label_[rand_idx]
        else:
          fake_label = self.to_var(torch.from_numpy(np.random.randint(0,2,[real_x_full.size(0),real_label_.size(1)]).astype(np.float32)))

        fake_x = self.G(real_x, fake_label)

        out_cls = self.C(real_x)
        
        loss_cls = F.binary_cross_entropy_with_logits(
          out_cls, fake_label, size_average=False) / real_x.size(0)

        self.reset_grad()
        loss_cls.backward()
        self.optimizer.step()

        # Logging
        loss = {}
        if len(loss_cum.keys())==0: 
          loss_cum['loss'] = []    
        if loss_cls is not None: 
          loss['loss'] = loss_cls.data[0]
          loss_cum['loss'].append(loss_cls.data[0])

        # Print out log info
        if (i+1) % self.log_step == 0 or (i+1)==last_model_step:
          if self.use_tensorboard:
            # print("Log path: "+self.log_path)
            for tag, value in loss.items():
              self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

      torch.save(self.D.state_dict(),
        os.path.join(self.model_save_path, '{}_{}.pth'.format(E, i+1)))        

      #F1 val
      f1 = self.val()
      if self.use_tensorboard:
        for idx, au in enumerate(cfg.AUs):
          self.logger.scalar_summary('F1_val_'+str(au).zfill(2), f1[idx], e * iters_per_epoch + i + 1)      
        self.logger.scalar_summary('F1_val_mean', np.array(f1).mean(), e * iters_per_epoch + i + 1)      

        for tag, value in loss_cum.items():
          self.logger.scalar_summary(tag, np.array(value).mean(), e * iters_per_epoch + i + 1)   
                 
      #Stats per epoch
      log = '![F1_VAL: %0.3f]'%(np.array(f1).mean())
      for tag, value in loss_cum.items():
        log += ", {}: {:.4f}".format(tag, np.array(value).mean())    

      print(log)


      # Decay learning rate
      if (e+1) % 10==0:
        lr = (self.lr / 10.0)
        self.update_lr(lr)
        print ('Decay learning rate to lr: {}.'.format(lr))

  #=======================================================================================#
  #=======================================================================================#
  #                                           VAL                                         #
  #=======================================================================================#
  #=======================================================================================#  
  def val(self, init=False, load=False):
    from data_loader import get_loader
    data_loader_val = get_loader(self.config.metadata_path, self.config.image_size,
                   self.config.image_size, self.config.batch_size, self.config.real_dataset, 'val', shuffling=False)

    if init:
      txt_path = os.path.join(self.config.model_save_path, 'init_val.txt')
    else:
      last_file = sorted(glob.glob(os.path.join(self.config.model_save_path,  '*.pth')))[-1]
      last_name = '_'.join(last_file.split('/')[-1].split('_')[:2])
      txt_path = os.path.join(self.config.model_save_path, '{}_{}_val.txt'.format(last_name,'{}'))
      try:
        output_txt  = sorted(glob.glob(txt_path.format('*')))[-1]
        number_file = len(glob.glob(output_txt))
      except:
        number_file = 0
      txt_path = txt_path.format(str(number_file).zfill(2)) 
    
    if load:
      C_path = os.path.join(self.config.model_save_path, '{}.pth'.format(last_name))
      self.C.load_state_dict(torch.load(C_path))

    self.C.eval()

    self.config.f=open(txt_path, 'a')   
    self.config.thresh = np.linspace(0.01,0.99,200).astype(np.float32)
    # ipdb.set_trace()
    # F1_real, F1_max, max_thresh_train  = self.F1_TEST(data_loader_train, mode = 'TRAIN')
    # _ = self.F1_TEST(data_loader_test, thresh = max_thresh_train)
    f1,_,_ = F1_TEST(self.config, data_loader_val, thresh = [0.5]*self.config.c_dim, mode='VAL', verbose=load)
    self.config.f.close()
    return f1


  #=======================================================================================#
  #=======================================================================================#
  #                                           TEST                                        #
  #=======================================================================================#
  #=======================================================================================#  
  def test(self):
    from data_loader import get_loader
    if self.config.pretrained_model=='':
      last_file = sorted(glob.glob(os.path.join(self.config.model_save_path,  '*.pth')))[-1]
      last_name = '_'.join(last_file.split('/')[-1].split('_')[:2])
    else:
      last_name = self.config.test_model

    C_path = os.path.join(self.config.model_save_path, '{}.pth'.format(last_name))
    txt_path = os.path.join(self.config.model_save_path, '{}_{}.txt'.format(last_name,'{}'))
    self.pkl_data = os.path.join(self.config.model_save_path, '{}_{}.pkl'.format(last_name, '{}'))
    print(" [!!] {} model loaded...".format(C_path))
    self.C.load_state_dict(torch.load(C_path))
    self.C.eval()
    data_loader_val = get_loader(self.config.metadata_path, self.config.image_size,
                 self.config.image_size, self.config.batch_size, self.config.real_dataset, 'val', no_flipping = True)
    data_loader_test = get_loader(self.config.metadata_path, self.config.image_size,
                 self.config.image_size, self.config.batch_size, self.config.real_dataset, 'test')

    if not hasattr(self, 'output_txt'):
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
    F1_real, F1_max, max_thresh_val  = F1_TEST(self.config, data_loader_val, mode = 'VAL')
    _ = F1_TEST(self.config, data_loader_test, thresh = max_thresh_val)
   
    self.f.close()