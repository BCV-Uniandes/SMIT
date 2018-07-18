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
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import ipdb
import config as cfg
import glob
import pickle
from utils import f1_score, f1_score_max, F1_TEST
from tqdm import tqdm
from datetime import datetime
import pytz

class Solver(object):

  def __init__(self, data_loader, config):
    # Data loader
    self.data_loader = data_loader

    self.config = config

    self.AUs_common = list(set(self.config.AUs['BP4D']).intersection(self.config.AUs['EMOTIONNET']))
    self.AUs = [data_loader.dataset.AUs for data_loader in self.data_loader]
    self.index_aus = [] 
    for AUs_dataset in self.AUs:
      self.index_aus.append([])
      # ipdb.set_trace()
      for au in self.AUs_common:
        self.index_aus[-1].append(AUs_dataset.index(au))

    self.config.lr = self.config.c_lr

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
    
    y1,y2 = self.C(self.to_var(torch.ones(1,3,self.config.image_size,self.config.image_size)))
    g=make_dot(y1, params=dict(self.C.named_parameters()))

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
      # ipdb.set_trace()
      self.G = Generator(self.config.image_size, self.config.g_conv_dim, self.config.c_dim, self.config.g_repeat_num, SAGAN='SAGAN' in self.config.GAN_options, debug=False)

      print("Loading Generator from {}".format(self.config.Generator_path))
      self.G.load_state_dict(torch.load(self.config.Generator_path))
      self.G.eval()

    if 'DENSENET' in self.config.CLS_options:
      # from models.densenet import densenet121 as Classifier
      model = self.config.model_CLS.lower()#'densenet201'
      exec('from models.densenet import {} as Classifier'.format(model))
      # self.config.model_save_path = self.config.model_save_path.replace('DENSENET', model.upper())
      # self.config.log_path = self.config.log_path.replace('DENSENET', model.upper())

      # self.C = Classifier(num_classes = len(self.AUs_common), pretrained=True) 
      self.C = Classifier(num_classes = len(self.AUs_common)) 
      print("Building {} with {} outputs".format(self.config.model_CLS.upper(), len(self.AUs_common)))

    elif 'RESNET' in self.config.CLS_options:
      from models.resnet import resnet50 as Classifier
      # self.C = Classifier(num_classes = len(self.AUs_common), pretrained=True)
      self.C = Classifier(num_classes = len(self.AUs_common))
      print("Building RESNET with {} outputs".format(len(self.AUs_common)))

    else:
      from model import Discriminator as Classifier
      self.C = Classifier(self.config.mage_size, self.config.d_conv_dim, len(self.AUs_common), self.config.d_repeat_num) 
      print("Building DISCRIMINATOR_GAN with {} outputs".format(len(self.AUs_common)))

    # Optimizers
    self.optimizer = torch.optim.Adam(self.C.parameters(), self.config.lr, [self.config.beta1, self.config.beta2])

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
  def update_lr(self, lr):
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = lr

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
  def to_tensor(self, x):
    # ipdb.set_trace()
    return torch.FloatTensor(x).view(1,-1,1,1)

  #=======================================================================================#
  #=======================================================================================#
  def denorm_CLS(self, x):   
    # ipdb.set_trace()
    out = (x * self.to_tensor(self.config.std)) + self.to_tensor(self.config.mean)
    return out.clamp_(0, 1)

  #=======================================================================================#
  #=======================================================================================#
  def norm_CLS(self, x):   
    out = (x - self.to_tensor(self.config.mean)) / self.to_tensor(self.config.std)
    return out

  #=======================================================================================#
  #=======================================================================================#
  def denorm_GAN(self, x):   
    # ipdb.set_trace()
    out = (x + 1) / 2
    return out.clamp_(0, 1)

  #=======================================================================================#
  #=======================================================================================#
  def norm_GAN(self, x):   
    # ipdb.set_trace()
    out = (self.denorm_CLS(x) - 0.5) * 2
    return out.clamp_(-1, 1)

  #=======================================================================================#
  #=======================================================================================#
  def threshold(self, x):
    x = x.clone()
    x = (x >= 0.5).float()
    return x

  #=======================================================================================#
  #=======================================================================================#
  def soft_labels(self, x):
    x = (x.clone()*0.8)+0.1 # [0.0, 1.0] -> [0.1, 0.9]
    return x

  #=======================================================================================#
  #=======================================================================================#
  @property
  def TimeNow(self):
    return str(datetime.now(pytz.timezone('Europe/Amsterdam'))).split('.')[0]

  #=======================================================================================#
  #=======================================================================================#
  def fake_label(self, batch_size):
    num_labels_on = 3
    # zeros = torch.zeros(batch_size, len(self.AUs_common))
    # ipdb.set_trace()
    zeros = torch.zeros(batch_size, len(self.config.AUs[self.config.dataset_fake.upper()]))
    # np.random.seed(111)
    for bs in range(zeros.shape[0]):
      for i in np.random.randint(0,zeros.shape[1],np.random.randint(0,num_labels_on+1,1)):
        zeros[bs, i] = 1
    return zeros

  #=======================================================================================#
  #=======================================================================================#
  def reduce_label(self, real_label, index_aus):
    # ipdb.set_trace()
    real_label_temp = real_label[:,:len(index_aus)]
    for idx, au in enumerate(index_aus):
      real_label_temp[:,idx] = real_label[:,au]
    # ipdb.set_trace()
    real_label = real_label_temp
    return real_label

  #=======================================================================================#
  #=======================================================================================#
  def imgShow(self, img):
    try:save_image(img.cpu(), 'dummy.jpg')
    except: save_image(img.data.cpu(), 'dummy.jpg')
    plt.imshow(imageio.imread('dummy.jpg'))
    plt.show()
    #os.system('eog dummy.jpg')  
    os.remove('dummy.jpg')


  #=======================================================================================#
  #=======================================================================================#
  def show_ONEimg(self, real_img, fake_label):          
    import matplotlib.pyplot as plt
    # ipdb.set_trace()
    print("")
    string = [map(int,i) for i in (self.config.AUs['EMOTIONNET']*fake_label.cpu().numpy()).tolist()]
    for idx, s in enumerate(string): print(idx+1, sorted(list(set(s))))
    real_img = self.to_var(real_img, volatile=True)
    fake_label = self.to_var(fake_label, volatile=True)
    fake_image = self.G(real_img, fake_label)
    fake_image = torch.cat((real_img, fake_image), dim=3)    
    # shape0 = min(8, fake_images.data.cpu().shape[0])
    # ipdb.set_trace()
    save_image(self.denorm_GAN(fake_image.data.cpu()), 'temp.jpg',nrow=1, padding=0)
    # print("Fake Label: \n"+str(fake_label.data.cpu().numpy()))
    # os.system('eog _temp.jpg')    
    # os.remove('_temp.jpg')
    

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
        if (i+1) %self.config.num_epochs_decay==0:
          lr = (lr / 10.)
          self.update_lr(lr)
          self.PRINT ('Decay learning rate to lr: {}.'.format(lr))     
    else:
      start = 0

    last_model_step = len(self.data_loader)

    # Start training
    self.PRINT("Log path: "+self.config.log_path)
    Log = "---> {} | batch size: {}, fold: {}, img: {}, GPU: {}, !{}, [{}]\n-> GAN_options:".format(\
        self.TimeNow, self.config.batch_size, self.config.fold, self.config.image_size, \
        self.config.GPU, self.config.mode_data, self.config.PLACE) 

    for item in self.config.GAN_options:
      Log += ' [*{}]'.format(item.upper())
    Log += ' [*{}]'.format(self.config.dataset_fake)
    Log +=' | -> CLS_options:'
    for item in self.config.CLS_options:
      Log += ' [*{}]'.format(item.upper())
    Log += ' [*{}]'.format(self.config.dataset_real)

    self.PRINT(Log)
    loss_cum = {}
    start_time = time.time()
    flag_init=True

    f1_val_prev = 0
    non_decreasing = 0

    for e in range(start, self.config.num_epochs):
      E = str(e+1).zfill(3)
      self.C.train()
      
      if flag_init:
        f1_val, loss, f1_1 = self.val(init=True)  
        F1_MEAN = np.mean(f1_val).mean()
        if self.config.pretrained_model: f1_val_prev=F1_MEAN 
        log = '[F1_VAL: %0.3f (F1_1: %0.3f) LOSS_VAL: %0.3f]'%(F1_MEAN, np.mean(f1_1), np.mean(loss))
        self.PRINT(log)
        flag_init = False

      desc_bar = 'Epoch: %d/%d'%(e,self.config.num_epochs)
      # ipdb.set_trace()
      progress_bar = tqdm(enumerate(self.data_loader[0]), \
          total=len(self.data_loader[0]), desc=desc_bar, ncols=10)
      for i, (real_x0, real_label0, files0) in progress_bar:      
        # ipdb.set_trace()
        if self.data_loader[0].dataset.name==self.config.dataset_fake:
          # label_gan = self.fake_label(batch_size=real_label0.size(0))
          label_gan = real_label0
          # fl_au = label_gan*torch.FloatTensor(self.config.AUs['EMOTIONNET'])
          real_label0 = self.reduce_label(label_gan, self.index_aus[0])
        else:         
          real_label0 = self.reduce_label(real_label0, self.index_aus[0])

        if len(self.data_loader)>1:
          try:
            real_x1, real_label1, files1 = data_iter.next()
          except:
            data_iter = iter(self.data_loader[1])
            real_x1, real_label1, files1 = data_iter.next()
          if self.data_loader[1].dataset.name==self.config.dataset_fake:
            label_gan = self.fake_label(batch_size=real_label1.size(0))
            real_label1 = self.reduce_label(label_gan, self.index_aus[1])
          else:         
            real_label1 = self.reduce_label(real_label1, self.index_aus[1])
          real_x = torch.cat((real_x0, real_x1), dim=0)
          real_label = torch.cat((real_label1, real_label2), dim=0)
        else:
          real_x = real_x0
          real_label = real_label0

        if 'JUST_REAL' in self.config.CLS_options:
          real_label = real_label0
        else:
          # fake_label = self.fake_label(batch_size=real_label0.size(0))
          if not self.config.dataset_fake=='EmotionNet':
            # ipdb.set_trace()
            # real_x_gan = self.norm_GAN(real_x)
            real_x_gan = real_x.clone()
            # self.show_ONEimg(real_x_gan, label_gan)
            # ipdb.set_trace()
            real_x_gan = self.to_var(real_x_gan, volatile=True) 
            label_gan = self.to_var(label_gan, volatile=True)
            fake_x = self.G(real_x_gan, label_gan)
            # real_x = self.denorm_GAN(fake_x.data.cpu())
            real_x = fake_x.data.cpu()
            real_label = self.reduce_label(label_gan.data, self.index_aus[0])

            # ipdb.set_trace()
          
        real_x = self.to_var(real_x)
        real_label = self.to_var(real_label) 

        out_cls = self.C(real_x)
        # ipdb.set_trace()
        if 'SOFT_LABELS' in self.config.CLS_options: real_label = self.soft_labels(real_label)
        loss_cls = F.binary_cross_entropy_with_logits(
          out_cls, real_label, size_average=False) / real_x.size(0)

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
        if (i+1) % self.config.log_step == 0 or (i+1)==last_model_step:
          if self.config.use_tensorboard:
            # print("Log path: "+self.log_path)
            for tag, value in loss.items():
              self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)       

      #F1 val
      f1_val, _, _ = self.val()
      F1_MEAN = np.array(f1_val).mean()
      if self.config.use_tensorboard:
        for idx, au in enumerate(cfg.AUs):
          self.logger.scalar_summary('F1_val_'+str(au).zfill(2), f1_val[idx], e * iters_per_epoch + i + 1)      
        self.logger.scalar_summary('F1_val_mean', F1_MEAN, e * iters_per_epoch + i + 1)      

        for tag, value in loss_cum.items():
          self.logger.scalar_summary(tag, np.array(value).mean(), e * iters_per_epoch + i + 1)   
                 
      #Stats per epoch
      log = '!%s | [F1_VAL: %0.3f] | Train'%(self.TimeNow, F1_MEAN)
      for tag, value in loss_cum.items():
        log += ", {}: {:.4f}".format(tag, np.array(value).mean())    
      self.PRINT(log)

      # if loss_val<loss_val_prev:
      if F1_MEAN>f1_val_prev:
        torch.save(self.C.state_dict(), os.path.join(self.config.model_save_path, '{}_{}.pth'.format(E, i+1)))   
        # os.system('rm -vf {}'.format(os.path.join(self.config.model_save_path, '{}_{}.pth'.format(str(int(E)-1).zfill(2), i+1))))
        # loss_val_prev = loss_val
        self.PRINT("! Saving model")
        f1_val_prev = F1_MEAN
        non_decreasing = 0

      else:
        non_decreasing+=1
        if non_decreasing == self.config.stop_training:
          print("During {} epochs LOSS VAL was not decreasing.".format(self.config.stop_training))
          return    

      # Decay learning rate
      if (e+1) % self.config.num_epochs_decay==0:
        lr = (lr / 10.0)
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
                   self.config.image_size, self.config.batch_size, dataset=[self.config.dataset_real],
                   mode='val', num_workers=self.config.num_workers, shuffling=False, AU=self.config.AUs)[0]

    if init:
      txt_path = os.path.join(self.config.model_save_path, 'val.txt')
    elif load:
      last_file = sorted(glob.glob(os.path.join(self.config.model_save_path,  '*.pth')))[-1]
      last_name = '_'.join(last_file.split('/')[-1].split('_')[:2])
      txt_path = os.path.join(self.config.model_save_path, '{}_{}_val.txt'.format(last_name,'{}'))
      try:
        output_txt  = sorted(glob.glob(txt_path.format('*')))[-1]
        number_file = len(glob.glob(output_txt))
      except:
        number_file = 0

      txt_path = txt_path.format(str(number_file).zfill(2)) 
      C_path = os.path.join(self.config.model_save_path, '{}.pth'.format(last_name))
      self.C.load_state_dict(torch.load(C_path))

    else:
      txt_path = os.path.join(self.config.model_save_path, 'val.txt')

    self.C.eval()

    self.config.f=open(txt_path, 'a')   
    self.config.thresh = np.linspace(0.01,0.99,200).astype(np.float32)
    # ipdb.set_trace()
    # F1_real, F1_max, max_thresh_train  = self.F1_TEST(data_loader_train, mode = 'TRAIN')
    # _ = self.F1_TEST(data_loader_test, thresh = max_thresh_train)
    f1,_,_,loss,f1_1 = F1_TEST(self, data_loader_val, thresh = [0.5]*len(self.index_aus[0]), index_au=self.index_aus[0], mode='VAL', verbose=load)
    self.config.f.close()
    return f1, loss, f1_1


  #=======================================================================================#
  #=======================================================================================#
  #                                           TEST                                        #
  #=======================================================================================#
  #=======================================================================================#  
  def test(self, dataset='', load=False):
    if dataset=='': dataset='BP4D'
    from data_loader import get_loader
    # ipdb.set_trace()
    if self.config.pretrained_model in ['',None] or load:
      last_file = sorted(glob.glob(os.path.join(self.config.model_save_path,  '*.pth')))[-1]
      last_name = os.path.basename(last_file).split('.')[0]
    else:
      last_name = self.config.pretrained_model

    C_path = os.path.join(self.config.model_save_path, '{}.pth'.format(last_name))
    txt_path = os.path.join(self.config.model_save_path, '{}_{}.txt'.format(last_name,'{}'))
    self.pkl_data = os.path.join(self.config.model_save_path, '{}_{}.pkl'.format(last_name, '{}'))
    print(" [!!] {} model loaded...".format(C_path))
    self.C.load_state_dict(torch.load(C_path))
    self.C.eval()
    data_loader_val = get_loader(self.config.metadata_path, self.config.image_size,
                 self.config.image_size, self.config.batch_size, dataset=[dataset],
                 mode='val', num_workers=self.config.num_workers, AU=self.config.AUs)[0]
    data_loader_test = get_loader(self.config.metadata_path, self.config.image_size,
                 self.config.image_size, self.config.batch_size, dataset=[dataset],
                 mode='test', num_workers=self.config.num_workers, AU=self.config.AUs)[0]

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
    F1_real, F1_max, max_thresh_val, _, _  = F1_TEST(self, data_loader_val, index_au=self.index_aus[0], mode = 'VAL')
    _ = F1_TEST(self, data_loader_test, index_au=self.index_aus[0], thresh = max_thresh_val)
   
    self.config.f.close()