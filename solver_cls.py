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

# CUDA_VISIBLE_DEVICES=3 ipython main.py -- --c_dim=12 --num_epochs 4  --dataset MultiLabelAU --batch_size 32 --image_size 256 --d_repeat_num 7 --g_repeat_num 7 --multi_binary --au 1 --au_model aunet
# CUDA_VISIBLE_DEVICES=0 ipython main.py -- --c_dim=12 --num_epochs 10  --dataset MultiLabelAU --batch_size 8 --image_size 256 --d_repeat_num 7 --g_repeat_num 7 --fold 1
# CUDA_VISIBLE_DEVICES=1 ipython main.py -- --c_dim=12 --num_epochs 10  --dataset MultiLabelAU --batch_size 8 --image_size 256 --d_repeat_num 7 --g_repeat_num 7 --fold 2


class Solver(object):

  def __init__(self, MultiLabelAU_loader, config, CelebA=None):
    # Data loader
    self.MultiLabelAU_loader = MultiLabelAU_loader
    self.MultiLabelAU_FULL_loader = config.MultiLabelAU_FULL_loader
    self.CelebA_loader = CelebA

    # Model hyper-parameters
    self.c_dim = config.c_dim
    # self.c2_dim = config.c2_dim
    self.image_size = config.image_size
    self.g_conv_dim = config.g_conv_dim
    self.d_conv_dim = config.d_conv_dim
    self.g_repeat_num = config.g_repeat_num
    self.d_repeat_num = config.d_repeat_num
    self.d_train_repeat = config.d_train_repeat

    # Hyper-parameteres
    self.lambda_cls = config.lambda_cls
    self.lambda_rec = config.lambda_rec
    self.lambda_gp = config.lambda_gp
    self.g_lr = config.g_lr
    self.d_lr = config.d_lr
    self.beta1 = config.beta1
    self.beta2 = config.beta2

    # Training settings
    self.dataset = config.dataset
    self.num_epochs = config.num_epochs
    self.num_epochs_decay = config.num_epochs_decay
    # self.num_iters = config.num_iters
    # self.num_iters_decay = config.num_iters_decay
    self.batch_size = config.batch_size
    self.use_tensorboard = config.use_tensorboard
    self.pretrained_model = config.pretrained_model

    self.FOCAL_LOSS = config.FOCAL_LOSS
    self.JUST_REAL = config.JUST_REAL
    self.FAKE_CLS = config.FAKE_CLS
    self.DENSENET = config.DENSENET
    self.COLOR_JITTER = config.COLOR_JITTER  
    self.GOOGLE = config.GOOGLE   

    #Normalization
    self.mean = config.mean
    self.MEAN = config.MEAN #string
    self.std = config.std
    self.D_norm = config.D_norm
    self.G_norm = config.G_norm

    if self.MEAN in ['data_full','data_image']:
      self.tanh=False
      if self.MEAN=='data_full':
        mean_img = 'data/face_{}_mean.npy'.format(config.mode_data)
        std_img = 'data/face_{}_std.npy'.format(config.mode_data)
        print("Mean and Std from data: %s and %s"%(mean_img,std_img))
        self.mean = np.load(mean_img).astype(np.float64).transpose(2,0,1)/255.
        self.std = np.load(std_img).astype(np.float64).transpose(2,0,1)/255.
        self.mean = torch.FloatTensor(self.mean.mean(axis=(1,2)))
        self.std = torch.FloatTensor(self.std.std(axis=(1,2)))
    else:
      self.tanh=True

    #Training Binary Classifier Settings
    # self.au_model = config.au_model
    # self.au = config.au
    # self.multi_binary = config.multi_binary
    self.pretrained_model_generator = config.pretrained_model_generator
    self.pretrained_model_discriminator = config.pretrained_model_discriminator

    # Test settings
    self.test_model = config.test_model
    self.metadata_path = config.metadata_path

    # Path
    self.log_path = config.log_path
    self.sample_path = config.sample_path
    self.model_save_path = config.model_save_path
    # self.result_path = config.result_path
    self.fold = config.fold
    self.mode_data = config.mode_data

    # Step size
    self.log_step = config.log_step
    self.sample_step = config.sample_step
    self.model_save_step = config.model_save_step

    self.GPU = config.GPU    

    # Build tensorboard if use
    self.build_model()
    if self.use_tensorboard:
      self.build_tensorboard()

    # Start with trained model
    if self.pretrained_model:
      self.load_pretrained_model()

  def build_model(self):
    # Define a generator and a discriminator

    from model import Generator
    # if not self.JUST_REAL: self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)

    if self.CelebA_loader is not None:
      self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)
    else:
      if not self.JUST_REAL: self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num, tanh=self.tanh)

    if self.DENSENET:
      from models.densenet import Generator, densenet121 as Discriminator
      self.D = Discriminator(num_classes = self.c_dim) 
    else:
      from model import Discriminator
      self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 


    # Optimizers
    self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

    # Print networks
    if not self.JUST_REAL:self.print_network(self.G, 'G')
    self.print_network(self.D, 'D')

    if torch.cuda.is_available():
      if not self.JUST_REAL: self.G.cuda()
      self.D.cuda()

  def print_network(self, model, name):
    num_params = 0
    for p in model.parameters():
      num_params += p.numel()
    # print(name)
    # print(model)
    # print("The number of parameters: {}".format(num_params))

  def load_pretrained_model(self):
    model = os.path.join(
      self.model_save_path, '{}_D.pth'.format(self.pretrained_model))
    self.D.load_state_dict(torch.load(model))
    print('loaded CLS trained model: {}!'.format(model))

  def build_tensorboard(self):
    from logger import Logger
    self.logger = Logger(self.log_path)

  def update_lr(self, d_lr):
    for param_group in self.d_optimizer.param_groups:
      param_group['lr'] = d_lr

  def reset_grad(self):
    self.d_optimizer.zero_grad()

  def to_cuda(self, x):
    if torch.cuda.is_available():
      x = x.cuda()
    return x

  def to_var(self, x, volatile=False, requires_grad=False):
    return Variable(self.to_cuda(x), volatile=volatile, requires_grad=requires_grad)

  def denorm(self, x, img_org=None):

    if self.MEAN=='data_full':
      # ipdb.set_trace()
      # mean = torch.unsqueeze(self.mean,0)
      # std = torch.unsqueeze(self.std,0)
      # mean_list = [mean for i in range(int(x.size(3)/x.size(2)))]
      # std_list = [std for i in range(int(x.size(3)/x.size(2)))]
      # mean_list = torch.cat(mean_list, dim=3)
      # std_list = torch.cat(std_list, dim=3)

      # out = (x.cpu()*std_list)+mean_list
      out = (x*self.std.view(1,-1,1,1))+self.mean.view(1,-1,1,1)
      return out.clamp_(0, 1)

    elif self.MEAN=='data_image':
      # ipdb.set_trace()
      if img_org is None: raise ValueError('Mean <data_image> require image to denorm')
      # if img_org is None: img_org = self.real_x.data
      # mean = (img_org.data.cpu().mean(dim=3).mean(dim=2)).view(img_org.size(0), img_org.size(1),1,1)
      # std = (img_org.data.cpu().std(dim=3).std(dim=2)).view(img_org.size(0), img_org.size(1),1,1)

      mean = torch.from_numpy(img_org.cpu().numpy().mean(axis=(3,2)).reshape(img_org.size(0),img_org.size(1),1,1))
      std = torch.from_numpy(img_org.cpu().numpy().std(axis=(3,2)).reshape(img_org.size(0),img_org.size(1),1,1))

      out = (x*std)+mean
      return out.clamp_(0, 1)   

    else:      
      out = (x + 1) / 2
      return out.clamp_(0, 1)

  #MEAN data_image data_full
  def norm(self, x):
    if self.MEAN=='data_image':
      # ipdb.set_trace()
      mean = torch.from_numpy(x.cpu().numpy().mean(axis=(3,2)).reshape(x.size(0),x.size(1),1,1))
      std  = torch.from_numpy(x.cpu().numpy().std(axis=(3,2)).reshape(x.size(0),x.size(1),1,1))

      mean = self.to_var(mean, volatile=True)
      std = self.to_var(std, volatile=True)

      out = (x - mean) / std

    elif self.MEAN=='data_full':
      # mean_img = 'data/face_{}_mean.npy'.format(self.mode_data)
      # std_img = 'data/face_{}_std.npy'.format(self.mode_data)
      # print("Mean and Std from data: %s and %s"%(mean_img,std_img))
      # mean = np.load(mean_img).astype(np.float64).transpose(2,0,1)/255.
      # std = np.load(std_img).astype(np.float64).transpose(2,0,1)/255.   
      # ipdb.set_trace()
      out = (x - self.mean.view(1,-1,1,1)) / self.std.view(1,-1,1,1)

    return out


  def threshold(self, x):
    x = x.clone()
    x = (x >= 0.5).float()
    return x

  def color_up(self, labels):
    where_pos = lambda x,y: np.where(x.data.cpu().numpy().flatten()==y)[0]
    color_up = 0.2
    rgb = np.zeros((self.batch_size, 3, 224, 224)).astype(np.float32)
    green_r_pos = where_pos(labels,1)
    rgb[green_r_pos,1,:,:] += color_up
    red_r_pos = where_pos(labels,0)
    rgb[red_r_pos,0,:,:] += color_up   
    rgb = Variable(torch.FloatTensor(rgb))
    if torch.cuda.is_available(): rgb = rgb.cuda()
    return rgb

  def compute_accuracy(self, x, y, dataset):
    if dataset == 'CelebA' or dataset=='MultiLabelAU':
      x = F.sigmoid(x)
      predicted = self.threshold(x)
      correct = (predicted == y).float()
      accuracy = torch.mean(correct, dim=0) * 100.0
    else:
      _, predicted = torch.max(x, dim=1)
      correct = (predicted == y).float()
      accuracy = torch.mean(correct) * 100.0
    return accuracy

  def one_hot(self, labels, dim):
    """Convert label indices to one-hot vector"""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    # ipdb.set_trace()
    out[np.arange(batch_size), labels.long()] = 1
    return out

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

  def train(self):
    """Train StarGAN within a single dataset."""

    # Set dataloader
    if self.dataset == 'MultiLabelAU':
      self.data_loader = self.MultiLabelAU_loader      
    elif self.dataset == 'au01_fold0':
      self.data_loader = self.au_loader    

    # The number of iterations per epoch
    iters_per_epoch = len(self.data_loader)

    if not self.JUST_REAL:
    
      print(" [!] Loading Generator from "+self.pretrained_model_generator)
      self.G.load_state_dict(torch.load(self.pretrained_model_generator))

      try:
        if not self.pretrained_model:
          self.D.load_state_dict(torch.load(self.pretrained_model_discriminator))
          print(" [!] Loading Discriminator from "+self.pretrained_model_discriminator)
      except:
        pass

    # lr cache for decaying
    g_lr = self.g_lr
    d_lr = self.d_lr

    # Start with trained model if exists
    if self.pretrained_model:
      start = int(self.pretrained_model.split('_')[0])
      # Decay learning rate
      # for i in range(start):
      #   if (i+1) > (self.num_epochs - self.num_epochs_decay):
      #     # g_lr -= (self.g_lr / float(self.num_epochs_decay))
      #     d_lr -= (self.d_lr / float(self.num_epochs_decay))
      #     self.update_lr(d_lr)
      #     print ('Decay learning rate to d_lr: {}.'.format(d_lr))      
    else:
      start = 0

    # Start training
    fake_loss_cls = None
    real_loss_cls = None
    log_real = 'N/A'
    log_fake = 'N/A'
    fake_iters = 1
    fake_rate = 1
    real_rate = 1
    start_time = time.time()

    last_model_step = len(self.data_loader)

    for i, (real_x, real_label, files) in enumerate(self.data_loader): break
    real_x = self.to_var(real_x, volatile=True)
    real_label = self.to_var(real_label, volatile=True)
    fake_list = []
    fake_c=real_label.clone()*0
    fake_list.append(fake_c.clone())
    for i in range(12):
      fake_c[:,i]=1
      fake_list.append(fake_c.clone())

    # ipdb.set_trace()    
    # self.show_img(real_x, real_label, fake_list)
    # ipdb.set_trace()

    print("Log path: "+self.log_path)

    Log = " [!!CLS] bs:{}, fold:{}, img:{}, GPU:{}, !{}".format(self.batch_size, self.fold, self.image_size, self.GPU, self.mode_data) 

    if self.COLOR_JITTER: Log += ' [*COLOR_JITTER]'
    if self.FOCAL_LOSS: Log += ' [*FL]'
    if self.DENSENET: Log += ' [*DENSENET]'
    # if self.FAKE_CLS: Log += ' [*FAKE_CLS]'
    if self.JUST_REAL: Log += ' [*JUST_REAL]' 
    if self.MEAN!='0.5': Log += ' [*{}]'.format(self.MEAN)     
    loss_cum = {}
    flag_init=True    

    for e in range(start, self.num_epochs):
      E = str(e+1).zfill(2)
      self.D.train()
      self.G.train()

      if flag_init:
        f1 = self.val_cls(init=True)   
        log = '[F1_VAL: %0.3f]'%(np.array(f1).mean())
        print(log)
        flag_init = False

      for i, (real_x, real_label, files) in tqdm(enumerate(self.data_loader), \
          total=len(self.data_loader), desc='Epoch: %d/%d | %s'%(e,self.num_epochs, Log)):        

        real_x = self.to_var(real_x)
        real_label = self.to_var(real_label) 

        if self.D_norm:
          real_x_D = self.norm(real_x)
        else:
          real_x_D = real_x

        # if (i+e)%2==0 or self.JUST_REAL:
        if True:
          _, real_out_cls = self.D(real_x_D)
          
          real_loss_cls = F.binary_cross_entropy_with_logits(
            real_out_cls, real_label, size_average=False) / real_x_D.size(0)

          self.reset_grad()
          real_loss_cls.backward()
          self.d_optimizer.step()

        #if e>=1 and i%fake_rate:
        # if (i+e+1)%2==0 and not self.JUST_REAL:
        if True:
          # ================== Train C FAKE ================== #

          # # Compute loss with fake images
          for _ in range(fake_iters):

            # ============== LOADING MISSING BP4D ============== #
            try:
              real_x_full, _l, _f = next(data_full_loader)
            except:
              # ipdb.set_trace()
              data_full_loader = iter(self.MultiLabelAU_FULL_loader)
              real_x_full, _l, _f = next(data_full_loader)  

            real_x_full = self.to_var(real_x_full)  
            # ================================================== #

            # ipdb.set_trace()
            real_label_ = real_label.clone()
            luck = 0#np.random.randint(0,2)
            if luck==0 and real_label.size(0)==real_x_full.size(0):
              rand_idx = self.to_cuda(torch.randperm(real_x_full.size(0)))
              fake_label = real_label_[rand_idx]
            else:
              fake_label = self.to_var(torch.from_numpy(np.random.randint(0,2,[real_x_full.size(0),real_label_.size(1)]).astype(np.float32)))

            # ipdb.set_trace()

            fake_c = fake_label.clone()

            if self.G_norm:
              real_x_G = self.norm(real_x_full)
            else:
              real_x_G = real_x_full
  
            try:
              fake_x = self.G(real_x_G, fake_c)
            except: 
              ipdb.set_trace()

            if self.D_norm and self.G_norm:
              fake_x_D = fake_x #self.norm(fake_x.data.clone())
            elif self.D_norm and not self.G_norm:
              fake_x_D = self.norm(fake_x)
            elif not self.D_norm and self.G_norm:
              fake_x_D = self.to_cuda(self.denorm(fake_x.data.cpu(), real_x_full.data))
            else:
              fake_x_D = fake_x

            fake_x_D = Variable(fake_x_D.data)
            _, fake_out_cls = self.D(fake_x_D)

            fake_loss_cls = F.binary_cross_entropy_with_logits(
                 fake_out_cls, fake_label, size_average=False) / fake_x.size(0)

            # # Backward + Optimize
            self.reset_grad()
            fake_loss_cls.backward()
            self.d_optimizer.step()

        # Logging
        loss = {}
        if len(loss_cum.keys())==0: 
          loss_cum['real_loss_cls'] = []; loss_cum['fake_loss_cls'] = []        
        if real_loss_cls is not None: 
          loss['real_loss_cls'] = real_loss_cls.data[0]
          loss_cum['real_loss_cls'].append(real_loss_cls.data[0])
        if fake_loss_cls is not None: 
          loss['fake_loss_cls'] = fake_loss_cls.data[0]
          loss_cum['fake_loss_cls'].append(fake_loss_cls.data[0])

        # Print out log info
        if (i+1) % self.log_step == 0 or (i+1)==last_model_step:
          if self.use_tensorboard:
            # print("Log path: "+self.log_path)
            for tag, value in loss.items():
              self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

      torch.save(self.D.state_dict(),
        os.path.join(self.model_save_path, '{}_{}_D.pth'.format(E, i+1)))        

      #F1 val
      f1 = self.val_cls()
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
      if (e+1) > (self.num_epochs - self.num_epochs_decay):
        # g_lr -= (self.g_lr / float(self.num_epochs_decay))
        d_lr -= (self.d_lr / float(self.num_epochs_decay))
        self.update_lr(d_lr)
        print ('Decay learning rate to d_lr: {}.'.format(d_lr))


  def val_cls(self, init=False, load=False):
    """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
    # Load trained parameters
    from data_loader import get_loader
    # ipdb.set_trace()
    data_loader_val = get_loader(self.metadata_path, self.image_size,
                   self.image_size, self.batch_size, 'MultiLabelAU', 'val', shuffling=False)

    if init:
      txt_path = os.path.join(self.model_save_path, 'init_val.txt')
    else:
      last_file = sorted(glob.glob(os.path.join(self.model_save_path,  '*_D.pth')))[-1]
      last_name = '_'.join(last_file.split('/')[-1].split('_')[:2])
      txt_path = os.path.join(self.model_save_path, '{}_{}_val.txt'.format(last_name,'{}'))
      try:
        output_txt  = sorted(glob.glob(txt_path.format('*')))[-1]
        number_file = len(glob.glob(output_txt))
      except:
        number_file = 0
      txt_path = txt_path.format(str(number_file).zfill(2)) 
    
    if load:
      D_path = os.path.join(self.model_save_path, '{}_D.pth'.format(last_name))
      self.D.load_state_dict(torch.load(D_path))

    self.D.eval()

    self.f=open(txt_path, 'a')   
    self.thresh = np.linspace(0.01,0.99,200).astype(np.float32)
    # ipdb.set_trace()
    # F1_real, F1_max, max_thresh_train  = self.F1_TEST(data_loader_train, mode = 'TRAIN')
    # _ = self.F1_TEST(data_loader_test, thresh = max_thresh_train)
    f1,_,_ = F1_TEST(self, data_loader_val, thresh = [0.5]*12, mode='VAL', verbose=load)
    self.f.close()
    return f1

  def test_cls(self):
    """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
    # Load trained parameters
    from data_loader import get_loader
    if self.test_model=='':
      last_file = sorted(glob.glob(os.path.join(self.model_save_path,  '*_D.pth')))[-1]
      last_name = '_'.join(last_file.split('/')[-1].split('_')[:2])
    else:
      last_name = self.test_model

    # G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(last_name))
    D_path = os.path.join(self.model_save_path, '{}_D.pth'.format(last_name))
    txt_path = os.path.join(self.model_save_path, '{}_{}.txt'.format(last_name,'{}'))
    self.pkl_data = os.path.join(self.model_save_path, '{}_{}.pkl'.format(last_name, '{}'))
    self.lstm_path = os.path.join(self.model_save_path, '{}_lstm'.format(last_name))
    if not os.path.isdir(self.lstm_path): os.makedirs(self.lstm_path)
    print(" [!!] {} model loaded...".format(D_path))
    # self.G.load_state_dict(torch.load(G_path))
    self.D.load_state_dict(torch.load(D_path))
    # self.G.eval()
    self.D.eval()
    # ipdb.set_trace()
    if self.dataset == 'MultiLabelAU':
      data_loader_val = get_loader(self.metadata_path, self.image_size,
                   self.image_size, self.batch_size, 'MultiLabelAU', 'val', no_flipping = True)
      data_loader_test = get_loader(self.metadata_path, self.image_size,
                   self.image_size, self.batch_size, 'MultiLabelAU', 'test')
    elif dataset == 'au01_fold0':
      data_loader = self.au_loader  


    if not hasattr(self, 'output_txt'):
      # ipdb.set_trace()
      self.output_txt = txt_path
      try:
        self.output_txt  = sorted(glob.glob(self.output_txt.format('*')))[-1]
        number_file = len(glob.glob(self.output_txt))
      except:
        number_file = 0
      self.output_txt = self.output_txt.format(str(number_file).zfill(2)) 
    
    self.f=open(self.output_txt, 'a')   
    self.thresh = np.linspace(0.01,0.99,200).astype(np.float32)
    # ipdb.set_trace()
    F1_real, F1_max, max_thresh_val  = F1_TEST(self, data_loader_val, mode = 'VAL')
    _ = F1_TEST(self, data_loader_test, thresh = max_thresh_val)
   
    self.f.close()

  def save_lstm(self, data, files):
    assert data.shape[0]==len(files)
    for i in range(len(files)):
      name = os.path.join(self.lstm_path, '/'.join(files[i].split('/')[-6:]))
      name = name.replace('jpg', 'npy')
      folder = os.path.dirname(name)
      if not os.path.isdir(folder): os.makedirs(folder)
      np.save(name, data[i])
