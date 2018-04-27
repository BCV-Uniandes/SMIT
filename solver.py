import tensorflow as tf
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

  def __init__(self, MultiLabelAU_loader, config, CelebA=None):
    # Data loader
    self.MultiLabelAU_loader = MultiLabelAU_loader
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
    self.PPT = config.PPT
    self.BLUR = config.BLUR
    self.GRAY = config.GRAY

    #Normalization
    self.mean = config.mean
    self.MEAN = config.MEAN #string
    self.std = config.std
    self.D_norm = config.D_norm
    self.G_norm = config.G_norm

    if self.MEAN in ['data_full','data_image']:
      self.tanh=False
      if self.MEAN=='data_full':
        mean_img = 'data/stats/channels_{}_mean.npy'.format(config.mode_data)
        std_img = 'data/stats/channels_{}_std.npy'.format(config.mode_data)
        print("Mean and Std from data: %s and %s"%(mean_img,std_img))
        # ipdb.set_trace()
        self.mean = np.load(mean_img).astype(np.float64)/255.#.transpose(2,0,1)/255.
        self.std = np.load(std_img).astype(np.float64)/255.#.transpose(2,0,1)/255.
        # self.mean = self.to_var(torch.FloatTensor(self.mean.mean(axis=(1,2))))
        # self.std = self.to_var(torch.FloatTensor(self.std.std(axis=(1,2))))

        self.mean = self.to_var(torch.FloatTensor(self.mean))
        self.std = self.to_var(torch.FloatTensor(self.std))

        self.MAX_DATASET = 20#10#self.get_max_dataset()
    else:
      self.tanh=True

    #Training Binary Classifier Settings
    # self.au_model = config.au_model
    # self.au = config.au
    # self.multi_binary = config.multi_binary
    # self.pretrained_model_generator = config.pretrained_model_generator

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

    self.blurrandom = 0

    # Build tensorboard if use
    self.build_model()
    if self.use_tensorboard:
      self.build_tensorboard()

    # Start with trained model
    if self.pretrained_model:
      self.load_pretrained_model()

  #=======================================================================================#
  #=======================================================================================#

  def build_model(self):
    # Define a generator and a discriminator
    if self.DENSENET:
      from models.densenet import Generator, densenet121 as Discriminator
      self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
      self.D = Discriminator(num_classes = self.c_dim) 
    else:
      from model import Generator, Discriminator
      if self.CelebA_loader is not None:
        self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num) 
      else:
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num, tanh=self.tanh)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num, mean=self.mean, std=self.std) 

    # Optimizers
    self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
    self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

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
    print(model)
    print("The number of parameters: {}".format(num_params))

  #=======================================================================================#
  #=======================================================================================#

  def load_pretrained_model(self):
    # ipdb.set_trace()
    self.G.load_state_dict(torch.load(os.path.join(
      self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
    self.D.load_state_dict(torch.load(os.path.join(
      self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
    print('loaded trained models (step: {})..!'.format(self.pretrained_model))

  #=======================================================================================#
  #=======================================================================================#

  def build_tensorboard(self):
    # ipdb.set_trace()
    from logger import Logger
    self.logger = Logger(self.log_path)

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
      out = (x*self.std.data.cpu().view(1,-1,1,1))+self.mean.data.cpu().view(1,-1,1,1)
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

  #=======================================================================================#
  #=======================================================================================#

  def norm(self, x):
    if self.MEAN=='data_image':
      # ipdb.set_trace()
      mean = torch.from_numpy(x.data.cpu().numpy().mean(axis=(3,2)).reshape(x.size(0),x.size(1),1,1))
      std  = torch.from_numpy(x.data.cpu().numpy().std(axis=(3,2)).reshape(x.size(0),x.size(1),1,1))

      mean = self.to_var(mean)
      std = self.to_var(std)

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

  #=======================================================================================#
  #=======================================================================================#

  def get_max_dataset(self):
    max_data = []
    mean=0.0
    std=0.0
    for real_x, real_label, files in tqdm(self.MultiLabelAU_loader, desc='Confirming mean'):
      if self.G_norm:
        real_x = self.norm(self.to_var(real_x, volatile=True))
      # ipdb.set_trace()
      max_data.append(real_x.max().data[0])#.max(dim=3)[0].max(dim=2)[0].max(dim=1)[0]
      # mean += real_x.data.mean(0) / (float(len(self.MultiLabelAU_loader.dataset)))
    # for real_x, real_label, files in tqdm(self.MultiLabelAU_loader, desc='Confirming std'):
    #   if self.G_norm:
    #     real_x = self.norm(self.to_var(real_x, volatile=True))
    #   ipdb.set_trace()
    #   std += ((real_x - mean)**2).sum(3).sum(2).sum(0) / (float(len(self.MultiLabelAU_loader.dataset))*real_x.size(2)*real_x.size(3))
    # ipdb.set_trace()
    # std = torch.sqrt(std_channels )
    return np.max(max_data)

  #=======================================================================================#
  #=======================================================================================#

  def threshold(self, x):
    x = x.clone()
    x = (x >= 0.5).float()
    return x

  #=======================================================================================#
  #=======================================================================================#

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
    resize = lambda x: skimage.transform.resize(imageio.imread(line), (self.image_size,self.image_size))
    imgs = [resize(line).transpose(2,0,1) for line in sorted(glob.glob('/home/afromero/aus_flat/*.jpeg'))]
    # ipdb.set_trace()
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
      if gray[i] and self.GRAY:
        conv_img[i] = torch.from_numpy(filters.gaussian_filter(img[i], sigma=sigma[i], truncate=trunc[i]))
      else:
        for j in range(img.size(1)):
          conv_img[i,j] = torch.from_numpy(filters.gaussian_filter(img[i,j], sigma=sigma[i], truncate=trunc[i]))

    return conv_img

  #=======================================================================================#
  #=======================================================================================#

  def show_img(self, img, real_label, fake_label, hist_match=None, ppt=False):         

    AUS_SHOW = self.get_aus()
    # if self.BLUR: img = self.blurRANDOM(img)
    # img = F.conv

    # conv_img=F.conv2d(img,self.to_var(torch.randn(3,3,3,3), volatile=True), padding=1)
    # conv_img = self.gauss(img,3,2)
    
    # for i in range(3,30,2):
    #   for j in range(1,50,7):
    #     save_image(self.denorm(self.gauss(img,i,j).data).cpu(), 'dummy%s_%s.jpg'%(i,j))
    # ipdb.set_trace()

    fake_image_list=[img]
    flag_time=True

    for fl in fake_label:
      # ipdb.set_trace()
      if self.G_norm:
        fl = fl*self.MAX_DATASET
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
    # ipdb.set_trace()
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
      # save_image(self.denorm(fake_images.data.cpu()[:shape0,:,:,self.image_size:]), 'tmp_fake.jpg',nrow=1, padding=0)
      # save_image(self.denorm(fake_images.data.cpu()[:shape0,:,:,:self.image_size]), 'tmp_real.jpg',nrow=1, padding=0)
      save_image(fake_images, file_name+'.jpg',nrow=1, padding=0)

      # print("Real Label: \n"+str(real_label.data.cpu()[:shape0].numpy()))
      # for fl in fake_label:
        # print("Fake Label: \n"+str(fl.data.cpu()[:shape0].numpy()))   
      # os.system('eog tmp_real.jpg')
      # os.system('eog tmp_fake.jpg')
      os.system('eog tmp_all.jpg')    
      # os.remove('tmp_real.jpg')
      # os.remove('tmp_fake.jpg')
      os.remove('tmp_all.jpg')

  #=======================================================================================#
  #=======================================================================================#

  def show_img_single(self, img):  
    # ipdb.set_trace()        
    img_ = self.denorm(img.data.cpu())
    save_image(img_.cpu(), 'show/tmp0.jpg',nrow=int(math.sqrt(img_.size(0))), padding=0)
    os.system('eog show/tmp0.jpg')    
    os.remove('show/tmp0.jpg')    

  #=======================================================================================#
  #=======================================================================================#

  def train(self):
    """Train StarGAN within a single dataset."""

    # Set dataloader
    if self.dataset == 'MultiLabelAU':
      self.data_loader = self.MultiLabelAU_loader     
    elif self.dataset == 'au01_fold0':
      self.data_loader = self.au_loader   

    # The number of iterations per epoch
    iters_per_epoch = len(self.data_loader)

    fixed_x = []
    real_c = []
    for i, (images, labels, files) in enumerate(self.data_loader):
      # ipdb.set_trace()
      if self.BLUR: images = self.blurRANDOM(images)
      fixed_x.append(images)
      real_c.append(labels)
      if i == 1:
        break

    # Fixed inputs and target domain labels for debugging
    # ipdb.set_trace()
    fixed_x = torch.cat(fixed_x, dim=0)
    fixed_x = self.to_var(fixed_x, volatile=True)
    real_c = torch.cat(real_c, dim=0)
    
    if self.dataset == 'CelebA':
      fixed_c_list = self.make_celeb_labels(real_c)
    # elif self.dataset == 'MultiLabelAU':
    #  fixed_c_list = [self.to_var(torch.FloatTensor(np.random.randint(0,2,[self.batch_size*4,self.c_dim])), volatile=True)]*4
    elif self.dataset == 'MultiLabelAU':
      fixed_c_list = [self.to_var(torch.zeros(fixed_x.size(0), self.c_dim), volatile=True)]
      for i in range(self.c_dim):
        # ipdb.set_trace()
        fixed_c = self.one_hot(torch.ones(fixed_x.size(0)) * i, self.c_dim)
        fixed_c_list.append(self.to_var(fixed_c, volatile=True))


    # lr cache for decaying
    g_lr = self.g_lr
    d_lr = self.d_lr

    # Start with trained model if exists
    if self.pretrained_model:
      start = int(self.pretrained_model.split('_')[0])
      for i in range(start):
        if (i+1) > (self.num_epochs - self.num_epochs_decay):
          g_lr -= (self.g_lr / float(self.num_epochs_decay))
          d_lr -= (self.d_lr / float(self.num_epochs_decay))
          self.update_lr(g_lr, d_lr)
          print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))     
    else:
      start = 0

    last_model_step = len(self.data_loader)

    # Start training

    print("Log path: "+self.log_path)

    Log = "---> batch size: {}, fold: {}, img: {}, GPU: {}, !{}".format(self.batch_size, self.fold, self.image_size, self.GPU, self.mode_data) 

    if self.COLOR_JITTER: Log += ' [*COLOR_JITTER]'
    if self.BLUR: Log += ' [*BLUR]'
    if self.GRAY: Log += ' [*GRAY]'
    if self.MEAN!='0.5': Log += ' [*{}]'.format(self.MEAN)
    print(Log)
    loss_cum = {}
    start_time = time.time()
    flag_init=True
    for e in range(start, self.num_epochs):
      E = str(e+1).zfill(3)
      self.D.train()
      self.G.train()

      if flag_init:
        f1, loss, f1_1 = self.val_cls(init=True)   
        log = '[F1_VAL: %0.3f (F1_1: %0.3f) LOSS_VAL: %0.3f]'%(np.mean(f1), np.mean(f1_1), np.mean(loss))
        print(log)
        flag_init = False

      desc_bar = 'Epoch: %d/%d'%(e,self.num_epochs)
      progress_bar = tqdm(enumerate(self.data_loader), \
          total=len(self.data_loader), desc=desc_bar)
      for i, (real_x, real_label, files) in progress_bar:
        
        # save_image(self.denorm(real_x.cpu()), 'dm1.png',nrow=1, padding=0)


        #=======================================================================================#
        #========================================== BLUR =======================================#
        #=======================================================================================#
        np.random.seed(i+(e*len(self.data_loader)))
        if self.BLUR and np.random.randint(0,2,1)[0]:
          # for i in range(3,27,2):
          #   for j in range(1,10):
          #     save_image(self.denorm(self.blur(real_x,i,j)), 'dummy%s_%s_color.jpg'%(i,j))      
          # ipdb.set_trace()    
          # save_image(self.denorm(self.blurRANDOM(real_x)), 'dummy.jpg')
          real_x = self.blurRANDOM(real_x)
          

        #=======================================================================================#
        #==================================== DATA 2 VAR =======================================#
        #=======================================================================================#
        # Generat fake labels randomly (target domain labels)
        rand_idx = torch.randperm(real_label.size(0))
        fake_label = real_label[rand_idx]
        # ipdb.set_trace()
        if self.dataset == 'CelebA' or self.dataset=='MultiLabelAU':
          real_c = real_label.clone()
          fake_c = fake_label.clone()
        else:
          real_c = self.one_hot(real_label, self.c_dim)
          fake_c = self.one_hot(fake_label, self.c_dim)

        # Convert tensor to variable
        real_x = self.to_var(real_x)
        real_c = self.to_var(real_c)       # input for the generator
        fake_c = self.to_var(fake_c)
        real_label = self.to_var(real_label)   # this is same as real_c if dataset == 'CelebA'
        fake_label = self.to_var(fake_label)
        
        #=======================================================================================#
        #======================================== Train D ======================================#
        #=======================================================================================#
        # ipdb.set_trace()
        # Compute loss with real images
        if self.D_norm:
          real_x_D = self.norm(real_x)
        else:
          real_x_D = real_x
        # ipdb.set_trace()
        out_src, out_cls = self.D(real_x_D)
        d_loss_real = - torch.mean(out_src)
        # ipdb.set_trace()
        # if self.FOCAL_LOSS:
        #  d_loss_cls = self.focal_loss(
        #    out_cls, real_label) / real_x.size(0)
        # else:
        d_loss_cls = F.binary_cross_entropy_with_logits(
          out_cls, real_label, size_average=False) / real_x.size(0)


        # Compute loss with fake images
        if self.G_norm:
          real_x_G = self.norm(real_x)
          # fake_c = fake_c*int(real_x_G.max)
          fake_c = fake_c*self.MAX_DATASET
          # ipdb.set_trace()
        else:
          real_x_G = real_x         

        fake_x = self.G(real_x_G, fake_c)

        if self.D_norm and self.G_norm:
          fake_x_D = fake_x #self.norm(fake_x.data.clone())
        elif self.D_norm and not self.G_norm:
          fake_x_D = self.norm(fake_x)
        elif not self.D_norm and self.G_norm:
          fake_x_D = self.to_var(self.denorm(fake_x.data.cpu(), real_x.data))
        else:
          fake_x_D = fake_x

        fake_x_D = Variable(fake_x_D.data)
        out_src, out_cls = self.D(fake_x_D)
        d_loss_fake = torch.mean(out_src)

        # Backward + Optimize
        
        d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls
        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # Compute gradient penalty
        alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
        # ipdb.set_trace()
        interpolated = Variable(alpha * real_x_D.data + (1 - alpha) * fake_x_D.data, requires_grad=True)
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
        d_loss = self.lambda_gp * d_loss_gp
        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # Logging
        loss = {}
        loss['D/real'] = d_loss_real.data[0]
        loss['D/fake'] = d_loss_fake.data[0]
        loss['D/cls'] = d_loss_cls.data[0]
        loss['D/gp'] = d_loss_gp.data[0]
        if len(loss_cum.keys())==0: 
          loss_cum['D/loss_real'] = []; loss_cum['D/loss_fake'] = []
          loss_cum['D/loss_cls'] = []; loss_cum['D/loss_gp'] = []
          loss_cum['G/loss_fake'] = []; loss_cum['G/loss_rec'] = []
          loss_cum['G/loss_cls'] = []
        loss_cum['D/loss_real'].append(d_loss_real.data[0])

        loss_cum['D/loss_fake'].append(d_loss_fake.data[0])
        loss_cum['D/loss_cls'].append(d_loss_cls.data[0])
        loss_cum['D/loss_gp'].append(d_loss_gp.data[0])

        #=======================================================================================#
        #======================================= Train G =======================================#
        #=======================================================================================#
        if (i+1) % self.d_train_repeat == 0:

          # Original-to-target and target-to-original domain
          fake_x = self.G(real_x_G, fake_c)
          if self.G_norm:
            real_c = real_c*int(fake_x.max())         
          rec_x = self.G(fake_x, real_c)

          if self.D_norm and self.G_norm:
            fake_x_D = fake_x #self.norm(fake_x.data.clone())
          elif self.D_norm and not self.G_norm:
            fake_x_D = self.norm(fake_x)
          elif not self.D_norm and self.G_norm:
            fake_x_D = self.to_var(self.denorm(fake_x.data.cpu(), real_x.data))
          else:
            fake_x_D = fake_x

          # fake_x_D = self.to_var(fake_x_D)
          out_src, out_cls = self.D(fake_x_D)
          g_loss_fake = - torch.mean(out_src)
          g_loss_rec = torch.mean(torch.abs(real_x_G - rec_x))

          # if self.FOCAL_LOSS:
          #  g_loss_cls = self.focal_loss(
          #    out_cls, fake_label) / fake_x.size(0)
          # else:
          g_loss_cls = F.binary_cross_entropy_with_logits(
            out_cls, fake_label, size_average=False) / fake_x.size(0)


          # Backward + Optimize
          g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
          self.reset_grad()
          g_loss.backward()
          self.g_optimizer.step()

          # Logging
          loss['G/fake'] = g_loss_fake.data[0]
          loss['G/rec'] = g_loss_rec.data[0]
          loss['G/cls'] = g_loss_cls.data[0]

          loss_cum['G/loss_fake'].append(g_loss_fake.data[0])
          loss_cum['G/loss_rec'].append(g_loss_rec.data[0])
          loss_cum['G/loss_cls'].append(g_loss_cls.data[0])

        #=======================================================================================#
        #========================================MISCELANEOUS===================================#
        #=======================================================================================#

        # Print out log info
        if (i+1) % self.log_step == 0 or (i+1)==last_model_step:
          # progress_bar.set_postfix(G_loss_rec=np.array(loss_cum['G/loss_rec']).mean())
          progress_bar.set_postfix(**loss)
          if (i+1)==last_model_step or self.D_norm or self.G_norm: progress_bar.set_postfix('')
          if self.use_tensorboard:
            for tag, value in loss.items():
              self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

        # Translate fixed images for debugging
        if (i+1) % self.sample_step == 0 or (i+1)==last_model_step or i+e==0:
          self.G.eval()
          if self.G_norm:
            fake_image_list = [self.norm(fixed_x)]
          else:
            fake_image_list = [fixed_x]
          # ipdb.set_trace()
          for fixed_c in fixed_c_list:
            if self.G_norm:
              fixed_x_G = self.norm(fixed_x)
              fixed_c = fixed_c*self.MAX_DATASET               
            else:
              fixed_x_G = fixed_x              
            fake_image_list.append(self.G(fixed_x_G, fixed_c))
          fake_images = torch.cat(fake_image_list, dim=3)
          # ipdb.set_trace()
          shape0 = min(64, fake_images.data.cpu().shape[0])
          img_denorm = self.denorm(fake_images.data.cpu()[:shape0], img_org=fixed_x.data)
          save_image(img_denorm,
            os.path.join(self.sample_path, '{}_{}_fake.jpg'.format(E, i+1)),nrow=1, padding=0)
          # print('Translated images and saved into {}..!'.format(self.sample_path))

      torch.save(self.G.state_dict(),
        os.path.join(self.model_save_path, '{}_{}_G.pth'.format(E, i+1)))
      torch.save(self.D.state_dict(),
        os.path.join(self.model_save_path, '{}_{}_D.pth'.format(E, i+1)))


      #=======================================================================================#
      #=========================================METRICS=======================================#
      #=======================================================================================#
      #F1 val
      f1, loss,_ = self.val_cls()
      if self.use_tensorboard:
        # print("Log path: "+self.log_path)
        for idx, au in enumerate(cfg.AUs):
          self.logger.scalar_summary('F1_val_'+str(au).zfill(2), f1[idx], e * iters_per_epoch + i + 1)      
          self.logger.scalar_summary('Loss_val_'+str(au).zfill(2), loss[idx], e * iters_per_epoch + i + 1)      
        self.logger.scalar_summary('F1_val_mean', np.array(f1).mean(), e * iters_per_epoch + i + 1)     
        self.logger.scalar_summary('Loss_val_mean', np.array(loss).mean(), e * iters_per_epoch + i + 1)     

        for tag, value in loss_cum.items():
          self.logger.scalar_summary(tag, np.array(value).mean(), e * iters_per_epoch + i + 1)   
                 
      #Stats per epoch
      elapsed = time.time() - start_time
      elapsed = str(datetime.timedelta(seconds=elapsed))
      log = '!Elapsed: %s | [F1_VAL: %0.3f LOSS_VAL: %0.3f]\nTrain'%(elapsed, np.array(f1).mean(), np.array(loss).mean())
      for tag, value in loss_cum.items():
        log += ", {}: {:.4f}".format(tag, np.array(value).mean())   

      print(log)

      # Decay learning rate     
      if (e+1) > (self.num_epochs - self.num_epochs_decay):
        g_lr -= (self.g_lr / float(self.num_epochs_decay))
        d_lr -= (self.d_lr / float(self.num_epochs_decay))
        self.update_lr(g_lr, d_lr)
        print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

  #=======================================================================================#
  #=======================================================================================#

  def save_fake_output(self, real_x, save_path):
    real_x = self.to_var(real_x, volatile=True)

    # target_c_list = []
    target_c= torch.from_numpy(np.zeros((real_x.size(0), self.c_dim), dtype=np.float32))
    target_c_list = [self.to_var(target_c, volatile=True)]
    for j in range(self.c_dim):
      target_c[:,j]=1       
      target_c_list.append(self.to_var(target_c, volatile=True))
      # target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
      # target_c_list.append(self.to_var(target_c, volatile=True))

    # Start translations
    fake_image_list = [real_x]

    for target_c in target_c_list:
      if self.G_norm:
        target_c = target_c*int(real_x.max())          
      fake_x = self.G(real_x, target_c)
      #out_src_temp, out_cls_temp = self.D(fake_x)
      #F.sigmoid(out_cls_temp)
      #accuracies = self.compute_accuracy(out_cls_temp, target_c, self.dataset)
      fake_image_list.append(fake_x)
    fake_images = torch.cat(fake_image_list, dim=3)

    # ipdb.set_trace()
    save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)

  #=======================================================================================#
  #=======================================================================================#

  def test(self):
    """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
    # Load trained parameters
    from data_loader import get_loader
    if self.test_model=='':
      last_file = sorted(glob.glob(os.path.join(self.model_save_path,  '*_D.pth')))[-1]
      last_name = '_'.join(last_file.split('/')[-1].split('_')[:2])
    else:
      last_name = self.test_model

    G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(last_name))
    self.G.load_state_dict(torch.load(G_path))
    self.G.eval()
    # ipdb.set_trace()

    data_loader_val = get_loader(self.metadata_path, self.image_size,
                 self.image_size, self.batch_size, 'MultiLabelAU', 'val')   

    for i, (real_x, org_c, files) in enumerate(data_loader_val):
      save_path = os.path.join(self.sample_path, '{}_fake_val_{}.jpg'.format(last_name, i+1))
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
      self.data_loader_val = get_loader(self.metadata_path, self.image_size,
                   self.image_size, self.batch_size, 'MultiLabelAU', 'val', shuffling=True)

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
    if self.test_model=='':
      last_file = sorted(glob.glob(os.path.join(self.model_save_path,  '*_D.pth')))[-1]
      last_name = '_'.join(last_file.split('/')[-1].split('_')[:2])
    else:
      last_name = self.test_model

    G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(last_name))
    D_path = os.path.join(self.model_save_path, '{}_D.pth'.format(last_name))
    txt_path = os.path.join(self.model_save_path, '{}_{}_test.txt'.format(last_name,'{}'))
    show_fake = os.path.join(self.sample_path, '{}_fake_{}_{}.jpg'.format(last_name,'{}', '{}'))
    self.pkl_data = os.path.join(self.model_save_path, '{}_{}.pkl'.format(last_name, '{}'))
    self.lstm_path = os.path.join(self.model_save_path, '{}_lstm'.format(last_name))
    if not os.path.isdir(self.lstm_path): os.makedirs(self.lstm_path)
    print(" [!!] {} model loaded...".format(D_path))
    # ipdb.set_trace()
    self.G.load_state_dict(torch.load(G_path))
    self.D.load_state_dict(torch.load(D_path))
    self.G.eval()
    self.D.eval()
    # ipdb.set_trace()
    if self.dataset == 'MultiLabelAU' and not self.GOOGLE:
      data_loader_val = get_loader(self.metadata_path, self.image_size,
                self.image_size, self.batch_size, 'MultiLabelAU', 'val', shuffling=True)
      data_loader_test = get_loader(self.metadata_path, self.image_size,
                self.image_size, self.batch_size, 'MultiLabelAU', 'test', shuffling=True)
    elif self.dataset == 'au01_fold0':
      data_loader = self.au_loader  

    if self.GOOGLE: 
      data_loader_google = get_loader('', self.image_size, self.image_size, \
                      self.batch_size, 'Google',  mode=self.mode_data, shuffling=True)
      # data_loader_google = get_loader(self.metadata_path, self.image_size, \
      #           self.image_size, self.batch_size, 'MultiLabelAU', 'test', shuffling=True)     

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
    # F1_real, F1_max, max_thresh_train  = self.F1_TEST(data_loader_train, mode = 'TRAIN')
    # _ = self.F1_TEST(data_loader_test, thresh = max_thresh_train)
    if self.GOOGLE: 
      _ = F1_TEST(self, data_loader_google)
    else: 
      F1_real, F1_max, max_thresh_val, _, _  = F1_TEST(self, data_loader_val, mode = 'VAL')
      _ = F1_TEST(self, data_loader_test, thresh = max_thresh_val, show_fake = show_fake) 
      # _ = self.F1_TEST(data_loader_test)
    self.f.close()

  #=======================================================================================#
  #=======================================================================================#

  def save_lstm(self, data, files):
    assert data.shape[0]==len(files)
    for i in range(len(files)):
      name = os.path.join(self.lstm_path, '/'.join(files[i].split('/')[-6:]))
      name = name.replace('jpg', 'npy')
      folder = os.path.dirname(name)
      if not os.path.isdir(folder): os.makedirs(folder)
      np.save(name, data[i])

