import torch, os, time, ipdb, glob, math, warnings, datetime
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import config as cfg
from tqdm import tqdm
from termcolor import colored
from misc.utils import create_dir, denorm, get_aus, get_loss_value, make_gif, PRINT, send_mail, target_debug_list, TimeNow, TimeNow_str, to_cuda, to_data, to_var
from misc.losses import _compute_kl, _compute_loss_smooth, _compute_vgg_loss, _GAN_LOSS, _get_gradient_penalty
import torch.utils.data.distributed
from misc.utils import _horovod
hvd = _horovod()

warnings.filterwarnings('ignore')

class Solver(object):

  def __init__(self, config, data_loader=None):
    # Data loader
    self.data_loader = data_loader
    self.config = config
    # ipdb.set_trace()
    # Build tensorboard if use
    self.build_model()
    if self.config.use_tensorboard:
      self.build_tensorboard()

    # Start with trained model
    if self.config.pretrained_model:
      self.load_pretrained_model()
    elif self.config.LOAD_SMIT:
      self.load_pretrained_smit()

  #=======================================================================================#
  #=======================================================================================#
  def build_model(self):
    # Define a generator and a discriminator
    if self.config.MultiDis>0:
      from model import MultiDiscriminator as Discriminator
    else:
      from model import Discriminator
    if 'AdaIn' in self.config.GAN_options and 'Stochastic' not in self.config.GAN_options:
      from model import AdaInGEN_Label as GEN
    elif 'AdaIn' in self.config.GAN_options:
      if 'DRIT' not in self.config.GAN_options: from model import AdaInGEN as GEN
      else: from model import DRITGEN as GEN
    elif 'DRITZ' in self.config.GAN_options: from model import DRITZGEN as GEN
    else: from model import Generator as GEN

    self.D = Discriminator(self.config, debug=self.config.mode=='train' and not self.config.HOROVOD) 
    to_cuda(self.D)
    D_parameters = filter(lambda p: p.requires_grad, self.D.parameters())
    self.d_optimizer = torch.optim.Adam(D_parameters, self.config.d_lr*hvd.size(), [self.config.beta1, self.config.beta2])    

    self.G = GEN(self.config, debug=self.config.mode=='train' and not self.config.HOROVOD, STYLE_ENC=self.D if 'STYLE_DISC' in self.config.GAN_options else None)
    if 'Split_Optim' in self.config.GAN_options:
      G_parameters = self.G.generator.parameters()
      S_parameters = self.G.enc_style.parameters()
      S_parameters = filter(lambda p: p.requires_grad, S_parameters)
      self.s_optimizer = torch.optim.Adam(S_parameters, self.config.g_lr, [self.config.beta1, self.config.beta2])
    else:
      G_parameters = self.G.parameters()    
    G_parameters = filter(lambda p: p.requires_grad, G_parameters)
    self.g_optimizer = torch.optim.Adam(G_parameters, self.config.g_lr*hvd.size(), [self.config.beta1, self.config.beta2])
    to_cuda(self.G)

    if self.config.mode=='train': 
      self.print_network(self.D, 'Discriminator')
      self.print_network(self.G, 'Generator')

    if ('L1_Perceptual' in self.config.GAN_options or 'Perceptual' in self.config.GAN_options) and self.config.mode=='train':
      import importlib
      perceptual = importlib.import_module('models.perceptual.{}'.format(self.config.PerceptualLoss))
      self.vgg = getattr(perceptual, self.config.PerceptualLoss)()
      to_cuda(self.vgg)
      self.vgg.eval()
      for param in self.vgg.parameters():
          param.requires_grad = False          

    if self.config.HOROVOD and self.config.mode=='train':
      self.d_optimizer = hvd.DistributedOptimizer(self.d_optimizer, named_parameters=self.D.named_parameters())      
      hvd.broadcast_parameters(self.D.state_dict(), root_rank=0)
      # hvd.broadcast_optimizer_state(self.d_optimizer, root_rank=0)          

      self.g_optimizer = hvd.DistributedOptimizer(self.g_optimizer, named_parameters=self.G.named_parameters())
      hvd.broadcast_parameters(self.G.state_dict(), root_rank=0)
      # hvd.broadcast_optimizer_state(self.g_optimizer, root_rank=0)          
  #=======================================================================================#
  #=======================================================================================#

  def print_network(self, model, name):
    if 'AdaIn' in self.config.GAN_options and name=='Generator':
      if 'Stochastic' in self.config.GAN_options:
        choices = ['generator', 'enc_style', 'adain_net']
      else:
        choices = ['generator', 'adain_net']
      if 'DRIT' in self.config.GAN_options: choices.pop(-1)
      for m in choices:
        submodel = getattr(model, m)
        num_params = 0
        for p in submodel.parameters():
          num_params += p.numel()
        self.PRINT("{} number of parameters: {}".format(m.upper(), num_params))
    else:
      num_params = 0
      for p in model.parameters():
        num_params += p.numel()   
      self.PRINT("{} number of parameters: {}".format(name.upper(), num_params))   
    # self.PRINT(name)
    # self.PRINT(model)
    # self.PRINT("{} number of parameters: {}".format(name, num_params))
    # self.display_net(name)

  #=======================================================================================#
  #=======================================================================================#
  def save(self, Epoch, iter):
    if hvd.rank() != 0: return
    name = os.path.join(self.config.model_save_path, '{}_{}_{}.pth'.format(Epoch, iter, '{}'))
    torch.save(self.G.state_dict(), name.format('G'))
    torch.save(self.g_optimizer.state_dict(), name.format('G_optim'))
    torch.save(self.D.state_dict(), name.format('D'))
    torch.save(self.d_optimizer.state_dict(), name.format('D_optim'))   
    if 'Split_Optim' in self.config.GAN_options:
      torch.save(self.s_optimizer.state_dict(), name.format('S_optim'))   

    if int(Epoch)>2:
      name_1 = os.path.join(self.config.model_save_path, '{}_{}_{}.pth'.format(str(int(Epoch)-1).zfill(3), iter, '{}'))
      if os.path.isfile(name_1.format('G')): os.remove(name_1.format('G'))
      if os.path.isfile(name_1.format('G_optim')): os.remove(name_1.format('G_optim'))
      if os.path.isfile(name_1.format('D')): os.remove(name_1.format('D'))
      if os.path.isfile(name_1.format('D_optim')): os.remove(name_1.format('D_optim'))   
      if 'Split_Optim' in self.config.GAN_options:
        if os.path.isfile(name_1.format('S_optim')): os.remove(name_1.format('S_optim'))           

  #=======================================================================================#
  #=======================================================================================#
  def load_pretrained_model(self):
    if hvd.rank() != 0: return
    self.PRINT('Resume model (step: {})..!'.format(self.config.pretrained_model))
    name = os.path.join(self.config.model_save_path, '{}_{}.pth'.format(self.config.pretrained_model, '{}'))
    if self.config.mode=='train': self.PRINT('Model: {}'.format(name))
    self.G.load_state_dict(torch.load(name.format('G'), map_location=lambda storage, loc: storage))
    self.D.load_state_dict(torch.load(name.format('D'), map_location=lambda storage, loc: storage))
    try:
      self.g_optimizer.load_state_dict(torch.load(name.format('G_optim'), map_location=lambda storage, loc: storage))        
      self.optim_cuda(self.g_optimizer)
      self.d_optimizer.load_state_dict(torch.load(name.format('D_optim'), map_location=lambda storage, loc: storage))      
      self.optim_cuda(self.d_optimizer)
      self.s_optimizer.load_state_dict(torch.load(name.format('S_optim'), map_location=lambda storage, loc: storage))      
      self.optim_cuda(self.s_optimizer)  
    except: pass

  #=======================================================================================#
  #=======================================================================================#    
  def optim_cuda(self, optimizer):
    for state in optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = to_cuda(v)#.cuda()    

  #=======================================================================================#
  #=======================================================================================#
  def load_pretrained_smit(self):
    from misc.utils import replace_weights
    if hvd.rank() != 0: return
    smit_G = sorted(glob.glob('snapshot/GAN/CelebA/models/faces/128/fold_0/RaGAN/L1_LOSS/lambda_l1_10.0/Stochastic/style_1.0_dim_4/AdaIn2/InterStyleConcatLabels/FC/Split_Optim/rec_style/*.pth'))[-1]
    smit_D = smit_G.replace('G.pth', 'D.pth')
    d_weights = torch.load(smit_D)
    replace_weights(d_weights, self.D.state_dict(), ['conv2.weight'])
    self.D.load_state_dict(d_weights)#, map_location=lambda storage, loc: storage))

    #Only loading G weights, not Style and not AdaIn.
    g_weights = {key:param for key,param in torch.load(smit_G).items() if 'style' not in key and 'adain' not in key}
    style_key = [key for key in self.G.state_dict().keys() if 'style' in key or 'adain' in key]
    replace_weights(g_weights, self.G.state_dict(), style_key)
    self.G.load_state_dict(g_weights)#, map_location=lambda storage, loc: storage))

    self.PRINT('loaded trained smit model from: {}..!'.format(smit_G))    

  #=======================================================================================#
  #=======================================================================================#
  def resume_name(self):
    if self.config.pretrained_model in ['',None]:
      last_file = sorted(glob.glob(os.path.join(self.config.model_save_path,  '*_G.pth')))[-1]
      last_name = '_'.join(os.path.basename(last_file).split('_')[:2])
    else:
      last_name = self.config.pretrained_model 
    return last_name

  #=======================================================================================#
  #=======================================================================================#
  def build_tensorboard(self):
    from misc.logger import Logger
    self.logger = Logger(self.config.log_path)

  #=======================================================================================#
  #=======================================================================================#
  def update_lr(self, g_lr, d_lr):
    for param_group in self.g_optimizer.param_groups:
      param_group['lr'] = g_lr*hvd.size()
    for param_group in self.d_optimizer.param_groups:
      param_group['lr'] = d_lr*hvd.size()

  #=======================================================================================#
  #=======================================================================================#
  def reset_grad(self):
    self.g_optimizer.zero_grad()
    self.d_optimizer.zero_grad()
    if 'Stochastic' in self.config.GAN_options and 'Split_Optim' in self.config.GAN_options:
      self.s_optimizer.zero_grad()
      if 'Split_Optim_all' in self.config.GAN_options:
        self.c_optimizer.zero_grad()

  #=======================================================================================#
  #=======================================================================================#
  def update_loss(self, loss, value):
    try:
      self.LOSS[loss].append(value)
    except:
      self.LOSS[loss] = []
      self.LOSS[loss].append(value)

  #=======================================================================================#
  #=======================================================================================#
  def get_aus(self):
    return get_aus(self.config.image_size, self.config.dataset_fake, attr=self.data_loader.dataset)

  #=======================================================================================#
  #=======================================================================================#
  def color(self, dict, key, color='red'):
    from termcolor import colored
    dict[key] = colored('%.2f'%(dict[key]), color)

  #=======================================================================================#
  #=======================================================================================#
  def get_randperm(self, x):
    if x.size(0)>2:
      rand_idx = to_var(torch.randperm(x.size(0)))
    elif x.size(0)==2:
      rand_idx = to_var(torch.LongTensor([1,0]))
    else:
      rand_idx = to_var(torch.LongTensor([0]))
    return rand_idx

  #=======================================================================================#
  #=======================================================================================#
  def PRINT(self, str):  
    if self.config.HOROVOD and hvd.rank() != 0: return
    if self.config.mode=='train': PRINT(self.config.log, str)
    else: print(str)

  #=======================================================================================#
  #=======================================================================================#
  def _criterion_style(self, output_style, target_style):
    criterion_style = torch.nn.MSELoss() if 'mse_style' in self.config.GAN_options else torch.nn.L1Loss()
    return criterion_style(output_style, target_style)

  #=======================================================================================#
  #=======================================================================================#
  def _compute_vgg_loss(self, data_x, data_y):
    return _compute_vgg_loss(self.vgg, data_x, data_y, IN=not 'NoPerceptualIn' in self.config.GAN_options)

  #=======================================================================================#
  #=======================================================================================#
  def _CLS(self, data):
    data = to_var(data, volatile=True)
    out_label = self.D(data)[1]
    if len(out_label)>1:
      out_label = torch.cat([F.sigmoid(out.unsqueeze(-1)) for out in out_label], dim=-1).mean(dim=-1)
    else:
      out_label = F.sigmoid(out_label[0])
    out_label = (out_label>0.5).float()    
    return out_label

  #=======================================================================================#
  #=======================================================================================#
  def _SAVE_IMAGE(self, save_path, fake_list, attn_list=[], im_size=256, gif=False, mode='fake'):
    Output = []
    if self.config.HOROVOD and hvd.rank() != 0: return Output
    fake_images = denorm(to_data(torch.cat(fake_list, dim=3), cpu=True))
    if fake_images.size(0)>1:
      fake_images = torch.cat((self.get_aus(), fake_images), dim=0)
      if gif: make_gif(fake_images, save_path, im_size=im_size)
    else:
      if gif: make_gif(fake_images, save_path, im_size=im_size)
      fake_images = torch.cat((self.get_aus(), fake_images), dim=0)
    _save_path = save_path.replace('fake', mode)    
    save_image(fake_images, _save_path, nrow=1, padding=0)
    Output.append(_save_path)
    if gif: 
      Output.append(_save_path.replace('jpg', 'gif'))
      Output.append(_save_path.replace('jpg', 'mp4'))
    if len(attn_list): 
      fake_attn = to_data(torch.cat(attn_list, dim=3), cpu=True)
      fake_attn = torch.cat((self.get_aus(), fake_attn), dim=0)
      if 'fake' not in os.path.basename(save_path): save_path=save_path.replace('.jpg', '_fake.jpg')
      _save_path = save_path.replace('fake', '{}_attn'.format(mode))
      save_image(fake_attn, _save_path, nrow=1, padding=0)
      Output.append(_save_path.replace('fake', 'attn'))    
    return Output

  #=======================================================================================#
  #=======================================================================================#
  def _GAN_LOSS(self, real_x, fake_x, label, is_fake=False):
    RafD = False#self.config.dataset_fake=='RafD'
    return _GAN_LOSS(self.D, real_x, fake_x, label, self.config.GAN_options, is_fake=is_fake, RafD=RafD)

  #=======================================================================================#
  #=======================================================================================#
  def _get_gradient_penalty(self, real_x, fake_x):
    return _get_gradient_penalty(self.D, real_x, fake_x)

  #=======================================================================================#
  #=======================================================================================#
  def Disc_update(self, real_x0, real_c0, GAN_options):

    rand_idx0 = self.get_randperm(real_c0)
    fake_c0 = real_c0[rand_idx0]
    fake_c0 = to_var(fake_c0.data)

    ############################# Stochastic Part ##################################
    if 'Stochastic' in GAN_options:
      style_fake0 = to_var(self.G.random_style(real_x0))
      if 'style_labels' in GAN_options:
        style_fake0 = style_fake0*fake_c0.unsqueeze(2)
    else:
      style_fake0 = None

    fake_x0 = self.G(real_x0, fake_c0, stochastic=style_fake0)[0]

    #=======================================================================================#
    #======================================== Train D ======================================#
    #=======================================================================================#
    # if self.config.dataset_fake=='RafD':
    #   d_loss_src, d_loss_cls, d_loss_cls_pose, _ = self._GAN_LOSS(real_x0, fake_x0, real_c0)
    #   d_loss_cls_pose = self.config.lambda_cls_pose * d_loss_cls_pose  
    #   self.loss['Dcls_p'] = get_loss_value(d_loss_cls_pose)          
    #   self.update_loss('Dcls_p', self.loss['Dcls_p'])      
    # else:
    d_loss_src, d_loss_cls, _ = self._GAN_LOSS(real_x0, fake_x0, real_c0)
    
    d_loss_cls = self.config.lambda_cls * d_loss_cls  

    self.loss['Dsrc'] = get_loss_value(d_loss_src)
    self.loss['Dcls'] = get_loss_value(d_loss_cls)          
    self.update_loss('Dsrc', self.loss['Dsrc'])
    self.update_loss('Dcls', self.loss['Dcls'])

    # Backward + Optimize       
    d_loss = d_loss_src + d_loss_cls

    self.reset_grad()
    d_loss.backward()
    self.d_optimizer.step()

    #=======================================================================================#
    #=================================== Gradient Penalty ==================================#
    #=======================================================================================#
    # Compute gradient penalty
    if not 'HINGE' in GAN_options:
      d_loss_gp = self._get_gradient_penalty(real_x0.data, fake_x0.data)
      d_loss = self.config.lambda_gp * d_loss_gp
      self.loss['Dgp'] = get_loss_value(d_loss)
      self.update_loss('Dgp', self.loss['Dgp'])
      self.reset_grad()
      d_loss.backward()
      self.d_optimizer.step()     

  #=======================================================================================#
  #=======================================================================================#    
  #=======================================================================================#
  #=======================================================================================#
  def train(self):

    # Fixed inputs and target domain labels for debugging
    GAN_options = self.config.GAN_options
    opt = torch.no_grad() if int(torch.__version__.split('.')[1])>3 else open('_null.txt', 'w')
    with opt:
      fixed_x = []
      fixed_label = []
      for i, (images, labels, files) in enumerate(self.data_loader):
        fixed_x.append(images)
        fixed_label.append(labels)
        if i == max(1,int(16/self.config.batch_size)):
          break
      fixed_x = torch.cat(fixed_x, dim=0)
      fixed_label = torch.cat(fixed_label, dim=0)
      if 'Stochastic' in GAN_options: style_fixed = self.G.random_style(fixed_x)
      else: style_fixed = None
    # lr cache for decaying
    g_lr = self.config.g_lr
    d_lr = self.config.d_lr

    # Start with trained model if exists
    if self.config.pretrained_model:
      start = int(self.config.pretrained_model.split('_')[0])
      for i in range(start):
        if (i+1) %self.config.num_epochs_decay==0:
          g_lr = (g_lr / 10.)
          d_lr = (d_lr / 10.)
          self.update_lr(g_lr, d_lr)
          self.PRINT ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))     
    else:
      start = 0

    # The number of iterations per epoch
    last_model_step = len(self.data_loader)

    self.PRINT("Current time: "+TimeNow())

    # Tensorboard log path
    if self.config.use_tensorboard: self.PRINT("Tensorboard Log path: "+self.config.log_path)
    self.PRINT("Debug Log txt: "+os.path.realpath(self.config.log.name))

    #RaGAN uses different data for Dis and Gen 
    batch_size = self.config.batch_size//2  if 'RaGAN' in GAN_options else self.config.batch_size

    # Log info
    Log = "---> batch size: {}, fold: {}, img: {}, GPU: {}, !{}, [{}]\n-> GAN_options:".format(\
        batch_size, self.config.fold, self.config.image_size, \
        self.config.GPU, self.config.mode_data, self.config.PLACE) 
    for item in sorted(GAN_options):
      Log += ' [*{}]'.format(item.upper())
    Log += ' [*{}]'.format(self.config.dataset_fake)
    if self.config.ALL_ATTR!=0: Log += ' [*ALL_ATTR={}]'.format(self.config.ALL_ATTR)
    if self.config.MultiDis: Log += ' [*MultiDisc={}]'.format(self.config.MultiDis)
    self.PRINT(Log)
    start_time = time.time()

    criterion_l1 = torch.nn.L1Loss()
    style_flag = True

    if self.config.HOROVOD:
      disable_tqdm = False#not hvd.rank() == 0
    else:
      disable_tqdm = False

    # Start training
    for e in range(start, self.config.num_epochs):
      E = str(e+1).zfill(3)
      self.D.train()
      self.G.train()
      self.LOSS = {}
      if self.config.HOROVOD: self.data_loader.sampler.set_epoch(e)
      desc_bar = 'Epoch: %d/%d'%(e,self.config.num_epochs)
      progress_bar = tqdm(enumerate(self.data_loader), unit_scale=True, 
          total=len(self.data_loader), desc=desc_bar, ncols=5, disable=disable_tqdm)
      for i, (real_x, real_c, files) in progress_bar: 

        self.loss = {}

        #=======================================================================================#
        #====================================== DATA2VAR =======================================#
        #=======================================================================================#
        # Convert tensor to variable
        real_x = to_var(real_x)
        real_c = to_var(real_c)       

        #RaGAN uses different data for Dis and Gen 
        if 'RaGAN' in GAN_options:
          split = lambda x: (x[:x.size(0)//2], x[x.size(0)//2:])
        else:
          split = lambda x: (x, x)

        real_x0, real_x1 = split(real_x)
        real_c0, real_c1 = split(real_c)          

        rand_idx1 = self.get_randperm(real_c1)
        fake_c1 = real_c1[rand_idx1]
        fake_c1 = to_var(fake_c1.data)

        self.Disc_update(real_x0, real_c0, GAN_options)        
        
        #=======================================================================================#
        #======================================= Train G =======================================#
        #=======================================================================================#
        if (i+1) % self.config.d_train_repeat == 0:

          ############################## Stochastic Part ##################################
          if 'Stochastic' in GAN_options:
            style_fake1 = to_var(self.G.random_style(real_x1))
            style_rec1 = to_var(self.G.random_style(real_x1))
            # style_rec1 = self.G.get_style(real_x1)
            if 'style_labels' in GAN_options:
              style_rec1 = style_rec1*real_c1.unsqueeze(-1)
              style_fake1 = style_fake1*fake_c1.unsqueeze(-1)     
          else:
            style_fake1 = style_rec1 = None

          fake_x1 = self.G(real_x1, fake_c1, stochastic = style_fake1, CONTENT='content_loss' in GAN_options)

          ## GAN LOSS
          # if self.config.dataset_fake=='RafD':
          #   g_loss_src, g_loss_cls, g_loss_cls_pose, style_disc = self._GAN_LOSS(fake_x1[0], real_x1, fake_c1, is_fake=True)
          #   g_loss_cls_pose = self.config.lambda_cls_pose * g_loss_cls_pose  
          #   self.loss['Gcls_p'] = get_loss_value(g_loss_cls_pose)          
          #   self.update_loss('Gcls_p', self.loss['Gcls_p'])      
          # else:
          g_loss_src, g_loss_cls, style_disc = self._GAN_LOSS(fake_x1[0], real_x1, fake_c1, is_fake=True)          

          g_loss_cls = g_loss_cls*self.config.lambda_cls
          self.loss['Gsrc'] = get_loss_value(g_loss_src)
          self.loss['Gcls'] = get_loss_value(g_loss_cls)
          self.update_loss('Gsrc', self.loss['Gsrc'])
          self.update_loss('Gcls', self.loss['Gcls'])

          ## REC LOSS
          rec_x1  = self.G(fake_x1[0], real_c1, stochastic = style_rec1, CONTENT='content_loss' in GAN_options) 
          if 'Perceptual' in GAN_options:
            g_loss_recp = self.config.lambda_perceptual*self._compute_vgg_loss(real_x1, rec_x1[0])     
            self.loss['Grecp'] = get_loss_value(g_loss_recp)
            self.update_loss('Grecp', self.loss['Grecp'])
            g_loss_rec = g_loss_recp

            g_loss_rec = 0.01*self.config.lambda_perceptual*self.config.lambda_rec*criterion_l1(rec_x1[0], real_x1)   
            self.loss['Grec'] = get_loss_value(g_loss_rec) 
            self.update_loss('Grec', self.loss['Grec'])
            g_loss_rec += g_loss_recp

          else:
            g_loss_rec = self.config.lambda_rec*criterion_l1(rec_x1[0], real_x1)
            self.loss['Grec'] = get_loss_value(g_loss_rec) 
            self.update_loss('Grec', self.loss['Grec'])            

            # if 'L1_LOSS' in GAN_options:
            #   g_loss_rec = self.config.lambda_rec*(criterion_l1(fake_x1[0], real_x1) + criterion_l1(rec_x1[0], fake_x1[0].detach()))
            #   self.loss['Grec'] = get_loss_value(g_loss_rec) 
            #   self.update_loss('Grec', self.loss['Grec']) 
            # else:
            #   g_loss_rec = self.config.lambda_rec*criterion_l1(rec_x1[0], real_x1)
            #   self.loss['Grec'] = get_loss_value(g_loss_rec) 
            #   self.update_loss('Grec', self.loss['Grec'])                  

          # Backward + Optimize
          g_loss = g_loss_src + g_loss_rec + g_loss_cls 


          ############################## Identity Part ###################################
          if 'Identity' in GAN_options:
            if 'STYLE_DISC' in GAN_options:
              style_identity = style_disc[0][1]            
            elif 'Stochastic' in GAN_options:
              style_identity = self.G.get_style(real_x1)
            else:
              style_identity = None
            # style_identity = to_var(self.G.random_style(real_x1))
            idt_x1  = self.G(real_x1, real_c1, stochastic = style_identity) 
            g_loss_idt = self.config.lambda_idt*criterion_l1(idt_x1[0], real_x1)
            self.loss['Gidt'] = get_loss_value(g_loss_idt)
            self.update_loss('Gidt', self.loss['Gidt'])    
            g_loss += g_loss_idt

          ############################## Background Consistency Part ###################################
          if 'L1_LOSS' in GAN_options:
            g_loss_rec1 = self.config.lambda_l1*(criterion_l1(fake_x1[0], real_x1) + criterion_l1(rec_x1[0], fake_x1[0].detach()))
            self.loss['Grec1'] = get_loss_value(g_loss_rec1)
            self.update_loss('Grec1', self.loss['Grec1'])    
            g_loss += g_loss_rec1

          ##############################      L1 Perceptual Part   ###################################
          if 'L1_Perceptual' in GAN_options:
            l1_perceptual = self._compute_vgg_loss(fake_x1[0], real_x1) + self._compute_vgg_loss(rec_x1[0], fake_x1[0])
            g_loss_recp1 = self.config.lambda_l1perceptual*l1_perceptual
            self.loss['Grecp1'] = get_loss_value(g_loss_recp1)
            self.update_loss('Grecp1', self.loss['Grecp1'])    
            g_loss += g_loss_recp1   

          ############################## Attention Part ###################################
          if 'Attention' in GAN_options or 'Attention2' in GAN_options or 'Attention3' in GAN_options:
            g_loss_mask = self.config.lambda_mask * (torch.mean(rec_x1[1]) + torch.mean(fake_x1[1]))
            g_loss_mask_smooth = self.config.lambda_mask_smooth * (_compute_loss_smooth(rec_x1[1]) + _compute_loss_smooth(fake_x1[1])) 
            self.loss['Gatm'] = get_loss_value(g_loss_mask)
            self.loss['Gats'] = get_loss_value(g_loss_mask_smooth)     
            self.update_loss('Gatm', self.loss['Gatm'])
            self.update_loss('Gats', self.loss['Gats']) 
            self.color(self.loss, 'Gatm', 'blue')
            g_loss += g_loss_mask + g_loss_mask_smooth  

          ############################## Content Part ###################################
          if 'content_loss' in GAN_options:
            g_loss_content = self.config.lambda_content * criterion_l1(rec_x1[-1], fake_x1[-1].detach())
            self.loss['Gcon'] = get_loss_value(g_loss_content)
            self.update_loss('Gcon', self.loss['Gcon'])       
            g_loss += g_loss_content      

          ############################## Stochastic Part ###################################
          if 'Stochastic' in GAN_options: 
            if 'STYLE_DISC' in GAN_options:
              _style_fake1 = style_disc[0][0]
            else:
              if 'AttentionStyle' in GAN_options: _style_fake1 = self.G.get_style(fake_x1[1])
              else: _style_fake1 = self.G.get_style(fake_x1[0])

            s_loss_style = (self.config.lambda_style) * self._criterion_style(_style_fake1, style_fake1)
            self.loss['Gsty'] = get_loss_value(s_loss_style)
            self.update_loss('Gsty', self.loss['Gsty'])
            # if self.loss['Gsty']>0.75 and e>6 and style_flag:
            #   send_mail(body='Gsty still in {}'.format(self.loss['Gsty']))
            #   style_flag = False
            g_loss += s_loss_style

            if 'rec_style' in GAN_options:
              if 'STYLE_DISC' in GAN_options:
                _style_rec1 = self.D(rec_x1[0])[-1][0]
              else:           
                if 'AttentionStyle' in GAN_options: _style_rec1 = self.G.get_style(rec_x1[1])
                else: _style_rec1 = self.G.get_style(rec_x1[0])
              s_loss_style_rec = (self.config.lambda_style) * self._criterion_style(_style_rec1, style_rec1.detach())
              self.loss['Gstyr'] = get_loss_value(s_loss_style_rec)
              self.update_loss('Gstyr', self.loss['Gstyr'])
              g_loss += s_loss_style_rec              

              if 'content_loss' in GAN_options:
                rec_content = self.G(rec_x1[0], real_c1, JUST_CONTENT=True)
                g_loss_rec_content = self.config.lambda_content * criterion_l1(rec_content, fake_x1[-1].detach())
                self.loss['Gconr'] = get_loss_value(g_loss_rec_content)
                self.update_loss('Gconr', self.loss['Gconr'])       
                g_loss += g_loss_rec_content           

            ############################## KL Part ###################################
            if 'kl_loss' in GAN_options:
              loss_kl = _compute_kl(_style_fake1)
              if 'rec_style' in GAN_options:  loss_kl += _compute_kl(_style_rec1)
              g_loss_kl = self.config.lambda_kl * loss_kl
              self.loss['Gkl'] = get_loss_value(g_loss_kl)
              self.update_loss('Gkl', self.loss['Gkl'])
              g_loss += g_loss_kl 


          self.reset_grad()
          g_loss.backward()
          if 'Stochastic' in GAN_options and 'Split_Optim' in GAN_options:
            self.s_optimizer.step()
          self.g_optimizer.step()          

        #=======================================================================================#
        #========================================MISCELANEOUS===================================#
        #=======================================================================================#
        # PRINT log info
        if (i+1) % self.config.log_step == 0 or (i+1)==last_model_step or i+e==0:
          progress_bar.set_postfix(**self.loss)
          if (i+1)==last_model_step: progress_bar.set_postfix('')
          if self.config.use_tensorboard:
            for tag, value in self.loss.items():
              self.logger.scalar_summary(tag, value, e * last_model_step + i + 1)

        # Save current fake
        if (i+1) % self.config.sample_step == 0 or (i+1)==last_model_step or i+e==0:
          name = os.path.join(self.config.sample_path, 'current_fake.jpg')
          self.save_fake_output(fixed_x, name, label=fixed_label, training=True, style_fixed=style_fixed)

      # Translate fixed images for debugging
      name = os.path.join(self.config.sample_path, '{}_{}_fake.jpg'.format(E, i+1))
      self.save_fake_output(fixed_x, name, label=fixed_label, training=True, style_fixed=style_fixed)

      self.save(E, i+1)
                 
      #Stats per epoch
      elapsed = time.time() - start_time
      elapsed = str(datetime.timedelta(seconds=elapsed))
      log = '--> %s | Elapsed (%d/%d) : %s | %s\nTrain'%(TimeNow(), e, self.config.num_epochs, elapsed, Log)
      for tag, value in sorted(self.LOSS.items()):
        log += ", {}: {:.4f}".format(tag, np.array(value).mean())   

      self.PRINT(log)
      self.data_loader.dataset.shuffle(e) #Shuffling dataset after each epoch

      # Decay learning rate     
      if (e+1) % self.config.num_epochs_decay ==0:
        g_lr = (g_lr / 10)
        d_lr = (d_lr / 10)
        self.update_lr(g_lr, d_lr)
        self.PRINT ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

  #=======================================================================================#
  #=======================================================================================#

  def save_fake_output(self, real_x, save_path, label=None, gif=False, output=False, training=False, Style=0, style_fixed=None, TIME=False):
    self.G.eval()  
    self.D.eval()
    Attention = 'Attention' in self.config.GAN_options or 'Attention2' in self.config.GAN_options or 'Attention3' in self.config.GAN_options
    Stochastic = 'Stochastic' in self.config.GAN_options
    n_rep = self.config.style_debug
    Output = []
    opt = torch.no_grad() if int(torch.__version__.split('.')[1])>3 else open('_null.txt', 'w')
    flag_time = True
    with opt:
      real_x = to_var(real_x, volatile=True)
      target_c_list = target_debug_list(real_x.size(0), self.config.c_dim, config=self.config)   

      # Start translations
      fake_image_list = [real_x]
      fake_attn_list  = []
      if Attention: 
        fake_attn_list = [to_var(denorm(real_x.data), volatile=True)]

      out_label = to_var(torch.zeros(real_x.size(0), self.config.c_dim), volatile=True)
      # if not self.config.NO_LABELCUM:
      if not self.config.dataset_fake=='RafD':
        if label is None:
          out_label = self._CLS(real_x)
        else:
          out_label = to_var(label, volatile=True)

      # Batch of images
      if Style==0:
        for k, target_c in enumerate(target_c_list):
          # target_c = torch.clamp(target_c+out_label,max=1)
          if self.config.dataset_fake=='CelebA' or self.config.dataset_fake=='EmotionNet':
            target_c = (out_label-target_c)**2 #Swap labels
          if Stochastic: 
            if style_fixed is None:
              style = to_var(self.G.random_style(real_x), volatile=True)
            else:
              style = to_var(style_fixed, volatile=True)

          else:
            style = None
          start_time = time.time()
          fake_x = self.G(real_x, target_c, stochastic=style)
          elapsed = time.time() - start_time
          elapsed = str(datetime.timedelta(seconds=elapsed))
          if TIME and flag_time: 
            print("Time/batch_size for one single forward generation (bs:{}): {}".format(real_x.size(0),elapsed))
            flag_time=False

          fake_image_list.append(fake_x[0])
          if Attention: fake_attn_list.append(fake_x[1].repeat(1,3,1,1))
        Output.extend(self._SAVE_IMAGE(save_path, fake_image_list, im_size=self.config.image_size, attn_list=fake_attn_list, gif=gif))

      #Same image different style
      for idx, real_x0 in enumerate(real_x):
        if training:
          _save_path = save_path
        else:
          _save_path = os.path.join(save_path.replace('.jpg', ''), '{}_{}.jpg'.format(Style, str(idx).zfill(3)))
          create_dir(_save_path)
        real_x0 = real_x0.repeat(n_rep,1,1,1)#.unsqueeze(0)
        _out_label = out_label[idx].repeat(n_rep,1)
        label_space = np.linspace(0,1,n_rep)

        fake_image_list = [real_x0]
        fake_attn_list  = []       
        if Attention: 
          fake_attn_list = [to_var(denorm(real_x0.data), volatile=True)]             
        for n_label, _target_c in enumerate(target_c_list):
          _target_c  = _target_c[0].repeat(n_rep,1)
          # target_c = torch.clamp(_target_c+_out_label, max=1)
          if self.config.dataset_fake=='CelebA' or self.config.dataset_fake=='EmotionNet':
            target_c = (_out_label-_target_c)**2 #Swap labels
          else: target_c = _target_c
          
          if Stochastic:
            style=to_var(self.G.random_style(real_x0), volatile=True)

            # Translate and translate back
            if Style==1:
              _style=to_var(self.G.random_style(real_x0), volatile=True)
              real_x0 = self.G(real_x0, target_c, stochastic=_style)[0]
              # ipdb.set_trace()
              if self.config.dataset_fake=='CelebA' or self.config.dataset_fake=='EmotionNet':
                target_c = (target_c-_target_c)**2

            #Style constant | progressive swap label
            elif Style==2: 
              for j, i in enumerate(range(real_x0.size(0))): 
                style[i] = style[0].clone()
                if n_label>0:
                  target_c[i][n_label-1].data.fill_((target_c[i][n_label-1]*label_space[j] + (1-target_c[i][n_label-1])*(1-label_space[j])).data[0])
                else:
                  target_c[i] = target_c[i]*label_space[j] + (1-target_c[i])*(1-label_space[j])

            #Style constant | progressive swap label
            elif Style==3: 
              for j, i in enumerate(range(real_x0.size(0))): 
                style[i] = style[1].clone()
                if n_label>0:
                  target_c[i][n_label-1].data.fill_((target_c[i][n_label-1]*label_space[j] + (1-target_c[i][n_label-1])*(1-label_space[j])).data[0])
                else:
                  target_c[i] = target_c[i]*label_space[j] + (1-target_c[i])*(1-label_space[j])

            #Style 0 | progressive swap label
            elif Style==4: 
              for j, i in enumerate(range(real_x0.size(0))): 
                style[i] = style[0]*0
                if n_label>0:
                  target_c[i][n_label-1].data.fill_((target_c[i][n_label-1]*label_space[j] + (1-target_c[i][n_label-1])*(1-label_space[j])).data[0])
                else:
                  target_c[i] = target_c[i]*label_space[j] + (1-target_c[i])*(1-label_space[j])          

            #Style constant | progressive label
            elif Style==5:
              for j, i in enumerate(range(real_x0.size(0))): 
                style[i] = style[2].clone()
                target_c[i] = _target_c[i]*0.2*j   

            #Style random | One label at a time
            elif Style==6: 
              target_c = _target_c

            #Extract style from the two before, current, and two after. 
            elif Style==7:
              _real_x0 = torch.zeros_like(real_x0)
              _range = range(-int(n_rep//2), 1+int(n_rep//2))
              if len(_range)>n_rep: _range.pop(0)
              for j, k in enumerate(_range):
                kk = (k+idx)%real_x.size(0) if k+idx >= real_x.size(0) else k+idx
                _real_x0[j] = real_x[kk]
              if 'STYLE_DISC' in self.config.GAN_options:
                style = self.D(_real_x0)[-1][0]
              else:
                style = self.G.get_style(_real_x0)
          else:
            style = None
            for j, i in enumerate(range(real_x0.size(0))): 
              target_c[i] = target_c[i]*label_space[j]            

          # ipdb.set_trace()
          fake_x = self.G(real_x0, target_c, stochastic=style)
          fake_image_list.append(fake_x[0])
          if Attention: fake_attn_list.append(fake_x[1].repeat(1,3,1,1))
        if 'Stochastic' in self.config.GAN_options: Output.extend(self._SAVE_IMAGE(_save_path, fake_image_list, attn_list=fake_attn_list, im_size=self.config.image_size, gif=gif, mode='style_'+chr(65+idx)))
        if idx==self.config.iter_style or (training and idx==self.config.style_train_debug-1) or 'Stochastic' not in self.config.GAN_options: break
    self.G.train()
    self.D.train()
    if output: return Output 
   
  #=======================================================================================#
  #=======================================================================================#

  def test(self, dataset='', load=False):
    import re
    from data_loader import get_loader
    last_name = self.resume_name()
    save_folder = os.path.join(self.config.sample_path, '{}_test'.format(last_name))
    if dataset=='': 
      dataset = self.config.dataset_fake
      data_loader_val = self.data_loader
    else:
      data_loader_val = get_loader(self.config.metadata_path, self.config.image_size, self.config.batch_size, shuffling=True, dataset=dataset, mode='test') 
    for i, (real_x, org_c, files) in enumerate(data_loader_val):
      create_dir(save_folder)
      save_path = os.path.join(save_folder, '{}_{}_{}.jpg'.format(dataset, i+1, '{}'))
      string = '{}'
      if self.config.NO_LABELCUM:
        string += '_{}'.format('NO_Label_Cum','{}')
      string = string.format(TimeNow_str())
      name = os.path.abspath(save_path.format(string))
      if 'Stochastic' in self.config.GAN_options:
        _debug = self.config.style_label_debug+1
      else:
        _debug = 1
      if self.config.dataset_fake==self.config.dataset_real:
        label = org_c
      else:
        label = None
      self.PRINT('Translated test images and saved into "{}"..!'.format(name))
      for k in range(_debug):
        output = self.save_fake_output(real_x, name, label=label, output=True, Style=k, TIME=not i)
        # send_mail(body='Images from '+self.config.sample_path, attach=output)
      if i==self.config.iter_test-1: break   

  #=======================================================================================#
  #=======================================================================================#

  def DEMO(self, path):
    from data_loader import get_loader
    import re
    last_name = self.resume_name()
    save_folder = os.path.join(self.config.sample_path, '{}_test'.format(last_name))
    batch_size = self.config.batch_size if not 'Stochastic' in self.config.GAN_options else 1
    data_loader = get_loader(path, self.config.image_size, batch_size, shuffling = False, dataset='DEMO', mode='test')
    for i, real_x in enumerate(data_loader):
      save_path = os.path.join(save_folder, 'DEMO_{}_{}.jpg'.format(i+1, TimeNow_str()))
      if 'Stochastic' in self.config.GAN_options:
        _debug = min(self.config.style_label_debug+1, 7)
      else:
        _debug = 1
      self.PRINT('Translated test images and saved into "{}"..!'.format(save_path))
      for k in range(_debug):
        output = self.save_fake_output(real_x, save_path, output=True, gif=True, Style=k, TIME=not i)    
        # send_mail(body='Images from '+self.config.sample_path, attach=output)
