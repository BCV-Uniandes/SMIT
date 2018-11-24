import torch, os, time, warnings, datetime
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as cfg
from tqdm import tqdm
from misc.utils import get_labels, get_loss_value, PRINT, target_debug_list, TimeNow, TimeNow_str, to_var
from misc.losses import _compute_kl, _compute_loss_smooth, _compute_vgg_loss, _GAN_LOSS, _get_gradient_penalty
import torch.utils.data.distributed
from misc.utils import _horovod
hvd = _horovod()

warnings.filterwarnings('ignore')

from solver import Solver
class Train(Solver):
  def __init__(self, config, data_loader):
    super(Train, self).__init__(config, data_loader)
    self.run()

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
    if 'Stochastic' in self.config.GAN_options and 'Split_Optim' in self.config.GAN_options and self.config.lambda_style!=0:
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
  def get_labels(self):
    return get_labels(self.config.image_size, self.config.dataset_fake, attr=self.data_loader.dataset)

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
  def debug_vars(self):
    opt = torch.no_grad() if int(torch.__version__.split('.')[1])>3 else open('_null.txt', 'w')
    with opt:
      fixed_x = []
      fixed_label = []
      for i, (images, labels, files) in enumerate(self.data_loader):
        # if self.config.dataset_fake=='Image2Edges': images = images[:,:,:,:256]
        fixed_x.append(images)
        fixed_label.append(labels)
        if i == max(1,int(16/self.config.batch_size)):
          break
      fixed_x = torch.cat(fixed_x, dim=0)
      fixed_label = torch.cat(fixed_label, dim=0)
      if 'Stochastic' in self.config.GAN_options: fixed_style = self.G.random_style(fixed_x)
      else: fixed_style = None    
    return fixed_x, fixed_label, fixed_style    

  #=======================================================================================#
  #=======================================================================================#
  def _criterion_style(self, output_style, target_style):
    criterion_style = torch.nn.MSELoss() if 'mse_style' in self.config.GAN_options else torch.nn.L1Loss()
    return criterion_style(output_style, target_style)

  #=======================================================================================#
  #=======================================================================================#
  def _compute_vgg_loss(self, data_x, data_y):
    return _compute_vgg_loss(self.vgg, data_x, data_y, IN=not 'NoPerceptualIN' in self.config.GAN_options)

  #=======================================================================================#
  #=======================================================================================#
  def _GAN_LOSS(self, real_x, fake_x, label, is_fake=False):
    cross_entropy = self.config.dataset_fake in ['painters_14', 'Animals', 'Image2Weather', 'Image2Season', 'Image2Edges', 'Yosemite']
    cross_entropy = cross_entropy or (self.config.dataset_fake=='RafD' and self.config.RafD_FRONTAL)
    cross_entropy = cross_entropy or (self.config.dataset_fake=='RafD' and self.config.RafD_EMOTIONS)
    if cross_entropy:
      label = torch.max(label, dim=1)[1]
    return _GAN_LOSS(self.D, real_x, fake_x, label, self.config.GAN_options, is_fake=is_fake, cross_entropy=cross_entropy)

  #=======================================================================================#
  #=======================================================================================#
  def _get_gradient_penalty(self, real_x, fake_x):
    return _get_gradient_penalty(self.D, real_x, fake_x)

  #=======================================================================================#
  #=======================================================================================#
  def Disc_update(self, real_x0, real_c0, GAN_options):

    rand_idx0 = self.get_randperm(real_c0)
    if self.config.c_dim==2:
      # fake_c0 = 1-real_c0
      fake_c0 = real_c0[rand_idx0]
    else:
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

  def run(self):

    GAN_options = self.config.GAN_options

    # lr cache for decaying
    g_lr = self.config.g_lr
    d_lr = self.config.d_lr
    #self.g_optimizer.param_groups[0]['lr']
    # self.update_lr(g_lr, d_lr)
    self.PRINT ('Training with learning rate g_lr: {}, d_lr: {}.'.format(self.g_optimizer.param_groups[0]['lr'], self.d_optimizer.param_groups[0]['lr']))  

    # Start with trained model if exists
    if self.config.pretrained_model:
      start = int(self.config.pretrained_model.split('_')[0])+1
      _iter = start*int(self.config.pretrained_model.split('_')[1])
      for e in range(start):
        if e!=0 and e%self.config.num_epochs_decay==0:
        # if e >= self.config.num_epochs_decay:
          # g_lr -= (self.config.g_lr / float(self.config.num_epochs - self.config.num_epochs_decay))
          # d_lr -= (self.config.d_lr / float(self.config.num_epochs - self.config.num_epochs_decay))
          g_lr = g_lr / 10.
          d_lr = d_lr / 10.
          self.update_lr(g_lr, d_lr)
          self.PRINT ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))            
    else:
      start = 0
      _iter = 0

    # The number of iterations per epoch
    last_model_step = len(self.data_loader)

    # Fixed inputs, target domain labels, and style for debugging
    fixed_x, fixed_label, fixed_style = self.debug_vars()
    if start==0:
      name = os.path.join(self.config.sample_path, 
        '{}_{}_fake.jpg'.format(str(0).zfill(4), str(0).zfill(len(str(last_model_step)))))
      if 'Stochastic' in GAN_options: self.save_fake_output(fixed_x, name, label=fixed_label, training=True, fixed_style=fixed_style)
      self.save_fake_output(fixed_x, name, label=fixed_label, training=True)

    self.PRINT("Current time: "+TimeNow())

    # Tensorboard log path
    if self.config.use_tensorboard: self.PRINT("Tensorboard Log path: "+self.config.log_path)
    self.PRINT("Debug Log txt: "+os.path.realpath(self.config.log.name))

    #RaGAN uses different data for Dis and Gen 
    batch_size = self.config.batch_size//2  if 'RaGAN' in GAN_options else self.config.batch_size

    # Log info
    Log = self.PRINT_LOG(batch_size)
    start_time = time.time()

    criterion_l1 = torch.nn.L1Loss()

    if self.config.HOROVOD:
      disable_tqdm = False#not hvd.rank() == 0
    else:
      disable_tqdm = False

    # Start training
    for e in range(start, self.config.num_epochs):
      # disable_tqdm =  e%self.config.save_epoch==0

      E = str(e).zfill(4)

      self.D.train()
      self.G.train()
      self.LOSS = {}
      if self.config.HOROVOD: self.data_loader.sampler.set_epoch(e)
      desc_bar = '[Iter: %d] Epoch: %d/%d'%(_iter, e,self.config.num_epochs)
      progress_bar = tqdm(enumerate(self.data_loader), unit_scale=True, 
          total=len(self.data_loader), desc=desc_bar, ncols=5, disable=disable_tqdm)
      for i, (real_x, real_c, files) in progress_bar: 
        self.loss = {}
        _iter+=1
        #RaGAN uses different data for Dis and Gen 
        if 'RaGAN' in GAN_options:
          try: 
            split = lambda x: (x[:x.size(0)//2], x[x.size(0)//2:])
            real_x0, real_x1 = split(real_x)
            real_c0, real_c1 = split(real_c)              
          except ValueError: 
            split = lambda x: (x, x)
            real_x0, real_x1 = split(real_x)
            real_c0, real_c1 = split(real_c)              
        else:
          split = lambda x: (x, x)   
          real_x0, real_x1 = split(real_x)
          real_c0, real_c1 = split(real_c)                 

        #=======================================================================================#
        #====================================== DATA2VAR =======================================# 
        #=======================================================================================#
        # Convert tensor to variable
        real_x0 = to_var(real_x0)
        real_c0 = to_var(real_c0)    

        real_x1 = to_var(real_x1)
        real_c1 = to_var(real_c1)            

        rand_idx1 = self.get_randperm(real_c1)
        if self.config.c_dim==2:
          # fake_c1 = 1-real_c1
          fake_c1 = real_c1[rand_idx1] 
        else:
          fake_c1 = real_c1[rand_idx1]        
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
            if 'ORG_REC' in GAN_options:
              style_rec1 = self.G.get_style(real_x1)
            else:
              style_rec1 = to_var(self.G.random_style(real_x1))
            # style_rec1 = self.G.get_style(real_x1)
            if 'style_labels' in GAN_options:
              style_rec1 = style_rec1*real_c1.unsqueeze(-1)
              style_fake1 = style_fake1*fake_c1.unsqueeze(-1)     
          else:
            style_fake1 = style_rec1 = None

          fake_x1 = self.G(real_x1, fake_c1, stochastic = style_fake1, CONTENT='content_loss' in GAN_options)

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

            # g_loss_rec = 0.01*self.config.lambda_perceptual*self.config.lambda_rec*criterion_l1(rec_x1[0], real_x1)   
            # self.loss['Grec'] = get_loss_value(g_loss_rec) 
            # self.update_loss('Grec', self.loss['Grec'])
            # g_loss_rec += g_loss_recp

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
              if self.config.lambda_style==0: style_identity = to_var(self.G.random_style(real_x1))
              else: style_identity = self.G.get_style(real_x1)
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
          if 'Stochastic' in GAN_options and self.config.lambda_style!=0: 
            if 'STYLE_DISC' in GAN_options:
              _style_fake1 = style_disc[0][0]
            else:
              if 'AttentionStyle' in GAN_options: _style_fake1 = self.G.get_style(fake_x1[1])
              else: _style_fake1 = self.G.get_style(fake_x1[0])

            s_loss_style = (self.config.lambda_style) * self._criterion_style(_style_fake1, style_fake1)
            self.loss['Gsty'] = get_loss_value(s_loss_style)
            self.update_loss('Gsty', self.loss['Gsty'])
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
          if 'Stochastic' in GAN_options and 'Split_Optim' in GAN_options and self.config.lambda_style!=0:
            self.s_optimizer.step()
          self.g_optimizer.step()          

        #===================================DEBUG====================================#
        # PRINT log info
        if (i+1) % self.config.log_step == 0 or (i+1)==last_model_step or i+e==0:
          progress_bar.set_postfix(**self.loss)
          if (i+1)==last_model_step: progress_bar.set_postfix('')
          if self.config.use_tensorboard:
            for tag, value in self.loss.items():
              self.logger.scalar_summary(tag, value, e * last_model_step + i + 1)
        # Save current fake
        if ((i+1) % self.config.sample_step == 0 or (i+1)==last_model_step or i+e==0) and self.config.image_size<=128:
          name = os.path.join(self.config.sample_path, 'current_fake.jpg')
          if 'Stochastic' in GAN_options: self.save_fake_output(fixed_x, name, label=fixed_label, training=True, fixed_style=fixed_style)
          self.save_fake_output(fixed_x, name, label=fixed_label, training=True)

      #=======================================================================================#
      #========================================MISCELANEOUS===================================#
      #=======================================================================================#

      self.data_loader.dataset.shuffle(e) #Shuffling dataset each epoch

      if e%self.config.save_epoch==0:
        # Save Weights
        self.save(E, i+1)

        # Save Translation
        name = os.path.join(self.config.sample_path, '{}_{}_fake.jpg'.format(E, i+1))
        if 'Stochastic' in GAN_options: self.save_fake_output(fixed_x, name, label=fixed_label, training=True, fixed_style=fixed_style)
        self.save_fake_output(fixed_x, name, label=fixed_label, training=True)
                   
        # Debug INFO
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))
        log = '--> %s | Elapsed [Iter: %d] (%d/%d) : %s | %s\nTrain'%(TimeNow(), _iter, e, self.config.num_epochs, elapsed, Log)
        for tag, value in sorted(self.LOSS.items()):
          log += ", {}: {:.4f}".format(tag, np.array(value).mean())   
        self.PRINT(log)
        if self.config.PLACE!='ETHZ': self.PLOT(e)

      # Decay learning rate     
      if e!=0 and e%self.config.num_epochs_decay==0:
      # if e >= self.config.num_epochs_decay:
        # g_lr -= (self.config.g_lr / float(self.config.num_epochs - self.config.num_epochs_decay))
        # d_lr -= (self.config.d_lr / float(self.config.num_epochs - self.config.num_epochs_decay))
        g_lr = g_lr / 10.
        d_lr = d_lr / 10.
        self.update_lr(g_lr, d_lr)
        self.PRINT ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))