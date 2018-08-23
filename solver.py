import torch, os, time, ipdb, glob, math, warnings, datetime
# import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import config as cfg
from tqdm import tqdm
from misc.utils import denorm, get_aus, get_loss_value, PRINT, TimeNow, TimeNow_str, to_cuda, to_var
from misc.losses import _compute_kl, _compute_loss_smooth, _GAN_LOSS, _get_gradient_penalty
warnings.filterwarnings('ignore')

class Solver(object):

  def __init__(self, data_loader, config):
    # Data loader
    self.data_loader = data_loader
    self.config = config

    # Build tensorboard if use
    self.build_model()
    if self.config.use_tensorboard:
      self.build_tensorboard()

    # Start with trained model
    if self.config.pretrained_model:
      self.load_pretrained_model()

  #=======================================================================================#
  #=======================================================================================#
  def build_model(self):
    # Define a generator and a discriminator
    from model import Discriminator
    if 'AdaIn' in self.config.GAN_options:
      if 'DRIT' not in self.config.GAN_options: from model import AdaInGEN as GEN
      else: from model import DRITGEN as GEN
    else: from model import Generator as GEN
    self.G = GEN(self.config, debug=True)
    G_parameters = filter(lambda p: p.requires_grad, self.G.parameters())
    self.g_optimizer = torch.optim.Adam(G_parameters, self.config.g_lr, [self.config.beta1, self.config.beta2])
    to_cuda(self.G)

    if self.config.mode=='train': 
      self.D = Discriminator(self.config, debug=True) 
      D_parameters = filter(lambda p: p.requires_grad, self.D.parameters())
      self.d_optimizer = torch.optim.Adam(D_parameters, self.config.d_lr, [self.config.beta1, self.config.beta2])
      to_cuda(self.D)
      self.print_network(self.D, 'Discriminator')
    self.print_network(self.G, 'Generator')

  #=======================================================================================#
  #=======================================================================================#

  def print_network(self, model, name):
    
    if 'AdaIn' in self.config.GAN_options and name=='Generator':
      choices = ['generator', 'enc_style', 'adain_net']
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
    name = os.path.join(self.config.model_save_path, '{}_{}_{}.pth'.format(Epoch, iter, '{}'))
    torch.save(self.G.state_dict(), name.format('G'))
    torch.save(self.D.state_dict(), name.format('D'))
    if int(Epoch)>2:
      name_1 = os.path.join(self.config.model_save_path, '{}_{}_{}.pth'.format(str(int(Epoch)-1).zfill(3), iter, '{}'))
      os.remove(name_1.format('G'))
      os.remove(name_1.format('D'))

  #=======================================================================================#
  #=======================================================================================#
  def load_pretrained_model(self):
    name = os.path.join(self.config.model_save_path, '{}_{}.pth'.format(self.config.pretrained_model, '{}'))
    self.G.load_state_dict(torch.load(name.format('G')))#, map_location=lambda storage, loc: storage))
    if self.config.mode=='train':  
      self.D.load_state_dict(torch.load(name.format('D')))#, map_location=lambda storage, loc: storage))
    self.PRINT('loaded trained models (step: {})..!'.format(self.config.pretrained_model))

  #=======================================================================================#
  #=======================================================================================#
  def build_tensorboard(self):
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
  def update_loss(self, loss, value):
    try:
      self.LOSS[loss].append(value)
    except:
      self.LOSS[loss] = []
      self.LOSS[loss].append(value)

  #=======================================================================================#
  #=======================================================================================#
  def get_aus(self):
    return get_aus(self.config.image_size, self.config.dataset_fake)

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
    PRINT(self.config.log, str)

  #=======================================================================================#
  #=======================================================================================#
  def _GAN_LOSS(self, real_x, fake_x, label, GEN=False):
    return _GAN_LOSS(self.D, real_x, fake_x, label, self.config.GAN_options, GEN=GEN)

  #=======================================================================================#
  #=======================================================================================#
  def _get_gradient_penalty(self, real_x, fake_x):
    return _get_gradient_penalty(self.D, real_x, fake_x)

  #=======================================================================================#
  #=======================================================================================#    
  #=======================================================================================#
  #=======================================================================================#
  def train(self):

    # Fixed inputs and target domain labels for debugging
    opt = torch.no_grad() if int(torch.__version__.split('.')[1])>3 else open('_null.txt', 'w')
    with opt:
      fixed_x = []
      for i, (images, labels, files) in enumerate(self.data_loader):
        fixed_x.append(images)
        if i == max(1,int(16/self.config.batch_size)):
          break
      fixed_x = torch.cat(fixed_x, dim=0)
    
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
    GAN_options = self.config.GAN_options

    self.PRINT("Current time: "+TimeNow())

    # Tensorboard log path
    self.PRINT("Log path: "+self.config.log_path)

    # Log info
    Log = "---> batch size: {}, fold: {}, img: {}, GPU: {}, !{}, [{}]\n-> GAN_options:".format(\
        self.config.batch_size, self.config.fold, self.config.image_size, \
        self.config.GPU, self.config.mode_data, self.config.PLACE) 
    for item in sorted(GAN_options):
      Log += ' [*{}]'.format(item.upper())
    Log += ' [*{}]'.format(self.config.dataset_fake)
    self.PRINT(Log)
    start_time = time.time()

    # Start training
    for e in range(start, self.config.num_epochs):
      E = str(e+1).zfill(3)
      self.D.train()
      self.G.train()
      self.LOSS = {}
      desc_bar = 'Epoch: %d/%d'%(e,self.config.num_epochs)
      progress_bar = tqdm(enumerate(self.data_loader), \
          total=len(self.data_loader), desc=desc_bar, ncols=5)
      for i, (real_x, real_c, files) in progress_bar: 

        loss = {}

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

        # Generat fake labels randomly (target domain labels)
        rand_idx0 = self.get_randperm(real_c0)
        fake_c0 = real_c0[rand_idx0]

        rand_idx1 = self.get_randperm(real_c1)
        fake_c1 = real_c1[rand_idx1]

        fake_c0 = to_var(fake_c0.data)
        fake_c1 = to_var(fake_c1.data)

          ############################## Stochastic Part ##################################
        if 'Stochastic' in GAN_options:
          style_real0 = self.G.get_style(real_x0)
          style_fake0 = [s[rand_idx0] for s in style_real0]
          if 'style_labels' in GAN_options:
            style_real0 = [s*real_c0.unsqueeze(2) for s in style_real0]
            style_fake0 = [s*fake_c0.unsqueeze(2) for s in style_fake0]
        else:
          style_real0 = style_fake0 = None

        fake_x0 = self.G(real_x0, fake_c0, stochastic=style_fake0[0])[0]

        #=======================================================================================#
        #======================================== Train D ======================================#
        #=======================================================================================#
        d_loss_src, d_loss_cls = self._GAN_LOSS(real_x0, fake_x0, real_c0)
        d_loss_cls = self.config.lambda_cls * d_loss_cls  

        # Backward + Optimize       
        d_loss = d_loss_src + d_loss_cls

        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

        loss['Dsrc'] = get_loss_value(d_loss_src)
        loss['Dcls'] = get_loss_value(d_loss_cls)          
        self.update_loss('Dsrc', loss['Dsrc'])
        self.update_loss('Dcls', loss['Dcls'])

        if 'kl_loss' in GAN_options:
          style_random0 = to_var(self.G.random_style(real_x0))
          if 'style_labels' in GAN_options:
            style_random0 *= fake_c0.unsqueeze(2)    
          fake_x0_random = self.G(real_x0, fake_c0, stochastic=style_random0)[0]
          d_loss_src, d_loss_cls = self._GAN_LOSS(real_x0, fake_x0_random, real_c0)
          d_loss_cls = self.config.lambda_cls * d_loss_cls  
          d_loss = d_loss_src + d_loss_cls
          self.reset_grad()
          d_loss.backward()
          self.d_optimizer.step()
          loss['Dsrcr'] = get_loss_value(d_loss_src)
          loss['Dclsr'] = get_loss_value(d_loss_cls)          
          self.update_loss('Dsrcr', loss['Dsrcr'])
          self.update_loss('Dclsr', loss['Dclsr'])            

        #=======================================================================================#
        #=================================== Gradient Penalty ==================================#
        #=======================================================================================#
        # Compute gradient penalty
        if not 'HINGE' in GAN_options:
          d_loss_gp = self._get_gradient_penalty(real_x0.data, fake_x0.data)
          d_loss = self.config.lambda_gp * d_loss_gp
          loss['Dgp'] = get_loss_value(d_loss)
          self.update_loss('Dgp', loss['Dgp'])
          self.reset_grad()
          d_loss.backward()
          self.d_optimizer.step()

          if 'kl_loss' in GAN_options:
            d_loss_gp = self._get_gradient_penalty(real_x0.data, fake_x0_random.data)
            d_loss = self.config.lambda_gp * d_loss_gp
            loss['Dgpr'] = get_loss_value(d_loss)
            self.update_loss('Dgpr', loss['Dgpr'])  
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()              
        
        #=======================================================================================#
        #======================================= Train G =======================================#
        #=======================================================================================#
        if (i+1) % self.config.d_train_repeat == 0:

          # Original-to-target and target-to-original domain

          ############################## Stochastic Part ##################################
          if 'Stochastic' in GAN_options:
            style_real1 = self.G.get_style(real_x1)
            style_fake1 = [s[rand_idx1] for s in style_real1]
            if 'style_labels' in GAN_options:
              style_real1 = [s*real_c1.unsqueeze(2) for s in style_real1]
              style_fake1 = [s*fake_c1.unsqueeze(2) for s in style_fake1]
          else:
            style_real1 = style_fake1 = None

          fake_x1 = self.G(real_x1, fake_c1, stochastic = style_fake1[0], CONTENT='content_loss' in GAN_options)
          rec_x1  = self.G(fake_x1[0], real_c1, stochastic = style_real1[0], CONTENT='content_loss' in GAN_options) 

          ## GAN LOSS
          g_loss_src, g_loss_cls = self._GAN_LOSS(fake_x1[0], real_x1, fake_c1, GEN=True)

          ## REC LOSS
          if 'L1_LOSS' in GAN_options:
            g_loss_rec = F.l1_loss(real_x1, fake_x1[0]) + \
                         F.l1_loss(fake_x1[0], rec_x1[0])         
          else:
            g_loss_rec = F.l1_loss(real_x1, rec_x1[0])

          g_loss_rec = g_loss_rec*self.config.lambda_rec
          g_loss_cls = g_loss_cls*self.config.lambda_cls

          loss['Gsrc'] = get_loss_value(g_loss_src)
          loss['Grec'] = get_loss_value(g_loss_rec)
          loss['Gcls'] = get_loss_value(g_loss_cls)

          self.update_loss('Gsrc', loss['Gsrc'])
          self.update_loss('Grec', loss['Grec'])
          self.update_loss('Gcls', loss['Gcls'])

          # Backward + Optimize
          g_loss = g_loss_src + g_loss_rec + g_loss_cls 

          ############################## Attention Part ###################################
          if 'Attention' in GAN_options:

            g_loss_mask = self.config.lambda_mask * (torch.mean(rec_x1[1]) + torch.mean(fake_x1[1]))
            g_loss_mask_smooth = self.config.lambda_mask_smooth * (_compute_loss_smooth(rec_x1[1]) + _compute_loss_smooth(fake_x1[1])) 

            loss['Gatm'] = get_loss_value(g_loss_mask)
            loss['Gats'] = get_loss_value(g_loss_mask_smooth)     
            self.update_loss('Gatm', loss['Gatm'])
            self.update_loss('Gats', loss['Gats'])
            g_loss += g_loss_mask + g_loss_mask_smooth

          ############################## KL Part ###################################
          if 'kl_loss' in GAN_options:
            g_loss_kl = self.config.lambda_kl * (_compute_kl(style_real1))
            loss['Gkl'] = get_loss_value(g_loss_kl)
            self.update_loss('Gkl', loss['Gkl'])
            g_loss += g_loss_kl

          ############################## Content Part ###################################
          if 'content_loss' in GAN_options:
            g_loss_content = self.config.lambda_content * F.l1_loss(fake_x1[-1], rec_x1[-1])
            loss['Gcon'] = get_loss_value(g_loss_content)
            self.update_loss('Gcon', loss['Gcon'])       
            g_loss += g_loss_content                      

          ############################## Stochastic Part ###################################
          if 'Stochastic' in GAN_options: 

            _style_fake1 = self.G.get_style(fake_x1[0])
            _style_rec1 = self.G.get_style(rec_x1[0])
            if 'style_labels' in GAN_options:
              _style_fake1 = [s*fake_c1.unsqueeze(2) for s in _style_fake1]
              _style_rec1 = [s*real_c1.unsqueeze(2) for s in _style_rec1]

            g_loss_style = (self.config.lambda_style/10) * (F.l1_loss(style_real1[0], _style_rec1[0]) +
                                                       F.l1_loss(_style_fake1[0], style_fake1[0]))
            loss['Gsty'] = get_loss_value(g_loss_style)
            self.update_loss('Gsty', loss['Gsty'])
            g_loss += g_loss_style

            if 'kl_loss' in GAN_options:

              self.reset_grad()
              # g_loss.backward(retain_graph=True)
              g_loss.backward()
              self.g_optimizer.step()

              style_real1 = self.G.get_style(real_x1)
              style_random1 = to_var(self.G.random_style(real_x1))

              if 'style_labels' in GAN_options:
                style_random1 *= fake_c1.unsqueeze(2)              
              fake_x1_random = self.G(real_x1, fake_c1, stochastic = style_random1)
              rec_x1_random  = self.G(fake_x1_random[0], real_c1, stochastic = style_real1[0]) 

              _style_fake_random1 = self.G.get_style(fake_x1_random[0])
              # _style_rec_random1 = self.G.get_style(rec_x1_random[0])                

              if 'style_labels' in GAN_options:
                _style_fake_random1 = [s*fake_c1.unsqueeze(2) for s in _style_fake_random1]

              mu_index = 1 if 'LOGVAR' in GAN_options else 0
              g_loss_style_random = self.config.lambda_style * (
                                      F.l1_loss(_style_fake_random1[mu_index], style_random1) 
                                      )
              g_loss_src_random, g_loss_cls_random = self._GAN_LOSS(fake_x1_random[0], real_x1, fake_c1, GEN=True)
              g_loss_cls_random = g_loss_cls_random*self.config.lambda_cls

              if 'L1_LOSS' in GAN_options:
                g_loss_rec_random = F.l1_loss(real_x1, fake_x1_random[0]) + F.l1_loss(fake_x1_random[0], rec_x1_random[0])
              else:
                g_loss_rec_random = F.l1_loss(real_x1, rec_x1_random[0])
              g_loss_rec_random = self.config.lambda_rec * g_loss_rec_random

              loss['Gsrcr'] = get_loss_value(g_loss_src_random)
              loss['Gclsr'] = get_loss_value(g_loss_cls_random)
              loss['Grecr'] = get_loss_value(g_loss_rec_random)
              loss['Gstyr'] = get_loss_value(g_loss_style_random)
              self.update_loss('Grecr', loss['Grecr'])       
              self.update_loss('Gstyr', loss['Gstyr'])
              self.update_loss('Gsrcr', loss['Gsrcr'])
              self.update_loss('Gclsr', loss['Gclsr'])
              g_loss = g_loss_src_random + g_loss_cls_random + g_loss_rec_random + g_loss_style_random

              if 'Attention' in GAN_options:

                g_loss_mask_random = self.config.lambda_mask * (
                          torch.mean(fake_x1_random[1]) +
                          torch.mean(rec_x1_random[1])
                          )
                g_loss_mask_smooth_random = self.config.lambda_mask_smooth * (
                        _compute_loss_smooth(fake_x1_random[1]) +
                        _compute_loss_smooth(rec_x1_random[1])
                        )     

                loss['Gatmr'] = get_loss_value(g_loss_mask_random)
                loss['Gatsr'] = get_loss_value(g_loss_mask_smooth_random)     
                self.update_loss('Gatmr', loss['Gatmr'])
                self.update_loss('Gatsr', loss['Gatsr'])    

                g_loss += g_loss_mask_random + g_loss_mask_smooth_random

          self.reset_grad()
          g_loss.backward()
          self.g_optimizer.step()

        #=======================================================================================#
        #========================================MISCELANEOUS===================================#
        #=======================================================================================#

        # PRINT log info
        if (i+1) % self.config.log_step == 0 or (i+1)==last_model_step or i+e==0:
          progress_bar.set_postfix(**loss)
          if (i+1)==last_model_step: progress_bar.set_postfix('')
          if self.config.use_tensorboard:
            for tag, value in loss.items():
              self.logger.scalar_summary(tag, value, e * last_model_step + i + 1)

        # Save current fake
        if (i+1) % (self.config.log_step*2) == 0 or (i+1)==last_model_step or i+e==0:
          name = os.path.join(self.config.sample_path, 'current_fake.jpg')
          self.save_fake_output(fixed_x, name)

        # Translate fixed images for debugging
        if (i+1) % self.config.sample_step == 0 or (i+1)==last_model_step or i+e==0:
          name = os.path.join(self.config.sample_path, '{}_{}_fake.jpg'.format(E, i+1))
          self.save_fake_output(fixed_x, name)

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
    opt = torch.no_grad() if int(torch.__version__.split('.')[1])>3 else open('_null.txt', 'w')
    with opt:
      real_x = to_var(real_x, volatile=True)
      target_c= torch.from_numpy(np.zeros((real_x.size(0), self.config.c_dim), dtype=np.float32))

      target_c_list = [to_var(target_c, volatile=True)]

      for j in range(self.config.c_dim):
        if label is None:
          target_c[:]=0 
          target_c[:,j]=1       
        else:
          print(j+1, sorted(list(set(map(int, (self.config.AUs['EMOTIONNET']*label[j].cpu().numpy()).tolist())))))
          target_c=label[j].repeat(label.size(0),1)      
        target_c_list.append(to_var(target_c, volatile=True))      

      # Start translations
      fake_image_list = [real_x]
      if Stochastic: 
        n_rep = 7
        n_img = 5
        real_x0 = real_x[n_img].repeat(n_rep,1,1,1)
        fake_rand_list = [real_x0]
        if Attention: 
          rand_attn_list = [to_var(denorm(real_x0.data), volatile=True)]
      if Attention: 
        fake_attn_list = [to_var(denorm(real_x.data), volatile=True)]
      for target_c in target_c_list:
        if Stochastic: 
          style = self.G.get_style(real_x, volatile=True)
          style = style[0]
          style_rand=to_var(self.G.random_style(real_x0), volatile=True)
        else: 
          style = None   

        if Stochastic:
          fake_x = self.G(real_x, target_c, stochastic=style)
          target_c0 = target_c[:n_rep]
                  
          style_rand[0] = style[n_img] #Compare with the original style
          style_rand[1] = style[6]
          style_rand[2] = style[9]     
          if 'mono_style' not in self.config.GAN_options:
            style_rand[3] = style_rand[3]*target_c0[0].unsqueeze(1)
            style_rand[4] = style_rand[4]*target_c0[0].unsqueeze(1)
          fake_x0 = self.G(real_x0, target_c0, stochastic=style_rand)   
          fake_rand_list.append(fake_x0[0])    
          if Attention: rand_attn_list.append(fake_x0[1].repeat(1,3,1,1))          
        else:
          fake_x = self.G(real_x, target_c)
        fake_image_list.append(fake_x[0])
        if Attention: fake_attn_list.append(fake_x[1].repeat(1,3,1,1))

      fake_images = denorm(torch.cat(fake_image_list, dim=3).data.cpu())
      fake_images = torch.cat((self.get_aus(), fake_images), dim=0)
      save_image(fake_images, save_path, nrow=1, padding=0)
      if Attention: 
        fake_attn = torch.cat(fake_attn_list, dim=3).data.cpu()
        fake_attn = torch.cat((self.get_aus(), fake_attn), dim=0)
        save_image(fake_attn, save_path.replace('fake', 'attn'), nrow=1, padding=0)
      if Stochastic:
        fake_images = denorm(torch.cat(fake_rand_list, dim=3).data.cpu())
        fake_images = torch.cat((self.get_aus(), fake_images), dim=0)
        save_image(fake_images, save_path.replace('fake', 'style'), nrow=1, padding=0)
        if Attention:
          fake_attn = torch.cat(rand_attn_list, dim=3).data.cpu()
          fake_attn = torch.cat((self.get_aus(), fake_attn), dim=0)
          save_image(fake_attn, save_path.replace('fake', 'style_attn'), nrow=1, padding=0)        
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

    data_loader_val = get_loader(self.config.metadata_path, self.config.image_size,
                 self.config.image_size, self.config.batch_size, shuffling = True,
                 dataset=dataset, mode='test', AU=self.config.AUs)[0]  

    for i, (real_x, org_c, files) in enumerate(data_loader_val):
      save_path = os.path.join(self.config.sample_path, '{}_fake_val_{}_{}_{}.jpg'.format(last_name, dataset, i+1, '{}'))
      name = save_path.format(TimeNow_str())
      if 'real_cS' in self.config.GAN_options: 
        self.save_fake_output(real_x, name, label=org_c)
      else: 
        self.save_fake_output(real_x, name)
      self.PRINT('Translated test images and saved into "{}"..!'.format(name))
      if i==1: break   

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

    data_loader = get_loader(path, self.config.image_size,
                 self.config.image_size, self.config.batch_size, shuffling = True,
                 dataset='DEMO', mode='test', AU=self.config.AUs)[0]  

    for real_x in data_loader:
      save_path = os.path.join(self.config.sample_path, '{}_fake_val_DEMO_{}.jpg'.format(last_name, TimeNow_str()))
      self.save_fake_output(real_x, save_path)
      self.PRINT('Translated test images and saved into "{}"..!'.format(save_path))