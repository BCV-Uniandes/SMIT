import torch, os, time, ipdb, glob, math, warnings, datetime
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import config as cfg
from tqdm import tqdm
from termcolor import colored
from misc.utils import circle_frame, color_frame, create_circle, create_dir, denorm, get_aus, get_loss_value, get_torch_version, make_gif, PRINT, send_mail, single_source, slerp, target_debug_list, TimeNow, TimeNow_str, to_cpu, to_cuda, to_data, to_var
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

    # self.get_aus()
    # ipdb.set_trace()
    # Build tensorboard if use

    if self.config.LPIPS_REAL:
      return

    self.build_model()
    if self.config.use_tensorboard:
      self.build_tensorboard()

    # Start with trained model
    if self.config.pretrained_model:
      self.load_pretrained_model()
    elif self.config.dataset_smit:
      self.load_pretrained_smit()
    else:
      if self.config.LPIPS_REAL or self.config.LPIPS_UNIMODAL or self.config.LPIPS_MULTIMODAL or self.config.INCEPTION:
        raise TypeError("No model trained.")

  #=======================================================================================#
  #=======================================================================================#
  def build_model(self):
    # Define a generator and a discriminator
    if self.config.MultiDis>0:
      from model import MultiDiscriminator as Discriminator
    else:
      from model import Discriminator
    if 'AdaIn' in self.config.GAN_options and 'Stochastic' not in self.config.GAN_options:
      from model import AdaInGEN as GEN
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
      if self.config.lambda_style!=0:
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
      if 'Stochastic' in self.config.GAN_options and self.config.lambda_style!=0:
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
    # ipdb.set_trace()
    if hvd.rank() != 0: return
    name = os.path.join(self.config.model_save_path, '{}_{}_{}.pth'.format(Epoch, iter, '{}'))
    torch.save(self.G.state_dict(), name.format('G'))
    torch.save(self.g_optimizer.state_dict(), name.format('G_optim'))
    torch.save(self.D.state_dict(), name.format('D'))
    torch.save(self.d_optimizer.state_dict(), name.format('D_optim'))   
    if 'Split_Optim' in self.config.GAN_options and self.config.lambda_style!=0:
      torch.save(self.s_optimizer.state_dict(), name.format('S_optim'))   

    if self.config.model_epoch!=1 and int(Epoch)%self.config.model_epoch==0:
      for _epoch in range(int(Epoch)-self.config.model_epoch+1, int(Epoch)):
        name_1 = os.path.join(self.config.model_save_path, '{}_{}_{}.pth'.format(str(_epoch).zfill(4), iter, '{}'))
        if os.path.isfile(name_1.format('G')): os.remove(name_1.format('G'))
        if os.path.isfile(name_1.format('G_optim')): os.remove(name_1.format('G_optim'))
        if os.path.isfile(name_1.format('D')): os.remove(name_1.format('D'))
        if os.path.isfile(name_1.format('D_optim')): os.remove(name_1.format('D_optim'))   
        if 'Split_Optim' in self.config.GAN_options and self.config.lambda_style!=0:
          if os.path.isfile(name_1.format('S_optim')): os.remove(name_1.format('S_optim'))           

  #=======================================================================================#
  #=======================================================================================#
  def load_pretrained_model(self):
    if hvd.rank() != 0: return
    self.PRINT('Resuming model (step: {})...'.format(self.config.pretrained_model))
    self.name = os.path.join(self.config.model_save_path, '{}_{}.pth'.format(self.config.pretrained_model, '{}'))
    self.PRINT('Model: {}'.format(self.name))
    # if self.config.mode=='train': self.PRINT('Model: {}'.format(name))
    # self.G.load_state_dict(torch.load(self.name.format('G')))
    # self.D.load_state_dict(torch.load(self.name.format('D')))
    self.G.load_state_dict(torch.load(self.name.format('G'), map_location=lambda storage, loc: storage))
    self.D.load_state_dict(torch.load(self.name.format('D'), map_location=lambda storage, loc: storage))

    try:
      # self.g_optimizer.load_state_dict(torch.load(self.name.format('G_optim')))
      self.g_optimizer.load_state_dict(torch.load(self.name.format('G_optim'), map_location=lambda storage, loc: storage))
      self.optim_cuda(self.g_optimizer)
      # self.d_optimizer.load_state_dict(torch.load(self.name.format('D_optim')))
      self.d_optimizer.load_state_dict(torch.load(self.name.format('D_optim'), map_location=lambda storage, loc: storage))
      self.optim_cuda(self.d_optimizer)
      if self.config.lambda_style!=0 and 'Stochastic' in self.config.GAN_options:
        # self.s_optimizer.load_state_dict(torch.load(self.name.format('S_optim')))
        self.s_optimizer.load_state_dict(torch.load(self.name.format('S_optim'), map_location=lambda storage, loc: storage))
        self.optim_cuda(self.s_optimizer)  
      print("Success!!")
    except: 
      print("Loading Failed!!")

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
    if self.config.dataset_fake==self.config.dataset_smit.split('-')[0]:
      if 'ALL_ATTR_' in self.config.model_save_path:
        smit_model = self.config.model_save_path.replace('ALL_ATTR_{}'.format(self.config.c_dim), self.config.dataset_smit.split('-')[1].upper())  
      else:
        ipdb.set_trace()
    else:
      smit_model = self.config.model_save_path.replace(self.config.dataset_fake, self.config.dataset_smit)
    smit_model = smit_model.replace('/Finetuning_'+self.config.dataset_smit, '')
    smit_G = sorted(glob.glob(os.path.join(smit_model, '*G.pth')))[-1]
    smit_D = smit_G.replace('G.pth', 'D.pth')
    # ipdb.set_trace()
    G_params = {key:value for key,value in self.G.named_parameters() if 'adain_net' in key}
    G_params.update({key:value for key,value in torch.load(smit_G).items() if 'adain_net' not in key})
    self.G.load_state_dict(G_params)#, map_location=lambda storage, loc: storage))   
    D_params = {key:value for key,value in self.D.named_parameters() if 'conv2' in key} 
    D_params.update({key:value for key,value in torch.load(smit_D).items() if 'conv2' not in key})
    self.D.load_state_dict(D_params)#, map_location=lambda storage, loc: storage))

    self.PRINT('!!! Loaded trained SMIT model from: {}..!'.format(smit_G))    

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
  def PRINT_LOG(self, batch_size):
    from termcolor import colored
    Log = "---> batch size: {}, fold: {}, img: {}, GPU: {}, !{}, [{}]\n-> GAN_options:".format(\
        batch_size, self.config.fold, self.config.image_size, \
        self.config.GPU, self.config.mode_data, self.config.PLACE) 
    for item in sorted(self.config.GAN_options):
      Log += ' [*{}]'.format(item.upper())
    if self.config.ALL_ATTR!=0: Log += ' [*ALL_ATTR={}]'.format(self.config.ALL_ATTR)
    if self.config.MultiDis: Log += ' [*MultiDisc={}]'.format(self.config.MultiDis)
    if self.config.lambda_style==0: Log += ' [*NO_StyleLoss]'
    if self.config.style_dim==0: Log += ' [*NO_StyleVector]'
    dataset_string = colored(self.config.dataset_fake, 'red')
    Log += ' [*{}]'.format(dataset_string)
    self.PRINT(Log)    
    return Log

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
  def PRINT(self, str):  
    if self.config.HOROVOD and hvd.rank() != 0: return
    if self.config.mode=='train': PRINT(self.config.log, str)
    else: print(str)

  #=======================================================================================#
  #=======================================================================================#
  def PLOT(self, Epoch):  
    from misc.utils import plot_txt
    LOSS = {key:np.array(value).mean() for key, value in self.LOSS.items()}
    if not os.path.isfile(self.config.loss_plot):
      with open(self.config.loss_plot, 'w') as f: f.writelines('{}\n'.format('\t'.join(['Epoch']+LOSS.keys())))
    with open(self.config.loss_plot, 'a') as f: f.writelines('{}\n'.format('\t'.join([str(Epoch)]+[str(i) for i in LOSS.values()])))
    plot_txt(self.config.loss_plot)

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

    if self.config.mode!='train': 
      circle = 1-torch.FloatTensor(create_circle(self.config.image_size))
      circle = circle.expand(fake_images.size(0),fake_images.size(1), self.config.image_size, self.config.image_size)
      circle = circle.repeat(1,1,1,fake_images.size(-1)//self.config.image_size)
      fake_images += circle
    if fake_images.size(0)>1:
      fake_images = torch.cat((self.get_aus(), fake_images), dim=0)
      if gif: make_gif(fake_images, save_path, im_size=im_size)
    else:
      if gif: make_gif(fake_images, save_path, im_size=im_size)
      fake_images = torch.cat((self.get_aus(), fake_images), dim=0)
    _save_path = save_path.replace('fake', mode)    

    save_image(fake_images, _save_path, nrow=1, padding=0)
    # ipdb.set_trace()
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
    cross_entropy = self.config.dataset_fake in ['painters_14', 'Animals', 'Image2Weather', 'Image2Season', 'Image2Edges', 'Yosemite']
    cross_entropy = cross_entropy or (self.config.dataset_fake=='RafD' and self.config.RafD_FRONTAL)
    cross_entropy = cross_entropy or (self.config.dataset_fake=='RafD' and self.config.RafD_EMOTIONS)
    if cross_entropy:
      # ipdb.set_trace()
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

  #=======================================================================================#
  #=======================================================================================#    
  #=======================================================================================#
  #=======================================================================================#
  def train(self):

    GAN_options = self.config.GAN_options

    # lr cache for decaying
    g_lr = self.config.g_lr
    d_lr = self.config.d_lr
    # ipdb.set_trace()
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
        # ipdb.set_trace()
        self.loss = {}
        _iter+=1
        #RaGAN uses different data for Dis and Gen 
        if 'RaGAN' in GAN_options:
          split = lambda x: (x[:x.size(0)//2], x[x.size(0)//2:])
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
        log = '--> %s | Elapsed (%d/%d) : %s | %s\nTrain'%(TimeNow(), e, self.config.num_epochs, elapsed, Log)
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

  #=======================================================================================#
  #=======================================================================================#

  def save_fake_output(self, real_x, save_path, label=None, gif=False, output=False, many_faces=False, training=False, Style=0, fixed_style=None, TIME=False):
    self.G.eval()  
    self.D.eval()
    Attention = 'Attention' in self.config.GAN_options or 'Attention2' in self.config.GAN_options or 'Attention3' in self.config.GAN_options
    Stochastic = 'Stochastic' in self.config.GAN_options
    n_rep = self.config.style_debug
    Output = []
    opt = torch.no_grad() if get_torch_version()>0.3 else open('_null.txt', 'w')
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

      if Stochastic:
        if fixed_style is None:
          style_all = self.G.random_style(max(real_x.size(0), n_rep))
          style=to_var(style_all[:real_x.size(0)].clone(), volatile=True)
        else:
          style=to_var(fixed_style[:real_x.size(0)].clone(), volatile=True)
        
      else:
        style = None

      # Batch of images
      if Style==0:
        for k, target_c in enumerate(target_c_list):
          # target_c = torch.clamp(target_c+out_label,max=1)
          if k==0: continue
          if self.config.dataset_fake in ['CelebA', 'EmotionNet', 'BP4D', 'DEMO']:
            target_c = (out_label-target_c)**2 #Swap labels
            if self.config.dataset_fake == 'CelebA' and self.config.c_dim==10 and k>=3 and k<=6:
              target_c[:,2:5]=0
              target_c[:,k-1]=1   
            if self.config.dataset_fake == 'CelebA' and self.config.c_dim==40:
              all_attr = self.data_loader.dataset.selected_attrs
              idx2attr = self.data_loader.dataset.idx2attr
              attr2idx = self.data_loader.dataset.attr2idx
              color_hair = ['Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
              style_hair = ['Bald', 'Straight_Hair', 'Wavy_Hair']
              ammount_hair = ['Bald', 'Bangs']
              if all_attr[k-1] in color_hair:
                # ipdb.set_trace()
                color_hair.remove(all_attr[k-1]) 
                for attr in color_hair: 
                  if attr in all_attr: target_c[:,attr2idx[attr]]=0
                target_c[:,k-1]=1 
              if all_attr[k-1] in style_hair:
                style_hair.remove(all_attr[k-1])  
                for attr in style_hair: 
                  if attr in all_attr: target_c[:,attr2idx[attr]]=0         
                target_c[:,k-1]=1 
              if all_attr[k-1] in ammount_hair:
                ammount_hair.remove(all_attr[k-1])  
                for attr in ammount_hair: 
                  if attr in all_attr: target_c[:,attr2idx[attr]]=0                        
              # target_c[:,k-1]=1
              # ipdb.set_trace()
            # if self.config.dataset_fake == 'CelebA' and self.config.c_dim==10 and k>=3 and k<=6:
            #   target_c[:,2:5]=0
            #   target_c[:,k-1]=1
          start_time = time.time()
          fake_x = self.G(real_x, target_c, stochastic=style)
          elapsed = time.time() - start_time
          elapsed = str(datetime.timedelta(seconds=elapsed))
          if TIME and flag_time: 
            print("Time/batch_size for one single forward generation (bs:{}): {}".format(real_x.size(0),elapsed))
            flag_time=False

          fake_image_list.append(fake_x[0])
          if Attention: fake_attn_list.append(fake_x[1].repeat(1,3,1,1))
        if many_faces: 
          return denorm(to_data(fake_image_list[np.random.randint(1,len(fake_image_list),1)[0]][0], cpu=True)).numpy().transpose(1,2,0)
        _name = '' if fixed_style is not None else '_Random'
        _save_path = save_path.replace('.jpg',_name+'.jpg')          
        Output.extend(self._SAVE_IMAGE(_save_path, fake_image_list, im_size=self.config.image_size, attn_list=fake_attn_list, gif=gif))

      #Same image different style
      for idx, real_x0 in enumerate(real_x):
        if training:
          _name = '' if fixed_style is not None else '_Random'
          _save_path = save_path.replace('.jpg',_name+'.jpg')
        else:
          _name = '' if fixed_style is not None else '_Random'
          _save_path = os.path.join(save_path.replace('.jpg', ''), '{}_{}{}.jpg'.format(Style, str(idx).zfill(4), _name))
          create_dir(_save_path)
        real_x0 = real_x0.repeat(n_rep,1,1,1)#.unsqueeze(0)
        _real_x0 = real_x0.clone()
        _out_label = out_label[idx].repeat(n_rep,1)
        label_space = np.linspace(0,1,n_rep)

        fake_image_list = [single_source(real_x0)]
        fake_attn_list  = []       

        if Attention: 
          fake_attn_list = [single_source(to_var(denorm(real_x0.data), volatile=True))] 
        for n_label, _target_c in enumerate(target_c_list):
          if n_label==0: continue
          _target_c  = _target_c[0].repeat(n_rep,1)
          # target_c = torch.clamp(_target_c+_out_label, max=1)
          if self.config.dataset_fake in ['CelebA', 'EmotionNet', 'BP4D', 'DEMO']:
            target_c = (_out_label-_target_c)**2 #Swap labels
            if self.config.dataset_fake == 'CelebA' and self.config.c_dim==10 and n_label>=3 and n_label<=6:
              target_c[:,2:5]=0
              target_c[:,n_label-1]=1   
            if self.config.dataset_fake == 'CelebA' and self.config.c_dim==40:
              all_attr = self.data_loader.dataset.selected_attrs
              idx2attr = self.data_loader.dataset.idx2attr
              attr2idx = self.data_loader.dataset.attr2idx
              color_hair = ['Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
              style_hair = ['Bald', 'Straight_Hair', 'Wavy_Hair']
              ammount_hair = ['Bald', 'Bangs']
              if all_attr[n_label-1] in color_hair:
                color_hair.remove(all_attr[n_label-1])  
                for attr in color_hair: 
                  if attr in all_attr: target_c[:,attr2idx[attr]]=0
                target_c[:,n_label-1]=1 
              if all_attr[n_label-1] in style_hair:
                style_hair.remove(all_attr[n_label-1])  
                for attr in style_hair: 
                  if attr in all_attr: target_c[:,attr2idx[attr]]=0         
                target_c[:,n_label-1]=1 
              if all_attr[n_label-1] in ammount_hair:
                ammount_hair.remove(all_attr[n_label-1])  
                for attr in ammount_hair: 
                  if attr in all_attr: target_c[:,attr2idx[attr]]=0                              
              # ipdb.set_trace()    
              # target_c[:,n_label-1]=1   
          else: target_c = _target_c
          
          if Stochastic:
            if fixed_style is None:
              style_all = self.G.random_style(50)
              style0 = to_var(style_all[:n_rep].clone(), volatile=True)  
            else:
              style0 = to_var(fixed_style[:n_rep].clone(), volatile=True)  
            
            style_rec0 = to_var(self.G.random_style(style0), volatile=True)
            _style = style.clone()
            _style0 = style0.clone()

            #Style interpolation | Translate
            if Style==1: 
              z0 = to_data(style0[0], cpu=True).numpy(); z1 = to_data(style0[1], cpu=True).numpy()
              z_interp = style0.clone()
              z_interp[:] = torch.FloatTensor(np.array([slerp(sz, z0, z1) for sz in np.linspace(0,1,style0.size(0))]))
              _style0 = z_interp

            #Style constant | progressive swap label
            elif Style==2: 
              for j, i in enumerate(range(real_x0.size(0))): 
                _style0[i] = style0[0].clone()
                if n_label>0:
                  target_c[i][n_label-1].data.fill_((target_c[i][n_label-1]*label_space[j] + (1-target_c[i][n_label-1])*(1-label_space[j])).data[0])
                else:
                  target_c[i] = target_c[i]*label_space[j] + (1-target_c[i])*(1-label_space[j])

            # Translate and translate back
            elif Style==3:
              real_x0 = self.G(_real_x0, target_c, stochastic=style_rec0)[0]
              # ipdb.set_trace()
              if self.config.dataset_fake in ['CelebA', 'EmotionNet', 'BP4D', 'DEMO']:
                target_c = (target_c-_target_c)**2

            #Style 0 | progressive swap label
            elif Style==4: 
              for j, i in enumerate(range(real_x0.size(0))): 
                _style0[i] = style0[0]*0
                if n_label>0:
                  target_c[i][n_label-1].data.fill_((target_c[i][n_label-1]*label_space[j] + (1-target_c[i][n_label-1])*(1-label_space[j])).data[0])
                else:
                  target_c[i] = target_c[i]*label_space[j] + (1-target_c[i])*(1-label_space[j])          

            #Style constant | progressive label
            elif Style==5:
              for j, i in enumerate(range(real_x0.size(0))): 
                _style0[i] = style0[2].clone()
                target_c[i] = _target_c[i]*0.2*j   

            #Style random | One label at a time
            elif Style==6: 
              target_c = _target_c

            #Extract style from the two before, current, and two after. 
            elif Style==7:
              # _real_x0 = torch.zeros_like(real_x0)
              # _range = range(-int(n_rep//2), 1+int(n_rep//2))
              # if len(_range)>n_rep: _range.pop(0)
              # for j, k in enumerate(_range):
              #   kk = (k+idx)%real_x.size(0) if k+idx >= real_x.size(0) else k+idx
                # _real_x0[j] = real_x[kk]
              _real_x0 = real_x[:n_rep]
              if 'STYLE_DISC' in self.config.GAN_options:
                _style0 = self.D(_real_x0)[-1][0]
              else:
                _style0 = self.G.get_style(_real_x0)
              # ipdb.set_trace()

          else:
            _style0 = None
            for j, i in enumerate(range(real_x0.size(0))): 
              target_c[i] = target_c[i]*label_space[j]            

          # ipdb.set_trace()
          # if Style==3: ipdb.set_trace()
          fake_x = self.G(real_x0, target_c, stochastic=_style0)
          # if Style>=1: fake_x[0] = color_frame(fake_x[0], thick=5, first=n_label==1) #After off
          if Style>=1: fake_x[0] = circle_frame(fake_x[0], thick=5, first=n_label==1) #After off
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
    create_dir(save_folder)
    if dataset=='': 
      dataset = self.config.dataset_fake
      data_loader = self.data_loader
    else:
      data_loader = get_loader(self.config.metadata_path, self.config.image_size, self.config.batch_size, shuffling=True, dataset=dataset, mode='test') 

    if 'Stochastic' in self.config.GAN_options:
      _debug = self.config.style_label_debug+1
      style_all = self.G.random_style(50)
    else:
      style_all = None
      _debug = 1

    for i, (real_x, org_c, files) in enumerate(data_loader):
      save_path = os.path.join(save_folder, '{}_{}_{}.jpg'.format(dataset, '{}', i+1))
      string = '{}'
      if self.config.NO_LABELCUM:
        string += '_{}'.format('NO_Label_Cum','{}')
      string = string.format(TimeNow_str())
      name = os.path.abspath(save_path.format(string))
      if self.config.dataset_fake==self.config.dataset_real:
        label = org_c
      else:
        label = None
      self.PRINT('Translated test images and saved into "{}"..!'.format(name))
      # output = self.save_fake_output(real_x, name, label=label, output=True, Style=3, TIME=not i)
      for k in range(_debug):
        output = self.save_fake_output(real_x, name, label=label, output=True, Style=k, fixed_style=style_all, TIME=not i)
        if 'Stochastic' in self.config.GAN_options: output = self.save_fake_output(real_x, name, label=label, output=True, Style=k) #random style
        # send_mail(body='Images from '+self.config.sample_path, attach=output)
      if i==self.config.iter_test-1: break   

  #=======================================================================================#
  #=======================================================================================#

  def DEMO(self, path):
    from data_loader import get_loader    
    import re
    last_name = self.resume_name()
    save_folder = os.path.join(self.config.sample_path, '{}_test'.format(last_name))
    create_dir(save_folder)
    batch_size = self.config.batch_size if not 'Stochastic' in self.config.GAN_options else 1
    data_loader = get_loader(path, self.config.image_size, batch_size, shuffling = False, dataset='DEMO', mode='test', many_faces=self.config.many_faces)
    label = self.config.DEMO_LABEL 
    if self.config.DEMO_LABEL!='':
      label = torch.FloatTensor([int(i) for i in label.split(',')]).view(1,-1)
      # ipdb.set_trace()
    else:
      label = None
    if 'Stochastic' in self.config.GAN_options:
      _debug = self.config.style_label_debug+1
      style_all = self.G.random_style(50)
    else:
      style_all = None
      _debug = 1

    name = TimeNow_str()
    Output = []
    if not self.config.many_faces:
      for i, real_x in enumerate(data_loader):
        save_path = os.path.join(save_folder, 'DEMO_{}_{}.jpg'.format(name, i+1))
        self.PRINT('Translated test images and saved into "{}"..!'.format(save_path))
        for k in range(_debug):
          output = self.save_fake_output(real_x, save_path, gif=False, label=label, output=True, Style=k, fixed_style=style_all, TIME=not i)
          if self.config.many_faces: Output.append(output); break
          if 'Stochastic' in self.config.GAN_options: 
            output = self.save_fake_output(real_x, save_path, gif=False, label=label, output=True, Style=k) #random style
          # send_mail(body='Images from '+self.config.sample_path, attach=output)

    if self.config.many_faces:
      import skimage.transform, imageio, matplotlib.pyplot as plt, pylab as pyl
      from PIL import Image
      # ipdb.set_trace()
      org_img = imageio.imread(data_loader.dataset.img_path)
      for i, real_x in enumerate(data_loader):
        face = self.save_fake_output(real_x, 'None', output=True, many_faces=True, Style=0, fixed_style=style_all)
        bbox = data_loader.dataset.lines[i]
        # image = Image.open(data_loader.dataset.img_path).convert('RGB').crop(bbox)
        bbox = [max(bbox[0],0), max(bbox[1],0), min(bbox[2], org_img.shape[1]), min(bbox[3],org_img.shape[0])]
        resize = [bbox[3]-bbox[1], bbox[2]-bbox[0]]
        img = (skimage.transform.resize(face, resize, mode='reflect', anti_aliasing=True)*255).astype(np.uint8)
        try:org_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = img
        except: ipdb.set_trace()
      plt.subplot(2,1,1)
      plt.imshow(imageio.imread(data_loader.dataset.img_path))
      plt.subplot(2,1,2)
      plt.imshow(org_img)
      plt.subplots_adjust(left=None, bottom=None, right=None, top=None,\
                          wspace=0.5, hspace=0.5)
      pyl.savefig('many_faces.pdf', dpi=100)

  #=======================================================================================#
  #=======================================================================================#

  def LPIPS(self):
    from misc.utils import compute_lpips   
    data_loader = self.data_loader  
    n_images = 100
    pair_styles = 20      
    model = None
    DISTANCE = {0:[], 1:[]}
    self.G.eval()
    for i, (real_x, org_c, files) in tqdm(enumerate(data_loader), desc='Calculating LPISP', total=n_images):
      for _real_x, _org_c in zip(real_x, org_c):
        _real_x = _real_x.unsqueeze(0)
        _org_c = _org_c.unsqueeze(0)
        if len(DISTANCE[_org_c[0,0]])>=i: continue
        _real_x = to_var(_real_x, volatile=True)
        target_c = to_var(1-_org_c, volatile=True)
        for _ in range(pair_styles):
          style0 = to_var(self.G.random_style(_real_x.size(0)), volatile=True)
          style1 = to_var(self.G.random_style(_real_x.size(0)), volatile=True)
          # ipdb.set_trace()
          fake_x0 = self.G(_real_x, target_c, stochastic=style0)
          fake_x1 = self.G(_real_x, target_c, stochastic=style1)
          distance, model = compute_lpips(fake_x0, fake_x1, model=model)
          DISTANCE[org_c[0,0]].append(distance)
        if i==len(DISTANCE[0,0])==len(DISTANCE[1]): break
    print("LPISP a-b: {}".format(np.array(DISTANCE[0]).mean()))
    print("LPISP b-a: {}".format(np.array(DISTANCE[1]).mean()))

  #=======================================================================================#
  #=======================================================================================#

  def LPIPS_REAL(self):
    from misc.utils import compute_lpips   
    data_loader = self.data_loader    
    model = None
    file_name = 'scores/{}_Attr_{}_LPIPS.txt'.format(self.config.dataset_fake, self.config.ALL_ATTR)
    if os.path.isfile(file_name):
      print(file_name)
      for line in open(file_name).readlines(): print(line.strip())
      return

    DISTANCE = {i:[] for i in range(len(data_loader.dataset.labels[0])+1)}#0:[], 1:[], 2:[]}
    n_images = {i:0 for i in range(len(data_loader.dataset.labels[0]))}
    for i, (real_x, org_c, files) in tqdm(enumerate(data_loader), desc='Calculating LPISP - {}'.format(file_name), total=len(data_loader)):
      org_label = torch.max(org_c,1)[1][0]
      for label in range(len(data_loader.dataset.labels[0])):
        for j, (_real_x, _org_c, _files) in enumerate(data_loader):
          if j<=i: continue
          _org_label = torch.max(_org_c,1)[1][0]
          for _label in range(len(data_loader.dataset.labels[0])):
            if _org_label == _label: continue
            distance, model = compute_lpips(real_x, _real_x, model=model)
            DISTANCE[len(data_loader.dataset.labels[0])].append(distance[0])
            if label==_label: 
              DISTANCE[_label].append(distance[0])        
    
    file_ = open(file_name, 'w')    
    DISTANCE = {k:np.array(v) for k,v in DISTANCE.items()}
    for key, values in DISTANCE.items():
      if key==len(data_loader.dataset.labels[0]): mode = 'All'
      else: mode = chr(65+key)
      PRINT(file_, "LPISP {}: {} +/- {}".format(mode, values.mean(), values.std()))
    # ipdb.set_trace()
    file_.close()

  #=======================================================================================#
  #=======================================================================================#

  def LPIPS_UNIMODAL(self):
    from misc.utils import compute_lpips   
    from shutil import copyfile
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    data_loader = self.data_loader    
    model = None
    style_fixed = True
    style_str = 'fixed' if style_fixed else 'random'
    file_name = os.path.join(self.name.replace('{}.pth', 'LPIPS_UNIMODAL_{}.txt'.format(style_str)))
    copy_name = 'scores/{}_Attr_{}_LPIPS_UNIMODAL_{}.txt'.format(self.config.dataset_fake, self.config.ALL_ATTR, style_str)
    if os.path.isfile(file_name):
      print(file_name)
      for line in open(file_name).readlines(): print(line.strip())
      return

    # ipdb.set_trace()
    DISTANCE = {i:[] for i in range(len(data_loader.dataset.labels[0])+1)}#0:[], 1:[], 2:[]}
    n_images = {i:0 for i in range(len(data_loader.dataset.labels[0]))}

    style0 = to_var(self.G.random_style(1), volatile=True) if 'Stochastic' in self.config.GAN_options else None
    print(file_name)
    for i, (real_x, org_c, files) in tqdm(enumerate(data_loader), desc='Calculating LPISP ', total=len(data_loader)):
      org_label = torch.max(org_c,1)[1][0]
      real_x = to_var(real_x, volatile=True)
      for label in range(len(data_loader.dataset.labels[0])):
        if org_label == label: continue
        target_c = to_var(org_c*0, volatile=True); target_c[:,label]=1
        if not style_fixed: style0 = to_var(self.G.random_style(real_x.size(0)), volatile=True)
        real_x = self.G(real_x, to_var(target_c, volatile=True), stochastic=style0)[0]
        n_images[label]+=1
        for j, (_real_x, _org_c, _files) in enumerate(data_loader):
          if j<=i: continue
          _org_label = torch.max(_org_c,1)[1][0]
          _real_x = to_var(_real_x, volatile=True)
          for _label in range(len(data_loader.dataset.labels[0])):
            if _org_label == _label: continue
            _target_c = to_var(_org_c*0, volatile=True); _target_c[:,_label]=1          
            if not style_fixed: style0 = to_var(self.G.random_style(_real_x.size(0)), volatile=True)
            _real_x = self.G(_real_x, to_var(_target_c, volatile=True), stochastic=style0)[0]
            # ipdb.set_trace()
            distance, model = compute_lpips(real_x.data, _real_x.data, model=model)
            DISTANCE[len(data_loader.dataset.labels[0])].append(distance[0])
            # if label==0: ipdb.set_trace()
            if label==_label: 
              # if label==0: ipdb.set_trace()
              DISTANCE[_label].append(distance[0])

    # ipdb.set_trace()
    file_ = open(file_name, 'w')    
    DISTANCE = {k:np.array(v) for k,v in DISTANCE.items()}
    for key, values in DISTANCE.items():
      if key==len(data_loader.dataset.labels[0]): mode = 'All'
      else: mode = chr(65+key)
      PRINT(file_, "LPISP {}: {} +/- {}".format(mode, values.mean(), values.std()))
    # PRINT(file_, "LPISP b-a: {} +/- {}".format(DISTANCE[1].mean(), DISTANCE[1].std()))
    # PRINT(file_, "LPISP All: {} +/- {}".format(DISTANCE[2].mean(), DISTANCE[2].std()))
    file_.close()    
    copyfile(file_name, copy_name)
    # ipdb.set_trace()

  #=======================================================================================#
  #=======================================================================================#

  def LPIPS_MULTIMODAL(self):
    from misc.utils import compute_lpips   

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    data_loader = self.data_loader    
    model = None
    n_images = 20
    file_name = 'scores/{}_Attr_{}_LPIPS_MULTIMODAL.txt'.format(self.config.dataset_fake, self.config.ALL_ATTR)
    if os.path.isfile(file_name):
      print(file_name)
      for line in open(file_name).readlines(): print(line.strip())
      return    
    # DISTANCE = {0:[], 1:[], 2:[]}
    DISTANCE = {i:[] for i in range(len(data_loader.dataset.labels[0])+1)}#0:[], 1:[], 2:[]}
    # n_images = {i:0 for i in range(len(data_loader.dataset.labels[0]))}
    for i, (real_x, org_c, files) in tqdm(enumerate(data_loader), desc='Calculating LPISP - {}'.format(file_name), total=len(data_loader)):
      org_label = torch.max(org_c,1)[1][0]
      for label in range(len(data_loader.dataset.labels[0])):
        if org_label == label: continue
        target_c = org_c*0; target_c[:,label]=1      
        # target_c = 1-org_c
        # label = 1-target_c[0,0]
        target_c = target_c.repeat(n_images,1)
        real_x_var = to_var(real_x.repeat(n_images,1,1,1), volatile=True)
        target_c = to_var(target_c, volatile=True)
        style = to_var(self.G.random_style(n_images), volatile=True)
        # ipdb.set_trace()
        fake_x = self.G(real_x_var, target_c, stochastic=style)[0].data
        fake_x = [f.unsqueeze(0) for f in fake_x]
        # ipdb.set_trace()
        _DISTANCE = []
        for ii, fake0 in enumerate(fake_x):
          for jj, fake1 in enumerate(fake_x):
            if jj<=ii: continue
            distance, model = compute_lpips(fake0, fake1, model=model)
            _DISTANCE.append(distance[0])
        # ipdb.set_trace()
        DISTANCE[len(data_loader.dataset.labels[0])].append(np.array(_DISTANCE).mean())
        DISTANCE[label].append(DISTANCE[len(data_loader.dataset.labels[0])][-1])


    file_ = open(file_name, 'w')    
    DISTANCE = {k:np.array(v) for k,v in DISTANCE.items()}
    for key, values in DISTANCE.items():
      if key==len(data_loader.dataset.labels[0]): mode = 'All'
      else: mode = chr(65+key)
      PRINT(file_, "LPISP {}: {} +/- {}".format(mode, values.mean(), values.std()))
    # PRINT(file_, "LPISP b-a: {} +/- {}".format(DISTANCE[1].mean(), DISTANCE[1].std()))
    # PRINT(file_, "LPISP All: {} +/- {}".format(DISTANCE[2].mean(), DISTANCE[2].std()))
    file_.close()  

    # ipdb.set_trace()
    # file_ = open(file_name, 'w')    
    # DISTANCE = {k:np.array(v) for k,v in DISTANCE.items()}
    # PRINT(file_, "LPISP a-b: {} +/- {}".format(DISTANCE[0].mean(), DISTANCE[0].std()))
    # PRINT(file_, "LPISP b-a: {} +/- {}".format(DISTANCE[1].mean(), DISTANCE[1].std()))
    # PRINT(file_, "LPISP All: {} +/- {}".format(DISTANCE[2].mean(), DISTANCE[2].std()))
    # file_.close()    
    # ipdb.set_trace()

  #=======================================================================================#
  #=======================================================================================#

  def INCEPTION(self):
    from misc.utils import load_inception
    from scipy.stats import entropy
    n_styles = 20
    net = load_inception()
    to_cuda(net)
    net.eval()
    self.G.eval()
    inception_up = nn.Upsample(size=(299, 299), mode='bilinear')
    if 'Stochastic' in self.config.GAN_options:
      mode = 'SMIT'
    elif 'Attention' in self.config.GAN_options:
      mode = 'GANimation'
    else:
      mode = 'StarGAN'
    data_loader = self.data_loader    
    file_name = 'scores/Inception_{}.txt'.format(mode)
    # if os.path.isfile(file_name):
    #   print(file_name)
    #   for line in open(file_name).readlines(): print(line.strip())
    #   return

    PRED_IS = {i:[] for i in range(len(data_loader.dataset.labels[0]))}#0:[], 1:[], 2:[]}
    CIS = {i:[] for i in range(len(data_loader.dataset.labels[0]))}
    IS = {i:[] for i in range(len(data_loader.dataset.labels[0]))}

    for i, (real_x, org_c, files) in tqdm(enumerate(data_loader), desc='Calculating CIS/IS - {}'.format(file_name), total=len(data_loader)):
      PRED_CIS = {i:[] for i in range(len(data_loader.dataset.labels[0]))}#0:[], 1:[], 2:[]}
      org_label = torch.max(org_c,1)[1][0]
      real_x = real_x.repeat(n_styles,1,1,1)#.unsqueeze(0)
      real_x = to_var(real_x, volatile=True)

      target_c = (org_c*0).repeat(n_styles,1)  
      target_c = to_var(target_c, volatile=True)
      for label in range(len(data_loader.dataset.labels[0])):
        if org_label == label: continue
        target_c *= 0
        target_c[:,label]=1   
        style = to_var(self.G.random_style(n_styles), volatile=True) if mode=='SMIT' else None

        fake = (self.G(real_x, target_c, style)[0]+1)/2
        # ipdb.set_trace()
        # save_image(denorm(real_x.data), 'dummy.jpg')
        # save_image(fake.data, 'dummy.jpg')
        # target_c *= 0; target_c[:,label]=1 ; fake = (self.G(real_x, target_c, style)[0]+1)/2; save_image(fake.data, 'dummy.jpg')
        pred = to_data(F.softmax(net(inception_up(fake)), dim=1), cpu=True).numpy()
        PRED_CIS[label].append(pred)
        PRED_IS[label].append(pred)

        # CIS for each image
        PRED_CIS[label] = np.concatenate(PRED_CIS[label], 0)
        py = np.sum(PRED_CIS[label], axis=0)  # prior is computed from outputs given a specific input
        for j in range(PRED_CIS[label].shape[0]):
          pyx = PRED_CIS[label][j, :]
          CIS[label].append(entropy(pyx, py))
      # ipdb.set_trace()

    for label in range(len(data_loader.dataset.labels[0])):
      PRED_IS[label] = np.concatenate(PRED_IS[label], 0)
      py = np.sum(PRED_IS[label], axis=0)  # prior is computed from all outputs
      for j in range(PRED_IS[label].shape[0]):
        pyx = PRED_IS[label][j, :]
        IS[label].append(entropy(pyx, py))      

    total_cis = []; total_is = []
    file_ = open(file_name, 'w')
    for label in range(len(data_loader.dataset.labels[0])):
      cis = np.exp(np.mean(CIS[label]))
      total_cis.append(cis)
      _is = np.exp(np.mean(IS[label]))
      total_is.append(_is)
      PRINT(file_, "Label {}".format(label))
      PRINT(file_, "Inception Score: {:.4f}".format(_is))
      PRINT(file_, "conditional Inception Score: {:.4f}".format(cis))
    PRINT(file_, "")
    PRINT(file_, "[TOTAL] Inception Score: {:.4f} +/- {:.4f}".format(np.array(total_is).mean(), np.array(total_is).std()))
    PRINT(file_, "[TOTAL] conditional Inception Score: {:.4f} +/- {:.4f}".format(np.array(total_cis).mean(), np.array(total_cis).std()))
    file_.close()  