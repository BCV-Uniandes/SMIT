import torch, os, time, ipdb, glob, warnings, datetime
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
from termcolor import colored
from misc.utils import circle_frame, color_frame, create_arrow, create_circle, create_dir, denorm, get_labels, get_torch_version, make_gif, PRINT, single_source, slerp, target_debug_list, TimeNow, TimeNow_str, to_cuda, to_data, to_var
import torch.utils.data.distributed
from misc.utils import _horovod
hvd = _horovod()

warnings.filterwarnings('ignore')

class Solver(object):

  def __init__(self, config, data_loader=None):
    # Data loader
    self.data_loader = data_loader
    self.config = config

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
        raise TypeError("No model trained at {}.".format(self.config.model_save_path))

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

    if self.config.mode=='train':
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
    else:
      print("Success!!")

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
      try:
        last_file = sorted(glob.glob(os.path.join(self.config.model_save_path,  '*_G.pth')))[-1]
      except IndexError:
        raise IndexError("No model found at "+self.config.model_save_path)
      last_name = '_'.join(os.path.basename(last_file).split('_')[:2])
    else:
      last_name = self.config.pretrained_model 
    return last_name

  #=======================================================================================#
  #=======================================================================================#
  def get_labels(self):
    return get_labels(self.config.image_size, self.config.dataset_fake, attr=self.data_loader.dataset)

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
  def _SAVE_IMAGE(self, save_path, fake_list, attn_list=[], im_size=256, gif=False, mode='fake', style=None, no_labels=False):
    Output = []
    if self.config.HOROVOD and hvd.rank() != 0: return Output
    fake_images = denorm(to_data(torch.cat(fake_list, dim=3), cpu=True))
    if len(attn_list): 
      fake_attn = to_data(torch.cat(attn_list, dim=3), cpu=True)

    # if self.config.mode!='train' and 'CelebA' in self.config.dataset_fake: 
    #   circle = 1-torch.FloatTensor(create_circle(self.config.image_size))
    #   circle = circle.expand(fake_images.size(0),fake_images.size(1), self.config.image_size, self.config.image_size)
    #   circle = circle.repeat(1,1,1,fake_images.size(-1)//self.config.image_size)
    #   fake_images += circle
    #   if len(attn_list): fake_attn += circle

    _save_path = save_path.replace('fake', mode)
    if 'DEMO' in _save_path:
      for k in range(fake_images.size(0)):
        for i in range(fake_images.size(3)//fake_images.size(2)):
          if k>0 and i==0: continue
          __save_path = _save_path.replace('.jpg', '{}_{}.jpg'.format(str(i).zfill(2), str(k).zfill(2)))
          save_image(fake_images[k,:,:,i*fake_images.size(2):(i+1)*fake_images.size(2)].unsqueeze(0), __save_path, nrow=1, padding=0)

    if no_labels:
      if gif: make_gif(fake_images, save_path, im_size=im_size)      
    elif fake_images.size(0)>1:
      fake_images = torch.cat((self.get_labels(), fake_images), dim=0)
      if gif: make_gif(fake_images, save_path, im_size=im_size)
    else:
      if gif: make_gif(fake_images, save_path, im_size=im_size)
      fake_images = torch.cat((self.get_labels(), fake_images), dim=0)    

    save_image(fake_images, _save_path, nrow=1, padding=0)
    if style is not None:
      create_arrow(_save_path, style, image_size = self.config.image_size)
    if no_labels:
      create_arrow(_save_path, 0, image_size = self.config.image_size, horizontal=True)      
    Output.append(_save_path)
    if gif: 
      Output.append(_save_path.replace('jpg', 'gif'))
      Output.append(_save_path.replace('jpg', 'mp4'))
    if len(attn_list): 
      if 'fake' not in os.path.basename(save_path): save_path=save_path.replace('.jpg', '_fake.jpg')
      _save_path = save_path.replace('fake', '{}_attn'.format(mode))      
      if 'DEMO' in _save_path:
        for k in range(fake_attn.size(0)):
          for i in range(fake_attn.size(3)//fake_attn.size(2)):
            if k>0 and i==0: continue
            __save_path = _save_path.replace('.jpg', '{}_{}.jpg'.format(str(i).zfill(2), str(k).zfill(2)))
            save_image(fake_attn[k,:,:,i*fake_attn.size(2):(i+1)*fake_attn.size(2)].unsqueeze(0), __save_path, nrow=1, padding=0)
      fake_attn = torch.cat((self.get_labels(), fake_attn), dim=0)
      save_image(fake_attn, _save_path, nrow=1, padding=0)
      Output.append(_save_path.replace('fake', 'attn'))    
    return Output  



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
          # if self.config.dataset_fake in ['CelebA', 'EmotionNet', 'BP4D', 'DEMO']:
          if self.config.dataset_fake in ['CelebA', 'EmotionNet', 'DEMO']:
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
        # downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        real_x0 = real_x0.repeat(n_rep,1,1,1)#.unsqueeze(0)
        _real_x0 = real_x0.clone()
        _out_label = out_label[idx].repeat(n_rep,1)
        label_space = np.linspace(0, 1, n_rep)
        if Style==3:
          real_x0 = real_x0.repeat(2,1,1,1)
          fake_image_list = [single_source(real_x0[:-1])]
        else:
          fake_image_list = [single_source(real_x0)]

        fake_image_list[0] = color_frame(fake_image_list[0], thick=5, color='green', first=True)

        if Attention: 
          if Style==3:
            fake_attn_list = [single_source(to_var(denorm(real_x0[:-1].data), volatile=True))] 
          else:
            fake_attn_list = [single_source(to_var(denorm(real_x0.data), volatile=True))] 
          fake_attn_list[0] = color_frame(fake_attn_list[0], thick=5, color='green', first=True)
        for n_label, _target_c in enumerate(target_c_list):
          if n_label==0: continue
          _target_c  = _target_c[0].repeat(n_rep,1)
          # target_c = torch.clamp(_target_c+_out_label, max=1)
          # if self.config.dataset_fake in ['CelebA', 'EmotionNet', 'BP4D', 'DEMO']:
          if self.config.dataset_fake in ['CelebA', 'EmotionNet', 'DEMO']:
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
              # target_c[:,n_label-1]=1   
          else: target_c = _target_c
          
          if Stochastic:
            if fixed_style is None:
              style_all = self.G.random_style(max(real_x.size(0),50))
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


            #Style interpolation | progressive swap label
            elif Style==3: 
              label_space = np.linspace(0.2, 0.8, n_rep-1)
              target_c = target_c.repeat(2,1)
              z0 = to_data(style0[0], cpu=True).numpy(); z1 = to_data(style0[1], cpu=True).numpy()
              z_interp = style0.clone()
              z_interp[:] = torch.FloatTensor(np.array([slerp(sz, z0, z1) for sz in np.linspace(0,1,style0.size(0))]))
              # for i in range(z_interp.size(1)):
              #   z_interp[:,i] = torch.FloatTensor(np.linspace(z0[i],z1[i],style0.size(0)))
              _style0 = style0.repeat(2,1)
              _style0[:real_x0.size(0)//2] = z_interp

              for j, i in enumerate(range(real_x0.size(0)//2,real_x0.size(0)-1)): 
                _style0[i] = _style0[(real_x0.size(0)//2)-1].clone()
                if n_label>0:
                  reverse = -1 if target_c[i][n_label-1].data[0]==1 else 1
                  target_c[i][n_label-1].data.fill_(label_space[::reverse][j])
                else:
                  target_c[i] = (target_c[i]*label_space[j] + (1-target_c[i])*(1-label_space[j]))

            # Translate and translate back
            elif Style==4:
              real_x0 = self.G(_real_x0, target_c, stochastic=style_rec0)[0]
              if self.config.dataset_fake in ['CelebA', 'EmotionNet', 'BP4D', 'DEMO']:
                target_c = (target_c-_target_c)**2

            #Style 0 | progressive swap label
            elif Style==5: 
              for j, i in enumerate(range(real_x0.size(0))): 
                _style0[i] = style0[0]*0
                if n_label>0:
                  target_c[i][n_label-1].data.fill_((target_c[i][n_label-1]*label_space[j] + (1-target_c[i][n_label-1])*(1-label_space[j])).data[0])
                else:
                  target_c[i] = target_c[i]*label_space[j] + (1-target_c[i])*(1-label_space[j])          

            #Style constant | progressive label
            elif Style==6:
              for j, i in enumerate(range(real_x0.size(0))): 
                _style0[i] = style0[2].clone()
                target_c[i] = _target_c[i]*0.2*j   

            #Style random | One label at a time
            elif Style==7: 
              target_c = _target_c

            #Extract style from the two before, current, and two after. 
            elif Style==8:
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

          else:
            _style0 = None
            for j, i in enumerate(range(real_x0.size(0))): 
              target_c[i] = target_c[i]*label_space[j]            

          fake_x = self.G(real_x0, target_c, stochastic=_style0)
          if Style==3: 
            fake_x[0] = fake_x[0][:-1]
            fake_x[1] = fake_x[1][:-1]
          # if Style>=1: fake_x[0] = color_frame(fake_x[0], thick=5, first=n_label==1) #After off
          if Attention: fake_x[1] = fake_x[1].repeat(1,3,1,1)
          # if Style>=1 and Style!=3: 
          #   for n in range(len(fake_x)):
          #     if 'CelebA' in self.config.dataset_fake: fake_x[n] = circle_frame(fake_x[n], thick=5) #After off
          # elif Style==3: 
          #   for n in range(len(fake_x)):
          #     if 'CelebA' in self.config.dataset_fake: fake_x[n] = circle_frame(fake_x[n], thick=5, row_color=0, color='green')
          #     if 'CelebA' in self.config.dataset_fake: fake_x[n] = circle_frame(fake_x[n], thick=5, row_color=n_rep-1, color='blue')
          #     if 'CelebA' in self.config.dataset_fake: fake_x[n] = circle_frame(fake_x[n], thick=5, row_color=(n_rep*2)-2, color='red')
          fake_image_list.append(fake_x[0])
          if Attention: fake_attn_list.append(fake_x[1])
        if 'Stochastic' in self.config.GAN_options: Output.extend(self._SAVE_IMAGE(_save_path, fake_image_list, attn_list=fake_attn_list, im_size=self.config.image_size, gif=gif, mode='style_'+chr(65+idx), style=Style))
        if idx==self.config.iter_style or (training and idx==self.config.style_train_debug-1) or 'Stochastic' not in self.config.GAN_options: break
    self.G.train()
    self.D.train()
    if output: return Output 