#!/usr/bin/ipython
from __future__ import print_function
import os
import argparse
from data_loader import get_loader
import glob
import math
import os, glob, ipdb, imageio, numpy as np, config as cfg, warnings, sys
from misc.utils import _horovod, PRINT
hvd = _horovod()
hvd.init()
# warnings.filterwarnings('ignore')

__DATASETS__ = [os.path.basename(line).split('.py')[0] for line in glob.glob('datasets/*.py')]

def _PRINT(config):
  string ='------------ Options -------------'
  PRINT(config.log, string)
  for k, v in sorted(vars(config).items()):
    string = '%s: %s' % (str(k), str(v))
    PRINT(config.log, string)
  string='-------------- End ----------------'
  PRINT(config.log, string)

def main(config):
  from torch.backends import cudnn
  import torch  
  from solver import Solver
  # For fast training
  cudnn.benchmark = True

  data_loader = get_loader(config.metadata_path, config.image_size, config.batch_size, config.dataset, 
                 config.mode, num_workers=config.num_workers, all_attr = config.ALL_ATTR, c_dim=config.c_dim,
                 HOROVOD=config.HOROVOD, continuous='CLS_L2' in config.GAN_options, many_faces=config.many_faces,
                 RafD_FRONTAL=config.RafD_FRONTAL, RafD_EMOTIONS=config.RafD_EMOTIONS)   
  solver = Solver(config, data_loader)

  if config.DISPLAY_NET and config.mode_train=='GAN': 
    solver.display_net(name='discriminator')
    solver.display_net(name='generator')
    return

  if config.LPIPS_REAL: 
    solver.LPIPS_REAL()
    return        

  if config.LPIPS_UNIMODAL: 
    solver.LPIPS_UNIMODAL()
    return    

  if config.LPIPS_MULTIMODAL: 
    solver.LPIPS_MULTIMODAL()
    return  

  if config.INCEPTION: 
    solver.INCEPTION()
    return                     

  if config.mode == 'train':
    solver.train()
    solver.test(dataset=config.dataset_real, load=True)
    # solver.test_cls()
  elif config.mode == 'val':
    solver.val(load=True)
  elif config.mode == 'test':
    if config.DEMO_PATH:
      solver.DEMO(config.DEMO_PATH)    
    else:
      solver.test(dataset=config.dataset_real)

    # solver.val_cls(load=True)
    # solver.test_cls()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Model hyper-parameters
  parser.add_argument('--dataset_fake',       type=str, default='CelebA', choices=__DATASETS__)
  parser.add_argument('--dataset_real',       type=str, default='', choices=['']+__DATASETS__)  
  parser.add_argument('--dataset_smit',       type=str, default='')  
  parser.add_argument('--fold',               type=str, default='0')
  parser.add_argument('--mode_data',          type=str, default='normal', choices=['normal', 'faces'])   
  parser.add_argument('--mode_train',         type=str, default='GAN', choices=['GAN', 'CLS'])   
  parser.add_argument('--mode',               type=str, default='train', choices=['train', 'val', 'test']) 
  parser.add_argument('--c_dim',              type=int, default=12)
  parser.add_argument('--color_dim',          type=int, default=3)
  parser.add_argument('--image_size',         type=int, default=128)
  parser.add_argument('--batch_size',         type=int, default=64)
  parser.add_argument('--num_workers',        type=int, default=4)
  parser.add_argument('--num_epochs',         type=int, default=100)
  parser.add_argument('--num_epochs_decay',   type=int, default=30)
  parser.add_argument('--save_epoch',         type=int, default=1) #Save samples how many epochs
  parser.add_argument('--model_epoch',        type=int, default=5) #Save models and weights every how many epochs
  parser.add_argument('--beta1',              type=float, default=0.5)
  parser.add_argument('--beta2',              type=float, default=0.999)
  parser.add_argument('--pretrained_model',   type=str, default=None)  

  # Path
  parser.add_argument('--metadata_path',      type=str, default='./data')
  parser.add_argument('--log_path',           type=str, default='./snapshot/logs')
  parser.add_argument('--model_save_path',    type=str, default='./snapshot/models')
  parser.add_argument('--sample_path',        type=str, default='./snapshot/samples')
  parser.add_argument('--DEMO_PATH',          type=str, default='')
  parser.add_argument('--DEMO_LABEL',         type=str, default='')

  # Generative 
  parser.add_argument('--MultiDis',           type=int, default=0)
  parser.add_argument('--PerceptualLoss',     type=str, default='', choices=['', 'DeepFace', 'Style', 'EmoNet', 'ImageNet'])
  parser.add_argument('--g_conv_dim',         type=int, default=64)
  parser.add_argument('--d_conv_dim',         type=int, default=64)
  parser.add_argument('--g_repeat_num',       type=int, default=6)
  parser.add_argument('--d_repeat_num',       type=int, default=6)
  parser.add_argument('--g_lr',               type=float, default=0.0001)
  parser.add_argument('--d_lr',               type=float, default=0.0001)
  parser.add_argument('--lambda_cls',         type=float, default=1.0)  
  parser.add_argument('--lambda_rec',         type=float, default=10.0)
  parser.add_argument('--lambda_gp',          type=float, default=10.0)
  parser.add_argument('--lambda_perceptual',  type=float, default=10.0)
  parser.add_argument('--lambda_idt',         type=float, default=10.0)
  parser.add_argument('--lambda_l1',          type=float, default=1.0)
  parser.add_argument('--lambda_l1perceptual',type=float, default=0.1)  
  parser.add_argument('--lambda_style',       type=float, default=1.0)
  parser.add_argument('--lambda_mask',        type=float, default=0.1)
  parser.add_argument('--lambda_mask_smooth', type=float, default=0.00001)
  parser.add_argument('--lambda_kl',          type=float, default=0.001)
  parser.add_argument('--lambda_content',     type=float, default=1.0)

  parser.add_argument('--style_dim',          type=int, default=16, choices=[0, 1, 4, 8, 16, 20])

  parser.add_argument('--d_train_repeat',     type=int, default=5)

  # Generative settings
  parser.add_argument('--GAN_options',        type=str, default='')

  # Misc
  parser.add_argument('--use_tensorboard',    action='store_true', default=False)
  parser.add_argument('--DISPLAY_NET',        action='store_true', default=False) 
  parser.add_argument('--DELETE',             action='store_true', default=False)
  parser.add_argument('--FOLDER',             action='store_true', default=False)
  parser.add_argument('--LPIPS_REAL',         action='store_true', default=False)
  parser.add_argument('--LPIPS_UNIMODAL',     action='store_true', default=False)
  parser.add_argument('--LPIPS_MULTIMODAL',   action='store_true', default=False)
  parser.add_argument('--INCEPTION',          action='store_true', default=False)
  parser.add_argument('--ALL_ATTR',           type=int, default=0)
  # parser.add_argument('--GRAY_DISC',          action='store_true', default=False)
  # parser.add_argument('--GRAY_STYLE',         action='store_true', default=False)
  # parser.add_argument('--STYLE_DISC',         action='store_true', default=False)
  parser.add_argument('--LOAD_SMIT',          action='store_true', default=False)
  parser.add_argument('--NO_LABELCUM',        action='store_true', default=False)
  parser.add_argument('--RafD_FRONTAL',       action='store_true', default=False)
  parser.add_argument('--RafD_EMOTIONS',      action='store_true', default=False)
  parser.add_argument('--HOROVOD',            action='store_true', default=False)
  parser.add_argument('--DISC_DILATE',        action='store_true', default=False)
  parser.add_argument('--many_faces',         action='store_true', default=False)
  parser.add_argument('--GPU',                type=str, default='0')
  parser.add_argument('--Interpolation',      type=str, default='Linear', choices=['Linear', 'Spherical'])


  # Step size
  parser.add_argument('--log_step',           type=int, default=10)
  parser.add_argument('--sample_step',        type=int, default=500)
  parser.add_argument('--model_save_step',    type=int, default=10000)

  # Debug options
  parser.add_argument('--iter_test',          type=int, default=1)
  parser.add_argument('--iter_style',         type=int, default=40)
  parser.add_argument('--style_debug',        type=int, default=5)
  parser.add_argument('--style_train_debug',  type=int, default=9)
  parser.add_argument('--style_label_debug',  type=int, default=2, choices=[0,1,2,3,4,5,6,7])

  config = parser.parse_args()
  config.GAN_options = config.GAN_options.split(',')
  
  
  if config.HOROVOD:
    # mpirun -np -N_GPU ./main.py ...
    _GPU = config.GPU.split(',')
    config.GPU = [_GPU]
    print(hvd.local_rank(), hvd.size())
    config.g_lr = config.g_lr/hvd.size()
    config.d_lr = config.d_lr/hvd.size()    
    # torch.cuda.set_device(int(_GPU[hvd.local_rank()]))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_GPU[hvd.local_rank()])
  else:
    # torch.cuda.set_device(config.GPU[0])
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    config.GPU = [int(i) for i in config.GPU.split(',')]

  import torch
  if not torch.cuda.is_available():
    config.GPU='no_cuda'

  config = cfg.update_config(config)

  if config.FOLDER:
    if hvd.rank() == 0: 
      try: print(os.path.abspath(sorted(glob.glob(os.path.join(config.sample_path, '*.jpg')))[-config.style_train_debug-3]))
      except: print(os.path.abspath(sorted(glob.glob(os.path.join(config.sample_path, '*.jpg')))[-2]))
    sys.exit() 

  if config.mode=='train':
    # Create directories if not exist
    if not os.path.exists(config.log_path): os.makedirs(config.log_path)
    if not os.path.exists(config.model_save_path): os.makedirs(config.model_save_path)
    if not os.path.exists(config.sample_path): os.makedirs(config.sample_path)

    org_log = os.path.abspath(os.path.join(config.sample_path, 'log.txt'))
    config.loss_plot = os.path.abspath(os.path.join(config.sample_path, 'loss.txt'))
    os.system('touch '+org_log)

  else:
    org_log = 'logs/dummy.txt'

  if config.mode=='train':
    of = 'a' if os.path.isfile(org_log) else 'wb'
    with open(org_log, of) as config.log:
      if hvd.rank() == 0: PRINT(config.log, ' '.join(sys.argv))
      if hvd.rank() == 0: PRINT(config.log, config) 
      main(config)
    # last_sample = sorted(glob.glob(config.sample_path+'/*.jpg'))[-1]
    # os.system('echo {0} | mail -s "Training done - GPU {1} free" -A "{0}" rv.andres10@uniandes.edu.co'.format(msj, config.GPU))
  else:
    main(config)
