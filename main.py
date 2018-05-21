#!/usr/local/bin/ipython
import os
import argparse
from data_loader import get_loader
import glob
import math
import ipdb
import imageio
import numpy as np
import config as cfg
import warnings
warnings.filterwarnings('ignore')
#CUDA_VISIBLE_DEVICES=0 ipython main.py -- --num_epochs 15 --batch_size 8 --image_size 256 --fold 0 --use_tensorboard --DYNAMIC_COLOR --CelebA_GAN

def str2bool(v):
  return v.lower() in ('true')

def main(config):
  from torch.backends import cudnn
  import torch  
  # For fast training
  cudnn.benchmark = True

  # Create directories if not exist
  if not os.path.exists(config.log_path):
    os.makedirs(config.log_path)
  if not os.path.exists(config.model_save_path):
    os.makedirs(config.model_save_path)
  if not os.path.exists(config.sample_path):
    os.makedirs(config.sample_path)
  # if not os.path.exists(config.result_path):
  #   os.makedirs(config.result_path)

  # Data loader
  CelebA_loader = None

  img_size = config.image_size

  MultiLabelAU_loader = get_loader(config.metadata_path, img_size,
                   img_size, config.batch_size, 'MultiLabelAU', config.mode, \
                   LSTM=config.LSTM, color_jitter=config.COLOR_JITTER, \
                   mean=config.mean, std=config.std, MEAN=config.MEAN, num_workers=config.num_workers)   

  if config.CLS:
    config.MultiLabelAU_FULL_loader = get_loader(config.metadata_path, img_size, 
                   img_size, config.batch_size, 'MultiLabelAU_FULL', config.mode, \
                   LSTM=config.LSTM, color_jitter=config.COLOR_JITTER, \
                   mean=config.mean, std=config.std, num_workers=config.num_workers)   

  # Solver
  if config.LSTM:
    from solver_lstm import Solver
  elif config.CLS:
    from solver_cls import Solver
  else:
    from solver import Solver    

  solver = Solver(MultiLabelAU_loader, config)

  if config.DISPLAY_NET: 
    solver.display_net(name='discriminator')
    solver.display_net(name='generator')
    return

  if config.mode == 'train':
    solver.train()
    solver.test()
    solver.test_cls()
  elif config.mode == 'val':
    solver.test()
  elif config.mode == 'test':
    # solver.val_cls(load=True)
    solver.test_cls()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Model hyper-parameters
  parser.add_argument('--c_dim', type=int, default=12)
  # parser.add_argument('--c2_dim', type=int, default=5)
  # parser.add_argument('--celebA_crop_size', type=int, default=178)
  # parser.add_argument('--rafd_crop_size', type=int, default=256)
  parser.add_argument('--image_size', type=int, default=256)
  parser.add_argument('--g_conv_dim', type=int, default=64)
  parser.add_argument('--d_conv_dim', type=int, default=64)
  parser.add_argument('--g_repeat_num', type=int, default=6)
  parser.add_argument('--d_repeat_num', type=int, default=6)
  parser.add_argument('--g_lr', type=float, default=0.0001)
  parser.add_argument('--d_lr', type=float, default=0.0001)
  parser.add_argument('--lambda_cls', type=float, default=1)
  parser.add_argument('--lambda_rec', type=float, default=10)
  parser.add_argument('--lambda_gp', type=float, default=10)
  parser.add_argument('--d_train_repeat', type=int, default=5)

  # Training settings
  parser.add_argument('--dataset', type=str, default='MultiLabelAU', choices=['CelebA', 'MultiLabelAU', 'RaFD', 'au01_fold0', 'Both'])
  parser.add_argument('--num_epochs', type=int, default=99)
  parser.add_argument('--num_epochs_decay', type=int, default=100)
  # parser.add_argument('--num_iters', type=int, default=300000)
  # parser.add_argument('--num_iters_decay', type=int, default=100000)
  parser.add_argument('--batch_size', type=int, default=12)
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--beta1', type=float, default=0.5)
  parser.add_argument('--beta2', type=float, default=0.999)
  parser.add_argument('--pretrained_model', type=str, default=None)  
  parser.add_argument('--FOCAL_LOSS', action='store_true', default=False)
  parser.add_argument('--JUST_REAL', action='store_true', default=False)
  parser.add_argument('--FAKE_CLS', action='store_true', default=False)
  parser.add_argument('--DENSENET', action='store_true', default=False)
  parser.add_argument('--CLS', action='store_true', default=False)
  parser.add_argument('--COLOR_JITTER', action='store_true', default=False)  
  parser.add_argument('--BLUR', action='store_true', default=False)   
  parser.add_argument('--GRAY', action='store_true', default=False)  
  parser.add_argument('--GOOGLE', action='store_true', default=False)   
  parser.add_argument('--PPT', action='store_true', default=False)  
  parser.add_argument('--VAL_SHOW', action='store_true', default=False)  
  parser.add_argument('--TEST', action='store_true', default=False)  
  # parser.add_argument('--CelebA_GAN', action='store_true', default=False)  
  parser.add_argument('--CelebA_CLS', action='store_true', default=False)  
  parser.add_argument('--LSGAN', action='store_true', default=False) 
  parser.add_argument('--L1_LOSS', action='store_true', default=False) 
  parser.add_argument('--L2_LOSS', action='store_true', default=False) 

  #Data Normalization
  parser.add_argument('--mean', type=str, default='0.5', choices=['0.5', 'data_image', 'data_full', 'data_full+image'])  
  parser.add_argument('--std', type=str, default='0.5', choices=['0.5', 'data_image', 'data_full', 'data_full+image'])  
  parser.add_argument('--G_norm', action='store_true', default=False)  
  parser.add_argument('--D_norm', action='store_true', default=False)  

  # Training LSTM
  parser.add_argument('--LSTM', action='store_true', default=False)
  parser.add_argument('--batch_seq', type=int, default=24)    

  # Test settings
  parser.add_argument('--test_model', type=str, default='')

  # Misc
  parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'])
  parser.add_argument('--use_tensorboard', action='store_true', default=False)
  parser.add_argument('--DISPLAY_NET', action='store_true', default=False) 
  parser.add_argument('--DELETE', action='store_true', default=False)
  parser.add_argument('--GPU', type=str, default='0')

  # Path
  parser.add_argument('--metadata_path', type=str, default='./data/MultiLabelAU')
  # parser.add_argument('--log_path', type=str, default='./stargan_MultiLabelAU_New/logs')
  # parser.add_argument('--model_save_path', type=str, default='./stargan_MultiLabelAU_New/models')
  # parser.add_argument('--sample_path', type=str, default='./stargan_MultiLabelAU_New/samples')
  # parser.add_argument('--result_path', type=str, default='./stargan_MultiLabelAU_New/results')
  parser.add_argument('--log_path', type=str, default='./stargan_MultiLabelAU/logs')
  parser.add_argument('--model_save_path', type=str, default='./stargan_MultiLabelAU/models')
  parser.add_argument('--sample_path', type=str, default='./stargan_MultiLabelAU/samples')
  # parser.add_argument('--result_path', type=str, default='./stargan_MultiLabelAU/results')  
  parser.add_argument('--fold', type=str, default='0')
  parser.add_argument('--mode_data', type=str, default='normal', choices=['normal', 'aligned'])  

  # Training Binary Classifier
  # parser.add_argument('--multi_binary', action='store_true', default=False)
  # parser.add_argument('--au', type=str, default='1')
  # parser.add_argument('--au_model', type=str, default='aunet')
  # parser.add_argument('--pretrained_model_generator', type=str, default='')


  # Step size
  parser.add_argument('--log_step', type=int, default=100)
  parser.add_argument('--sample_step', type=int, default=1000)
  parser.add_argument('--model_save_step', type=int, default=20000)

  config = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU

  config = cfg.update_config(config)

  print(config)
  main(config)

  last_sample = sorted(glob.glob(config.sample_path+'/*.jpg'))[-1]

  os.system('echo {0} | mail -s "Training done - GPU {1} free" -A "{0}" rv.andres10@uniandes.edu.co'.format(msj, config.GPU))
