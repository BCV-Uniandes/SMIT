import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.transforms.functional import rotate, adjust_brightness, adjust_contrast, adjust_saturation, to_grayscale
from torchvision.datasets import ImageFolder
from PIL import Image
import ipdb
import numpy as np
import imageio
import glob
import torch.utils.data.distributed
import importlib
from misc.utils import _horovod
hvd = _horovod()   

######################################################################################################
###                                              LOADER                                            ###
######################################################################################################
def get_loader(metadata_path, image_size, batch_size, \
        dataset='BP4D', mode='train', shuffling = False, num_workers=0, HOROVOD=False, **kwargs):

  mean = (0.5,0.5,0.5)
  std  = (0.5,0.5,0.5)

  transform = []
  if 'face' in metadata_path or mode!='train':
    transform+=[transforms.Resize((image_size, image_size), interpolation=Image.ANTIALIAS)]
  elif dataset=='RafD' or dataset=='EmotionNet': 
    window = int(image_size/10)
    transform+=[transforms.Resize((image_size+window, image_size+window), interpolation=Image.ANTIALIAS)]
    # transform+=[transforms.CenterCrop((image_size, image_size))]
    transform+=[transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.8, 1.2))]
  else:
    window = int(image_size/10)
    transform+=[transforms.Resize((image_size+window, image_size+window), interpolation=Image.ANTIALIAS)]
    # transform+=[transforms.CenterCrop((image_size, image_size))]
    # transform+=[transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.2))] 
    transform+=[transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.8, 1.2))]    

  if dataset!='RafD' and mode=='train':  transform+=[transforms.RandomHorizontalFlip()]
  transform+=[transforms.ToTensor(), transforms.Normalize(mean, std)]  

  transform = transforms.Compose(transform)
  dataset_module = getattr(importlib.import_module('datasets.{}'.format(dataset)), dataset)
  dataset = dataset_module(image_size, metadata_path, transform, mode, shuffling=shuffling or mode=='train', **kwargs)
  if not HOROVOD:
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
  else:
    data_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())    
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=data_sampler, shuffle=False, num_workers=num_workers)
  return data_loader

######################################################################################################
######################################################################################################