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

  crop_size = image_size 
  if 'face' in metadata_path: window = 0
  else: window = int(crop_size/10)
  horizontal_flip = [transforms.RandomHorizontalFlip()] if dataset!='RafD' else []
  if mode == 'train':
    transform = transforms.Compose([
      transforms.Resize((crop_size+window, crop_size+window), interpolation=Image.ANTIALIAS),
      transforms.CenterCrop((crop_size, crop_size)),
      ]+
      horizontal_flip+
      [
      transforms.ToTensor(),
      transforms.Normalize(mean, std)])  
  else:
    # if dataset!='DEMO': resize = [transforms.Resize((crop_size, crop_size), interpolation=Image.ANTIALIAS)]
    # else: resize = []
    resize = [transforms.Resize((crop_size, crop_size), interpolation=Image.ANTIALIAS)]
    transform = transforms.Compose(resize+[
      transforms.ToTensor(),
      transforms.Normalize(mean, std)])

  
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