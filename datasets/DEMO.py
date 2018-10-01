import torch
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import ipdb
import numpy as np
import glob  
from misc.utils import _horovod
hvd = _horovod()   

######################################################################################################
###                                              DEMO                                              ###
######################################################################################################
class DEMO(Dataset):
  def __init__(self, image_size, img_path, transform, mode, shuffling=False, **kwargs):
    self.img_path = img_path
    self.transform = transform

    if os.path.isdir(img_path):
      self.lines = sorted(glob.glob(os.path.join(img_path, '*.jpg'))+glob.glob(os.path.join(img_path, '*.png')))
    else:
      self.lines = [self.img_path]
    self.len = len(self.lines)

  def __getitem__(self, index):
    image = Image.open(self.lines[index]).convert('RGB')
    # size = min(image.size)-1 if min(image.size)%2 else min(image.size)
    # image.thumbnail((size,size), Image.ANTIALIAS)
    return self.transform(image)

  def __len__(self):
    return self.len