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
###                                              RafD                                              ###
######################################################################################################
class Birds(Dataset):
  #'PROBABLY' LABEL IS TAKEN AS ONE
  def __init__(self, image_size, metadata_path, transform, mode, shuffling=False, c_dim=12, **kwargs):
    from data.Birds.CUB_200_2011.birds import data
    self.birds = data(c_dim) #N most frequents
    self.transform = transform
    self.image_size = image_size
    self.shuffling = shuffling
    self.name = 'Birds'
    if mode!='val' and hvd.rank() == 0: print ('Start preprocessing %s: %s!'%(self.name, mode))
    random.seed(1234)
    self.preprocess()
    if mode!='val' and hvd.rank() == 0: print ('Finished preprocessing %s: %s (%d)!'%(self.name, mode, self.num_data))

  def preprocess(self):
    self.selected_attrs = ['HasBackPattern_solid', 'HasCrownColor_black', 'HasWingColor_grey', 'HasLegColor_black', 'HasWingShape_rounded-wings']
    # self.selected_attrs = self.birds.selected_attrs #N most frequents
    # ipdb.set_trace()
    self.idx_selected = [self.birds.attributes2idx[_attr] for _attr in self.selected_attrs]
    self.all_attrs = self.birds.idx2attributes.values()
    self.filenames = []
    self.labels = []
    self.bbox = []
    items = self.birds.images.items()
    if self.shuffling: random.shuffle(items) 
    for idx, line in items:
      line = os.path.abspath(os.path.join(self.birds.path, 'images', line))
      all_attr = self.birds.attributes[idx]
      reduced_attr = [all_attr[_idx] for _idx in self.idx_selected]
      self.filenames.append(line)
      self.labels.append(reduced_attr)
      self.bbox.append(self.birds.bboxes[idx]) # <x>, <y>, <width>, <height>

    self.num_data = len(self.filenames)
    # ipdb.set_trace()
  def get_data(self):
    return self.filenames, self.labels

  def __getitem__(self, index):
    image = Image.open(self.filenames[index]).convert('RGB')
    crop = (self.bbox[index][0], self.bbox[index][1], self.bbox[index][0]+self.bbox[index][2], self.bbox[index][1]+self.bbox[index][3])
    image = image.crop(crop) #  (left, upper, right, lower)
    label = self.labels[index]
    return self.transform(image), torch.FloatTensor(label), self.filenames[index]

  def __len__(self):
    return self.num_data    

  def shuffle(self, seed):
    random.seed(seed)
    random.shuffle(self.filenames)
    random.seed(seed)
    random.shuffle(self.labels)   
    random.seed(seed)
    random.shuffle(self.bbox)       