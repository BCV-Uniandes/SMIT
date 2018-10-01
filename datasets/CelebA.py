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
###                                              CelebA                                            ###
######################################################################################################
class CelebA(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, shuffling=False, all_attr=-1, **kwargs):
    self.transform = transform
    self.image_size = image_size
    self.shuffling = shuffling
    self.name = 'CelebA'
    self.all_attr = all_attr
    self.metadata_path = metadata_path
    self.lines = open(os.path.abspath('data/CelebA/list_attr_celeba.txt')).readlines()
    self.attr2idx = {}
    self.idx2attr = {}

    if mode!='val' and hvd.rank() == 0: print ('Start preprocessing %s: %s!'%(self.name, mode))
    random.seed(1234)
    self.preprocess()
    if mode!='val' and hvd.rank() == 0: print ('Finished preprocessing %s: %s (%d)!'%(self.name, mode, self.num_data))


  def preprocess(self):
    attrs = self.lines[1].split()
    for i, attr in enumerate(attrs):
      self.attr2idx[attr] = i
      self.idx2attr[i] = attr

    if self.all_attr==1:
      self.selected_attrs = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
        'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
        'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'
      ]
    elif self.all_attr==2:
      self.selected_attrs = [
        'Big_Lips', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
        'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Mustache',
        'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Rosy_Cheeks',
        'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Young'
      ]

    elif self.all_attr==0:
      self.selected_attrs = ['Eyeglasses', 'Male', 'Pale_Skin', 'Smiling', 'Young'] 
    self.filenames = []
    self.labels = []

    lines = self.lines[2:]
    # if self.shuffling: random.shuffle(lines) 
    for i, line in enumerate(lines):

      splits = line.split()
      img_size = '_'+str(self.image_size) if self.image_size==128 else '' 
      if os.path.isdir('/home/afromero/ssd2/CelebA'):
        if 'faces' in self.metadata_path:
          filename = os.path.abspath('/home/afromero/ssd2/CelebA/Faces/{}'.format(splits[0]))
        else:
          filename = os.path.abspath('/home/afromero/ssd2/CelebA/data{}/{}'.format(img_size, splits[0]))
      else:
        if 'faces' in self.metadata_path:
          filename = os.path.abspath('data/CelebA/Faces/{}'.format(splits[0]))
        else:
          filename = os.path.abspath('data/CelebA/data{}/{}'.format(img_size, splits[0]))
      if not os.path.isfile(filename): continue
      values = splits[1:]

      label = []
      for idx, value in enumerate(values):
        attr = self.idx2attr[idx]
        if attr in self.selected_attrs:
          if value == '1':
            label.append(1)
          else:
            label.append(0)
      self.filenames.append(filename)
      self.labels.append(label)

    self.num_data = len(self.filenames)

  def get_data(self):
    return self.filenames, self.labels

  def __getitem__(self, index):
    image = Image.open(self.filenames[index]).convert('RGB')
    label = self.labels[index]
    return self.transform(image), torch.FloatTensor(label), self.filenames[index]

  def __len__(self):
    return self.num_data    

  def shuffle(self, seed):
    random.seed(seed)
    random.shuffle(self.filenames)
    random.seed(seed)
    random.shuffle(self.labels)