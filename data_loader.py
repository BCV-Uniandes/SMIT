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
import scipy.misc
import glob

######################################################################################################
###                                                BP4D                                            ###
######################################################################################################
class BP4D(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, shuffling = False, **kwargs):
    self.transform = transform
    self.mode = mode
    self.shuffling = shuffling
    self.image_size = image_size
    self.metadata_path = metadata_path
    self.name = 'BP4D'
    file_txt = os.path.abspath(os.path.join(metadata_path.format('BP4D'), mode+'.txt'))
    if mode!='val': print("Data from: "+file_txt)
    self.lines = open(file_txt, 'r').readlines()
    if mode!='val': print ('Start preprocessing %s: %s!'%(self.name, mode))
    random.seed(1234)
    self.preprocess()
    self.num_data = len(self.filenames)
    if mode!='val': print ('Finished preprocessing %s: %s (%d)!'%(self.name, mode, self.num_data))

  def preprocess(self):
    self.filenames = []
    self.labels = []
    lines = [i.strip() for i in self.lines]
    if self.mode=='train' or self.shuffling: random.shuffle(lines)   # random shuffling
    for i, line in enumerate(lines):
      splits = line.split()
      filename = splits[0]
      if not 'faces' in self.metadata_path:
        filename = filename.replace('Faces', 'Sequences_400')
      if not os.path.isfile(filename) or os.stat(filename).st_size==0: 
        ipdb.set_trace()
      values = splits[1:]
      label = []
      for value in values:
        label.append(int(value))
      self.filenames.append(filename)
      self.labels.append(label)

  def get_data(self):
    return self.filenames, self.labels

  def __getitem__(self, index):
    image = Image.open(self.filenames[index])
    label = self.labels[index]
    return self.transform(image), torch.FloatTensor(label), self.filenames[index]

  def __len__(self):
    return self.num_data

  def shuffle(self, seed):
    random.seed(seed)
    random.shuffle(self.filenames)
    random.seed(seed)
    random.shuffle(self.labels)  

######################################################################################################
###                                            EmotionNet                                          ###
######################################################################################################
class EmotionNet(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, shuffling = False, **kwargs):
    self.transform = transform
    self.mode = mode
    self.shuffling = shuffling
    self.name = 'EmotionNet'
    self.image_size = image_size if image_size>=128 else 130
    if os.path.isdir('/home/afromero/ssd2/EmotionNet2018'):
      if 'faces' in metadata_path:
        self.ssd = '/home/afromero/ssd2/EmotionNet2018/faces/{}'.format(mode)
      else:
        self.ssd = '/home/afromero/ssd2/EmotionNet2018/data_{}/{}'.format(self.image_size, mode)
    else:
      self.ssd = '/scratch_net/pengyou/Felipe/EmotionNet2018/data_{}/{}'.format(self.image_size, mode)

    file_txt = os.path.abspath(os.path.join(metadata_path.format('EmotionNet'), mode+'.txt'))
    if mode!='val': print("Data from: "+file_txt)
    self.lines = open(file_txt, 'r').readlines()

    if mode!='val': print ('Start preprocessing %s: %s!'%(self.name, mode))
    random.seed(1234)
    self.preprocess()
    self.num_data = len(self.filenames)
    if mode!='val': print ('Finished preprocessing %s: %s (%d)!'%(self.name, mode, self.num_data))

  def preprocess(self):
    self.filenames = []
    self.labels = []
    lines = [i.strip() for i in self.lines]
    if self.mode=='train' or self.shuffling: random.shuffle(lines)   # random shuffling
    for i, line in enumerate(lines):
      splits = line.split()
      filename = os.path.join(self.ssd, splits[0])
      if not os.path.isfile(filename):# or os.stat(filename).st_size==0: 
        ipdb.set_trace()
      values = splits[1:]

      label = []
      for value in values:
        label.append(int(value))

      self.filenames.append(filename)
      self.labels.append(label)

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

######################################################################################################
###                                              CelebA                                            ###
######################################################################################################
class CelebA(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, shuffling=False, all_attr=False):
    self.transform = transform
    self.image_size = image_size
    self.shuffling = shuffling
    self.name = 'CelebA'
    self.all_attr = all_attr
    self.metadata_path = metadata_path
    self.lines = open(os.path.abspath('data/CelebA/list_attr_celeba.txt')).readlines()
    self.attr2idx = {}
    self.idx2attr = {}

    if mode!='val': print ('Start preprocessing %s: %s!'%(self.name, mode))
    random.seed(1234)
    self.preprocess()
    if mode!='val': print ('Finished preprocessing %s: %s (%d)!'%(self.name, mode, self.num_data))


  def preprocess(self):
    attrs = self.lines[1].split()
    for i, attr in enumerate(attrs):
      self.attr2idx[attr] = i
      self.idx2attr[i] = attr
    #All attr: 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose 
    #          Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses 
    #          Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache 
    #          Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks 
    #          Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick 
    #          Wearing_Necklace Wearing_Necktie Young 
    # self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    if self.all_attr:
      self.selected_attrs = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
        'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
        'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'
      ]

    else:
      self.selected_attrs = ['Eyeglasses', 'Male', 'Pale_Skin', 'Smiling', 'Young'] 
    self.filenames = []
    self.labels = []

    lines = self.lines[2:]
    if self.shuffling: random.shuffle(lines) 
    for i, line in enumerate(lines):

      splits = line.split()
      img_size = '_'+str(self.image_size) if self.image_size==128 else '' 
      if os.path.isdir('/home/afromero/ssd2/CelebA'):
        if 'faces' in self.metadata_path:
          filename = os.path.abspath('/home/afromero/ssd2/CelebA/Faces/{}'.format(splits[0]))
        else:
          filename = os.path.abspath('/home/afromero/ssd2/CelebA/data{}/{}'.format(img_size, splits[0]))
      else:
        filename = os.path.abspath('/home/roandres/bcv002/ssd2/CelebA/data{}/{}'.format(img_size, splits[0]))
      if not os.path.isfile(filename): continue
      values = splits[1:]

      label = []
      smile = False
      for idx, value in enumerate(values):
        attr = self.idx2attr[idx]
        if attr in self.selected_attrs:
          if value == '1':
            label.append(1)
          else:
            label.append(0)
      if smile: continue
      self.filenames.append(filename)
      self.labels.append(label)

    self.num_data = len(self.filenames)

  def get_data(self):
    return self.filenames, self.labels

  def __getitem__(self, index):
    image = Image.open(self.filenames[index])
    label = self.labels[index]
    return self.transform(image), torch.FloatTensor(label), self.filenames[index]

  def __len__(self):
    return self.num_data    

  def shuffle(self, seed):
    random.seed(seed)
    random.shuffle(self.filenames)
    random.seed(seed)
    random.shuffle(self.labels)

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

######################################################################################################
###                                              LOADER                                            ###
######################################################################################################
def get_loader(metadata_path, image_size, batch_size, \
        dataset='BP4D', mode='train', all_attr=False,\
        shuffling = False, num_workers=0):

  mean = (0.5,0.5,0.5)
  std  = (0.5,0.5,0.5)

  crop_size = image_size 
  if 'face' in metadata_path: window = 0
  else: window = int(crop_size/10)

  if mode == 'train':
    transform = transforms.Compose([
      transforms.Resize((crop_size+window, crop_size+window), interpolation=Image.ANTIALIAS),
      transforms.CenterCrop((crop_size, crop_size)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)])  
  else:
    # if dataset!='DEMO': resize = [transforms.Resize((crop_size, crop_size), interpolation=Image.ANTIALIAS)]
    # else: resize = []
    resize = [transforms.Resize((crop_size, crop_size), interpolation=Image.ANTIALIAS)]
    transform = transforms.Compose(resize+[
      transforms.ToTensor(),
      transforms.Normalize(mean, std)])

  dataset = globals()[dataset](image_size, metadata_path, transform, mode, all_attr=all_attr, shuffling=shuffling)
  data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
  return data_loader

######################################################################################################
######################################################################################################