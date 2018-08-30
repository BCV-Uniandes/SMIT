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

def get_aus_emotionnet(metadata_path, mode):
  lines_aus = os.path.abspath(os.path.join(metadata_path.format('EmotionNet'), mode+'.txt'))
  lines_aus = [lines.strip().split(' ')[1:] for lines in open(lines_aus, 'r').readlines()]
  lines_aus = map(list,set(map(tuple, lines_aus)))

  labels_au = []
  for au in lines_aus:
    labels_au.append([])
    for value in au:
        if value == '1':
          labels_au[-1].append(1)
        else:
          labels_au[-1].append(0)

  return labels_au

######################################################################################################
###                                                BP4D                                            ###
######################################################################################################
class BP4D(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, shuffling = False, AUs=[]):
    self.transform = transform
    self.mode = mode
    self.shuffling = shuffling
    self.image_size = image_size
    self.AUs = AUs
    self.name = 'BP4D'
    self.labels_au = get_aus_emotionnet(metadata_path, mode)
    file_txt = os.path.abspath(os.path.join(metadata_path.format('BP4D'), mode+'.txt'))
    if mode!='val': print("Data from: "+file_txt)
    self.lines = open(file_txt, 'r').readlines()

    if mode!='val': print ('Start preprocessing %s: %s!'%(self.name, mode))
    random.seed(1234)
    # random.seed(10)
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
      # name = 'Faces' if not 'aligned' in filename else 'Faces_aligned'
      # filename = filename.replace('Faces', 'Faces_256')#+str(self.image_size))
      filename = filename.replace('Faces', 'Sequences_400')#+str(self.image_size))
      if not os.path.isfile(filename) or os.stat(filename).st_size==0: 
        # continue
        ipdb.set_trace()
        imageio.imwrite(filename, np.zeros((self.image_size, self.image_size,3)).astype(np.uint8))
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
###                                           BP4D_FULL                                            ###
######################################################################################################
class BP4D_FULL(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, no_flipping = False, shuffling = False, AUs=[]):
    self.transform = transform
    self.mode = mode
    self.no_flipping = no_flipping
    self.shuffling = shuffling
    self.image_size = image_size
    self.AUs = AUs    
    self.lines = open(os.path.join(metadata_path.format('BP4D'), mode+'.txt'), 'r').readlines()
    self.img = [line.split(' ')[0] for line in self.lines]

    file_txt = os.path.abspath(os.path.join(metadata_path.format('BP4D'), mode+'_full.txt'))
    if mode!='val': print("Data from: "+file_txt)
    self.lines_full = [i.strip() for i in open(file_txt, 'r').readlines()]
    subjects = sorted(list(set([line.split('/')[-3] for line in self.lines])))#Training subjects
    self.lines_full = [line for line in self.lines_full if line.split('/')[-3] in subjects and line not in self.img]
      
    print ('Start preprocessing BP4D: %s_full!'%(mode))
    random.seed(1234)
    self.preprocess()
    print ('Finished preprocessing BP4D: %s_full!'%(mode))
    
    self.num_data = len(self.filenames)

  def preprocess(self):
    self.filenames = []
    lines = [i.strip() for i in self.lines_full]
    if self.mode=='train' or self.shuffling: random.shuffle(lines)   # random shuffling
    for i, line in enumerate(lines):
      splits = line.split()
      filename = splits[0]
      name = 'Faces' if not 'aligned' in filename else 'Faces_aligned'
      filename = filename.replace(name, name+'_'+str(self.image_size))

      if not os.path.isfile(filename): 
        ipdb.set_trace()
        imageio.imwrite(filename, np.zeros((self.image_size, self.image_size,3)).astype(np.uint8))

      self.filenames.append(filename)

  def get_data(self):
    return self.filenames, self.labels

  def __getitem__(self, index):
    image = Image.open(self.filenames[index])

    return self.transform(image), torch.FloatTensor([0]*12), self.filenames[index]

  def __len__(self):
    return self.num_data

######################################################################################################
###                                              Google                                            ###
######################################################################################################
class GooglePhotos(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode='aligned', shuffling=False):
    self.transform = transform
    self.image_size = image_size
    self.mode = mode
    self.shuffling = shuffling
    MODE = '_aligned_{}'.format(image_size) if mode=='aligned' else ''
    # file_txt = os.path.abspath(os.path.join('data/Google/data_faces{}.txt'.format(MODE)))
    file_txt = os.path.abspath(os.path.join('data/Google/data.txt'))
    print('Images from: '+file_txt)
    self.lines = open(file_txt, 'r').readlines()
    self.lines = [line.replace('Faces', 'Org') for line in self.lines]

    print ('Start preprocessing dataset: Google!')
    random.seed(11111)
    # random.seed()
    self.preprocess()
    print ('Finished preprocessing dataset: Google!')
    
    self.num_data = len(self.filenames)

######################################################################################################
###                                            EmotionNet                                          ###
######################################################################################################
class EmotionNet(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, shuffling = False, AUs=[]):
    self.transform = transform
    self.mode = mode
    self.shuffling = shuffling
    self.AUs = AUs
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
        # continue
        ipdb.set_trace()
        imageio.imwrite(filename, np.zeros((self.image_size, self.image_size,3)).astype(np.uint8))
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
  def __init__(self, image_size, metadata_path, transform, mode, shuffling=False, AUs=[]):
    self.transform = transform
    self.image_size = image_size
    self.shuffling = shuffling
    self.AUs = AUs
    self.name = 'CelebA'
    # self.lines = open(metadata_path, 'r').readlines()
    # self.lines = open('/home/afromero/datos2/CelebA/Img/img_align_celeba/_data_aligned.txt').readlines()
    self.lines = open(os.path.abspath('data/CelebA/list_attr_celeba.txt')).readlines()
    self.labels_au = get_aus_emotionnet(metadata_path, mode)
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

    # self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    self.selected_attrs = ['Eyeglasses', 'Male', 'Mustache', 'No_Beard', 'Pale_Skin', 'Wearing_Hat', 'Young']
    self.filenames = []
    self.labels = []

    lines = self.lines[2:]
    if self.shuffling: random.shuffle(lines)   # random shuffling
    for i, line in enumerate(lines):

      splits = line.split()
      img_size = '_'+str(self.image_size) if self.image_size==128 else '' 
      if os.path.isdir('/home/afromero/ssd2/CelebA'):
        filename = os.path.abspath('/home/afromero/ssd2/CelebA/data{}/{}'.format(img_size, splits[0]))
        # filename = os.path.abspath('/home/afromero/ssd2/CelebA/Faces/{}'.format(splits[0]))
      else:
        filename = os.path.abspath('/home/roandres/bcv002/ssd2/CelebA/data{}/{}'.format(img_size, splits[0]))
      if not os.path.isfile(filename): continue
      values = splits[1:]

      label = []
      smile = False
      for idx, value in enumerate(values):
        attr = self.idx2attr[idx]
        # if attr=='Smiling' and value=='1': 
        #   smile=True
        #   break
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
  def __init__(self, image_size, img_path, transform, mode, shuffling=False, AUs=[]):
    self.img_path = img_path
    self.transform = transform

    if os.path.isdir(img_path):
      self.lines = glob.glob(os.path.join(img_path, '*.jpg'))+glob.glob(os.path.join(img_path, '*.png'))
    else:
      self.lines = [self.img_path]
    self.len = len(self.lines)

  def __getitem__(self, index):
    image = Image.open(self.lines[index]).convert('RGB')
    return self.transform(image)

  def __len__(self):
    return self.len

######################################################################################################
###                                              LOADER                                            ###
######################################################################################################
def get_loader(metadata_path, crop_size, image_size, batch_size, \
        dataset='BP4D', mode='train', \
        shuffling = False, color_jitter=False, AU=dict,\
        mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), num_workers=0):
  """Build and return data loader."""

  # resize = image_size + (image_size//20)
  crop_size = image_size 

  if 'face' in metadata_path:
    transform_resize = [transforms.Resize((crop_size+5, crop_size+5), interpolation=Image.ANTIALIAS)] if crop_size==64 else [] 
  else:
    transform_resize = [transforms.Resize((crop_size, crop_size), interpolation=Image.ANTIALIAS)] if crop_size==64 else [] 

  if mode == 'train':
    if color_jitter:
      transform = transforms.Compose([
        # transforms.Resize(resize, interpolation=Image.ANTIALIAS),
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.6, contrast=0.3, saturation=0.3, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])  
    else:
      transform = transforms.Compose(transform_resize + [
        # transform_resize,
        # transforms.Resize((crop_size, crop_size), interpolation=Image.ANTIALIAS),
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])  


  else:
    transform = transforms.Compose([
      transforms.Resize((crop_size, crop_size), interpolation=Image.ANTIALIAS),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)])

  dataset = globals()[dataset](image_size, metadata_path, transform, mode, shuffling=shuffling, AUs=AU[dataset.upper()])
  data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
  return data_loader

######################################################################################################
######################################################################################################