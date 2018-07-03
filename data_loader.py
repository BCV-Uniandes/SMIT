import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import rotate, adjust_brightness, adjust_contrast, adjust_saturation, to_grayscale
from torchvision.datasets import ImageFolder
from PIL import Image
import ipdb
import numpy as np
import imageio
import scipy.misc
import glob

class BP4D(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, no_flipping = False, shuffling = False, MEAN=''):
    # ipdb.set_trace()
    self.transform = transform
    self.mode = mode
    self.no_flipping = no_flipping
    self.shuffling = shuffling
    self.image_size = image_size
    self.MEAN = MEAN

    file_txt = os.path.abspath(os.path.join(metadata_path.format('BP4D'), mode+'.txt'))
    print("Data from: "+file_txt)
    self.lines = open(file_txt, 'r').readlines()

    if mode!='val': print ('Start preprocessing dataset: %s!'%(mode))
    random.seed(1234)
    # random.seed(10)
    self.preprocess()
    self.num_data = len(self.filenames)
    if mode!='val': print ('Finished preprocessing dataset: %s (%d)!'%(mode, self.num_data))
    # ipdb.set_trace()

  def preprocess(self):
    self.filenames = []
    self.labels = []
    lines = [i.strip() for i in self.lines]
    if self.mode=='train' or self.shuffling: random.shuffle(lines)   # random shuffling
    for i, line in enumerate(lines):
      splits = line.split()
      filename = splits[0]
      # name = 'Faces' if not 'aligned' in filename else 'Faces_aligned'
      # filename = filename.replace(name, name+'_256')#+str(self.image_size))
      filename = filename.replace('Faces', 'Sequences_400')#+str(self.image_size))
      if not os.path.isfile(filename) or os.stat(filename).st_size==0: 
        # continue
        ipdb.set_trace()
        imageio.imwrite(filename, np.zeros((self.image_size, self.image_size,3)).astype(np.uint8))
      # ipdb.set_trace()
      values = splits[1:]

      label = []
      for value in values:
        label.append(int(value))

      self.filenames.append(filename)
      self.labels.append(label)

  def get_data(self):
    return self.filenames, self.labels

  def __getitem__(self, index):
    # ipdb.set_trace()
    image = Image.open(self.filenames[index])
    return self.transform(image), torch.FloatTensor(label), self.filenames[index]

  def __len__(self):
    return self.num_data

class BP4D_FULL(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, no_flipping = False, shuffling = False):
    # ipdb.set_trace()
    self.transform = transform
    self.mode = mode
    self.no_flipping = no_flipping
    self.shuffling = shuffling
    self.image_size = image_size
    self.lines = open(os.path.join(metadata_path.format('BP4D'), mode+'.txt'), 'r').readlines()
    self.img = [line.split(' ')[0] for line in self.lines]

    file_txt = os.path.abspath(os.path.join(metadata_path.format('BP4D'), mode+'_full.txt'))
    print("Data from: "+file_txt)
    self.lines_full = [i.strip() for i in open(file_txt, 'r').readlines()]
    subjects = sorted(list(set([line.split('/')[-3] for line in self.lines])))#Training subjects
    self.lines_full = [line for line in self.lines_full if line.split('/')[-3] in subjects and line not in self.img]
      
    print ('Start preprocessing dataset: %s_full!'%(mode))
    random.seed(1234)
    self.preprocess()
    print ('Finished preprocessing dataset: %s_full!'%(mode))
    
    self.num_data = len(self.filenames)
    # ipdb.set_trace()

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
    # ipdb.set_trace()
    image = Image.open(self.filenames[index])

    return self.transform(image), torch.FloatTensor([0]*12), self.filenames[index]

  def __len__(self):
    return self.num_data

class GooglePhotos(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode='aligned', shuffling=False):
    # ipdb.set_trace()
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

class EmotionNet(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, shuffling = False):
    # ipdb.set_trace()
    self.transform = transform
    self.mode = mode
    self.shuffling = shuffling
    self.image_size = image_size
    self.ssd = '/home/afromero/ssd2/EmotionNet2018/data_128/'+mode
    file_txt = os.path.abspath(os.path.join(metadata_path.format('EmotionNet'), mode+'.txt'))
    print("Data from: "+file_txt)
    self.lines = open(file_txt, 'r').readlines()

    if mode!='val': print ('Start preprocessing dataset: %s!'%(mode))
    random.seed(1234)
    # random.seed(10)
    self.preprocess()
    self.num_data = len(self.filenames)
    if mode!='val': print ('Finished preprocessing dataset: %s (%d)!'%(mode, self.num_data))
    # ipdb.set_trace()

  def preprocess(self):
    self.filenames = []
    self.labels = []
    lines = [i.strip() for i in self.lines]
    if self.mode=='train' or self.shuffling: random.shuffle(lines)   # random shuffling
    for i, line in enumerate(lines):
      splits = line.split()
      filename = os.path.join(self.ssd, splits[0])
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
    # ipdb.set_trace()
    image = Image.open(self.filenames[index]).convert('RGB')
    label = self.labels[index]
    return self.transform(image), torch.FloatTensor(label), self.filenames[index]

  def __len__(self):
    return self.num_data

class CelebA(Dataset):
  def __init__(self, image_size, metadata_path, transform):
    self.transform = transform
    self.image_size = image_size
    # self.lines = open(metadata_path, 'r').readlines()
    # self.lines = open('/home/afromero/datos2/CelebA/Img/img_align_celeba/_data_aligned.txt').readlines()
    self.lines = open('data/CelebA/list_attr_celeba.txt').readlines()
    self.attr2idx = {}
    self.idx2attr = {}

    random.seed(1234)
    self.preprocess()
    print ('Finished preprocessing dataset..!')

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
    random.shuffle(lines)   # random shuffling
    for i, line in enumerate(lines):

      splits = line.split()
      filename = os.path.abspath('/home/afromero/ssd2/CelebA/data_{}/{}'.format(self.image_size, splits[0]))
      # ipdb.set_trace()
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
    # ipdb.set_trace()
    image = Image.open(self.filenames[index])
    label = self.labels[index]
    return self.transform(image), torch.FloatTensor(label), self.filenames[index]

  def __len__(self):
    return self.num_data    

class Fusion(Dataset):
  def __init__(self, filenames, labels, transform):
    self.transform = transform
    self.filenames = filenames
    self.labels = labels

  def __getitem__(self, index):
    image = Image.open(self.filenames[index]).convert('RGB')
    label = self.labels[index]
    return self.transform(image), torch.FloatTensor(label), self.filenames[index]

  def __len__(self):
    return self.filenames   

def get_loader(metadata_path, crop_size, image_size, batch_size, \
        dataset=['BP4D'], mode='train', \
        shuffling = False, color_jitter=False, \
        mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), num_workers=0):
  """Build and return data loader."""

  if mode == 'train':
    if color_jitter:
      transform = transforms.Compose([
        # transforms.CenterCrop(crop_size),
        transforms.Resize((image_size,image_size), interpolation=Image.ANTIALIAS),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.6, contrast=0.3, saturation=0.3, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])  
    else:
      transform = transforms.Compose([
        # transforms.CenterCrop(crop_size),
        transforms.Resize((image_size,image_size), interpolation=Image.ANTIALIAS),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])  


  else:
    transform = transforms.Compose([
      transforms.Resize((image_size,image_size), interpolation=Image.ANTIALIAS),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)])

  if dataset[0]=='Google':
    dataset = globals()["Google"](image_size, 'data', transform, mode=mode, shuffling=shuffling)
  elif len(dataset)==1:
    dataset = globals()[dataset[0]](image_size, metadata_path, transform, mode, shuffling=shuffling)
  elif len(dataset)==2:
    dataset1 = globals()[dataset[0]](image_size, metadata_path, transform, mode, shuffling=shuffling)
    images1, labels1 = dataset1.get_data()
    dataset2 = globals()[dataset[1]](image_size, metadata_path, transform, mode, shuffling=shuffling)
    images2, labels2 = dataset2.get_data()
    images = images1+images2
    labels = labels1+labels2
    dataset = Fusion(images, labels, transform)
  # shuffle = shuffling
  # if mode == 'train' or shuffling:
  #  shuffle = True

  data_loader = DataLoader(dataset=dataset,
               batch_size=batch_size,
               shuffle=False,
               num_workers=num_workers)
  return data_loader
