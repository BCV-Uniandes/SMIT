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

class MultiLabelAU(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, no_flipping = False, shuffling = False, MEAN=''):
    # ipdb.set_trace()
    self.transform = transform
    self.mode = mode
    self.no_flipping = no_flipping
    self.shuffling = shuffling
    self.image_size = image_size
    self.MEAN = MEAN
    if 'emotionnet' in metadata_path.lower(): self.ssd = '/home/afromero/ssd2/EmotionNet2018/'+mode
    else: self.ssd = ''
    file_txt = os.path.abspath(os.path.join(metadata_path, mode+'.txt'))
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
      name = 'Faces' if not 'aligned' in filename else 'Faces_aligned'
      filename = filename.replace(name, name+'_256')#+str(self.image_size))
      # if self.no_flipping and 'flip' in filename: continue

      if not os.path.isfile(os.path.join(self.ssd, filename)): 
        ipdb.set_trace()
        imageio.imwrite(filename, np.zeros((self.image_size, self.image_size,3)).astype(np.uint8))
      # ipdb.set_trace()
      values = splits[1:]

      label = []
      for value in values:
        label.append(int(value))

      self.filenames.append(filename)
      self.labels.append(label)

  def __getitem__(self, index):
    # ipdb.set_trace()
    image = Image.open(os.path.join(self.ssd, self.filenames[index])).convert('RGB')
    label = self.labels[index]

    return self.transform(image), torch.FloatTensor(label), self.filenames[index]

  def __len__(self):
    return self.num_data

class MultiLabelAU_FULL(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, no_flipping = False, shuffling = False):
    # ipdb.set_trace()
    self.transform = transform
    self.mode = mode
    self.no_flipping = no_flipping
    self.shuffling = shuffling
    self.image_size = image_size
    self.lines = open(os.path.join(metadata_path, mode+'.txt'), 'r').readlines()
    self.img = [line.split(' ')[0] for line in self.lines]

    file_txt = os.path.abspath(os.path.join(metadata_path, mode+'_full.txt'))
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
      # if self.no_flipping and 'flip' in filename: continue

      if not os.path.isfile(filename): imageio.imwrite(filename, np.zeros((self.image_size, self.image_size,3)).astype(np.uint8))

      self.filenames.append(filename)

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
    file_txt = os.path.abspath(os.path.join('data/Google/data_faces{}.txt'.format(MODE)))
    print('Images from: '+file_txt)
    self.lines = open(file_txt, 'r').readlines()

    print ('Start preprocessing dataset: Google!')
    random.seed(11111)
    # random.seed()
    self.preprocess()
    print ('Finished preprocessing dataset: Google!')
    
    self.num_data = len(self.filenames)

  def preprocess(self):
    self.filenames = []
    self.labels = []
    lines = [i.strip() for i in self.lines]
    if self.shuffling: random.shuffle(lines)   # random shuffling
    for i, line in enumerate(lines):
      filename = line
      self.filenames.append(filename)

  def __getitem__(self, index):
    # image = Image.open(self.filenames[index])
    image = imageio.imread(self.filenames[index])
    # ipdb.set_trace()

    if not 'demo' in self.filenames[index]:
      # file_dir = os.path.dirname(self.filenames[index])
      # file_name = os.path.basename(self.filenames[index])
      # target_file = os.path.join(file_dir, 'demo0_Faces_aligned.jpg')
      name = 'normal' if self.mode!='aligned' else self.mode
      target_file = 'data/face_%s_mean.jpg'%(name)
      # print("Impose histogram from: "+target_file)
      # image = hist_match(image, imageio.imread(target_file))
    image = Image.fromarray(image.astype(np.uint8))
    # image = rotate(image, -13)
    # image = adjust_brightness(image, 0.7)
    # image = adjust_saturation(image, 1.5)
    # image = to_grayscale(image, num_output_channels=3)
    return self.transform(image), torch.FloatTensor([0]*12), self.filenames[index]

  def __len__(self):
    return self.num_data    

def get_loader(metadata_path, crop_size, image_size, batch_size, \
        dataset='MultiLabelAU', mode='train', LSTM=False, \
        shuffling = False, no_flipping=False, color_jitter=False, \
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
      # transforms.CenterCrop(crop_size),
      transforms.Resize((image_size,image_size), interpolation=Image.ANTIALIAS),
      # transforms.Scale(image_size, interpolation=Image.ANTIALIAS),
      transforms.ToTensor(),
      transforms.Normalize(mean, std)])

  if dataset=='Google':
    dataset = GooglePhotos(image_size, 'data', transform, mode=mode, shuffling=shuffling)
  else:
    dataset = MultiLabelAU(image_size, metadata_path, transform, mode, \
              no_flipping = no_flipping or LSTM, shuffling=shuffling)

  # shuffle = shuffling
  # if mode == 'train' or shuffling:
  #  shuffle = True

  data_loader = DataLoader(dataset=dataset,
               batch_size=batch_size,
               shuffle=False,
               num_workers=num_workers)
  if LSTM: return data_loader, dataset.filenames
  else: return data_loader
