import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import ipdb
import numpy as np
import imageio
import scipy.misc
import glob

def hist_match(source, template):
  """
  Adjust the pixel values of a grayscale image such that its histogram
  matches that of a target image

  Arguments:
  -----------
    source: np.ndarray
      Image to transform; the histogram is computed over the flattened
      array
    template: np.ndarray
      Template image; can have different dimensions to source
  Returns:
  -----------
    matched: np.ndarray
      The transformed output image
  """

  oldshape = source.shape
  source = source.ravel()
  template = template.ravel()

  # get the set of unique pixel values and their corresponding indices and
  # counts
  s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                      return_counts=True)
  t_values, t_counts = np.unique(template, return_counts=True)

  # take the cumsum of the counts and normalize by the number of pixels to
  # get the empirical cumulative distribution functions for the source and
  # template images (maps pixel value --> quantile)
  s_quantiles = np.cumsum(s_counts).astype(np.float64)
  s_quantiles /= s_quantiles[-1]
  t_quantiles = np.cumsum(t_counts).astype(np.float64)
  t_quantiles /= t_quantiles[-1]

  # interpolate linearly to find the pixel values in the template image
  # that correspond most closely to the quantiles in the source image
  interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

  return interp_t_values[bin_idx].reshape(oldshape)

def get_resize(file_, img_size):
  org_file = file_.replace('BP4D_256', 'BP4D')
  new_file = file_.replace('BP4D_256', 'BP4D_'+str(img_size))
  dir_ = os.path.dirname(new_file)

  if not os.path.isdir(dir_): os.system('mkdir -p '+dir_)  

  if not os.path.isfile(new_file): 
    imageio.imwrite(new_file, scipy.misc.imresize(scipy.misc.imread(org_file), (img_size,img_size)))
  return new_file

class CelebDataset_Custom(Dataset):
  def __init__(self, image_size, metadata_path, transform):
    self.transform = transform
    self.image_size = image_size
    # self.lines = open(metadata_path, 'r').readlines()
    file_ = '/home/afromero/datos2/CelebA/Img/img_align_celeba/_data_aligned_{}.txt'.format(image_size)
    print("Reading from: "+file_)
    self.lines = open(file_).readlines()
    self.num_data = int(self.lines[0])
    self.attr2idx = {}
    self.idx2attr = {}

    print ('Start preprocessing dataset..!')
    random.seed(1234)
    self.preprocess()
    print ('Finished preprocessing dataset..!')


  def preprocess(self):

    # self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    # self.selected_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', \
    #           'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', \
    #           'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', \
    #           'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', \
    #           'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', \
    #           'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', \
    #           'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    self.filenames = []
    self.labels = []

    random.shuffle(lines)   # random shuffling
    for i, line in enumerate(lines):

      filename = line.strip()
      label = 1

      self.filenames.append(filename)
      self.labels.append(label)

  def __getitem__(self, index):
    image = Image.open(self.filenames[index])
    label = self.labels[index]
    
    return self.transform(image), torch.FloatTensor(label), self.filenames[index]

  def __len__(self):
    return self.num_data  


class CelebDataset(Dataset):
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
    self.selected_attrs = ['Male', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
    self.filenames = []
    self.labels = []

    lines = self.lines[2:]
    random.shuffle(lines)   # random shuffling
    for i, line in enumerate(lines):

      splits = line.split()
      filename = os.path.abspath('data/CelebA/Faces_aligned_{}/{}'.format(self.image_size, splits[0]))
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

  def __getitem__(self, index):
    # ipdb.set_trace()
    image = Image.open(self.filenames[index])
    label = self.labels[index]
    return self.transform(image), torch.FloatTensor(label), self.filenames[index]

  def __len__(self):
    return self.num_data


class MultiLabelAU(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, no_flipping = False, shuffling = False, MEAN=''):
    # ipdb.set_trace()
    self.transform = transform
    self.mode = mode
    self.no_flipping = no_flipping
    self.shuffling = shuffling
    self.image_size = image_size
    self.MEAN = MEAN
    file_txt = os.path.abspath(os.path.join(metadata_path, mode+'.txt'))
    print("Data from: "+file_txt)
    self.lines = open(file_txt, 'r').readlines()

    if mode!='val': print ('Start preprocessing dataset: %s!'%(mode))
    random.seed(1234)
    self.preprocess()
    if mode!='val': print ('Finished preprocessing dataset: %s!'%(mode))
    
    self.num_data = len(self.filenames)

  def preprocess(self):
    self.filenames = []
    self.labels = []
    lines = [i.strip() for i in self.lines]
    if self.mode=='train' or self.shuffling: random.shuffle(lines)   # random shuffling
    for i, line in enumerate(lines):
      splits = line.split()
      filename = splits[0]
      name = 'Faces' if not 'aligned' in filename else 'Faces_aligned'
      filename = filename.replace(name, name+'_'+str(self.image_size))
      # if self.no_flipping and 'flip' in filename: continue

      if self.image_size==512:
        filename_512 = filename.replace('BP4D_256', 'BP4D_'+str(self.image_size))
        if not os.path.isfile(filename_512): 

          filename = get_resize(filename_512, self.image_size)
        else: 
          filename = filename_512
      if not os.path.isfile(filename): imageio.imwrite(filename, np.zeros((self.image_size, self.image_size,3)).astype(np.uint8))
      # ipdb.set_trace()
      values = splits[1:]

      label = []
      for value in values:
        label.append(int(value))

      self.filenames.append(filename)
      self.labels.append(label)

  def __getitem__(self, index):
    # ipdb.set_trace()
    image = Image.open(self.filenames[index])
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
    random.seed(111)
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
    return self.transform(image), torch.FloatTensor([0]*12), self.filenames[index]

  def __len__(self):
    return self.num_data    

def get_loader(metadata_path, crop_size, image_size, batch_size, \
        dataset='MultiLabelAU', mode='train', LSTM=False, \
        shuffling = False, no_flipping=False, color_jitter=False, \
        mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), MEAN='', num_workers=0):
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
  elif dataset=='CelebA':
    dataset = CelebDataset(image_size, 'data', transform)
  elif dataset=='MultiLabelAU_FULL':
    dataset = MultiLabelAU_FULL(image_size, metadata_path, transform, mode, \
              no_flipping = no_flipping or LSTM, shuffling=shuffling)
  else:
    dataset = MultiLabelAU(image_size, metadata_path, transform, mode, \
              no_flipping = no_flipping or LSTM, shuffling=shuffling, MEAN=MEAN)

  # shuffle = shuffling
  # if mode == 'train' or shuffling:
  #  shuffle = True

  data_loader = DataLoader(dataset=dataset,
               batch_size=batch_size,
               shuffle=False,
               num_workers=num_workers)
  if LSTM: return data_loader, dataset.filenames
  else: return data_loader
