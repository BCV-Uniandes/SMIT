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

def get_resize(file_, img_size):
  org_file = file_.replace('BP4D_256', 'BP4D')
  new_file = file_.replace('BP4D_256', 'BP4D_'+str(img_size))
  dir_ = os.path.dirname(new_file)

  if not os.path.isdir(dir_): os.system('mkdir -p '+dir_)     

  if not os.path.isfile(new_file): 
    imageio.imwrite(new_file, scipy.misc.imresize(scipy.misc.imread(org_file), (img_size,img_size)))
  return new_file

class MultiLabelAU(Dataset):
    def __init__(self, image_size, metadata_path, transform, mode, no_flipping = False, shuffling = False):
        # ipdb.set_trace()
        self.transform = transform
        self.mode = mode
        self.no_flipping = no_flipping
        self.shuffling = shuffling
        self.image_size = image_size
        self.lines = open(os.path.join(metadata_path, mode+'.txt'), 'r').readlines()

        print ('Start preprocessing dataset: %s!'%(mode))
        random.seed(1234)
        self.preprocess()
        print ('Finished preprocessing dataset: %s!'%(mode))
        
        self.num_data = len(self.filenames)

    def preprocess(self):
        self.filenames = []
        self.labels = []
        lines = [i.strip() for i in self.lines]
        if self.mode=='train' or self.shuffling: random.shuffle(lines)   # random shuffling
        for i, line in enumerate(lines):
            splits = line.split()
            filename = splits[0]
            # filename = filename.replace('BP4D_256', 'BP4D_'+str(self.image_size))
            if self.no_flipping and 'flip' in filename: continue

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
        image = Image.open(self.filenames[index])
        label = self.labels[index]

        return self.transform(image), torch.FloatTensor(label), self.filenames[index]

    def __len__(self):
        return self.num_data

class GooglePhotos(Dataset):
    def __init__(self, image_size, metadata_path, transform):
        # ipdb.set_trace()
        self.transform = transform
        self.image_size = image_size
        self.lines = open(os.path.join(metadata_path, 'Google/data_aligned.txt'), 'r').readlines()

        print ('Start preprocessing dataset: Google!')
        random.seed(1234)
        self.preprocess()
        print ('Finished preprocessing dataset: Google!')
        
        self.num_data = len(self.filenames)

    def preprocess(self):
        self.filenames = []
        self.labels = []
        lines = [i.strip() for i in self.lines]
        # if self.mode=='train' or self.shuffling: random.shuffle(lines)   # random shuffling
        for i, line in enumerate(lines):
            filename = line
            self.filenames.append(filename)

    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        # ipdb.set_trace()
        return self.transform(image), torch.FloatTensor([0]*12), self.filenames[index]

    def __len__(self):
        return self.num_data        

def get_loader(metadata_path, crop_size, image_size, batch_size, dataset='MultiLabelAU', mode='train', LSTM=False, shuffling = False, no_flipping=False):
    """Build and return data loader."""

    if mode == 'train':
        transform = transforms.Compose([
            # transforms.CenterCrop(crop_size),
            transforms.Resize(image_size, interpolation=Image.ANTIALIAS),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            # transforms.CenterCrop(crop_size),
            # transforms.Resize(image_size, interpolation=Image.ANTIALIAS),
            transforms.Scale(image_size, interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if dataset=='Google':
        # ipdb.set_trace()
        dataset = GooglePhotos(image_size, 'data', transform)
    else:
        dataset = MultiLabelAU(image_size, metadata_path, transform, mode, no_flipping = no_flipping or LSTM, shuffling=shuffling)

    shuffle = False
    if mode == 'train' or shuffling:
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    if LSTM: return data_loader, dataset.filenames
    else: return data_loader
