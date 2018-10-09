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
class Image2Season(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, shuffling=False, all_attr=-1, **kwargs):
    self.transform = transform
    self.image_size = image_size
    self.shuffling = shuffling
    self.name = 'Image2Season'
    self.all_attr = all_attr
    self.metadata_path = metadata_path
    self.lines = sorted(glob.glob('data/{}/{}*/*.jpg'.format(self.name, mode)))
    self.attr2idx = {line.split('/')[-1].split('_')[1]:idx for idx, line in enumerate(sorted(glob.glob('data/{}/{}*'.format(self.name, mode))))}
    self.idx2attr = {idx:line.split('/')[-1].split('_')[1] for idx, line in enumerate(sorted(glob.glob('data/{}/{}*'.format(self.name, mode))))}
    if mode!='val' and hvd.rank() == 0: print ('Start preprocessing %s: %s!'%(self.name, mode))
    random.seed(1234)
    self.preprocess()
    if mode!='val' and hvd.rank() == 0: print ('Finished preprocessing %s: %s (%d)!'%(self.name, mode, self.num_data))

  def histogram(self):
    values = {key:0 for key in self.attr2idx.keys()}
    for line in self.lines:
      key = line.split('/')[-2].split('_')[1]
      values[key] += 1
    total=0
    with open('datasets/{}_histogram_attributes.txt'.format(self.name), 'w') as f:
      for key,value in sorted(values.iteritems(), key=lambda (k,v): (v,k), reverse=True):
        print(key, value)
        print>>f, '{}\t{}'.format(key,value)
        total+=value
      print>>f, 'TOTAL\t{}'.format(total)

  def preprocess(self):
    self.histogram()
    if self.all_attr==1:
      self.selected_attrs = [key for key,value in sorted(self.attr2idx.iteritems(), key=lambda (k,v): (v,k))]#self.attr2idx.keys()
      # ['beksinski', 'boudin', 'burliuk', 'cezanne', 'chagall', 'corot', 
      #  'earle', 'gauguin', 'hassam', 'levitan', 'monet', 'picasso', 'ukiyoe', 'vangogh']
    else:
      self.selected_attrs = ['autumn', 'spring', 'summer', 'winter']
    self.filenames = []
    self.labels = []

    if self.shuffling: random.shuffle(self.lines) 
    for i, line in enumerate(self.lines):
      filename = os.path.abspath(line)
      key = line.split('/')[-2].split('_')[1]
      if key not in self.selected_attrs: continue
      label = []
      for attr in self.selected_attrs:
        if attr==key:
          label.append(1)
        else:
          label.append(0)
      # ipdb.set_trace()
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