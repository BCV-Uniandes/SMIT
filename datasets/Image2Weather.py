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
###                                              AwA2                                              ###
######################################################################################################
class Image2Weather(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, shuffling=False, all_attr=-1, continuous=True, **kwargs):
    self.transform = transform
    self.image_size = image_size
    self.shuffling = shuffling
    self.name = 'Image2Weather'
    self.all_attr = all_attr
    self.metadata_path = metadata_path
    data_root = os.path.join('data','Image2Weather', 'Image')
    self.lines = sorted(glob.glob(os.path.abspath(os.path.join(data_root,'*', '*.jpg'))))
    self.all_classes = [os.path.basename(line) for line in sorted(glob.glob(os.path.join(data_root,'*')))]
    # ipdb.set_trace()
    self.idx2cls = {idx:key for idx,key in enumerate(self.all_classes)}
    self.cls2idx = {key:idx for idx,key in enumerate(self.all_classes)}

    if mode!='val' and hvd.rank() == 0: print ('Start preprocessing %s: %s!'%(self.name, mode))
    random.seed(1234)
    self.preprocess()
    if mode!='val' and hvd.rank() == 0: print ('Finished preprocessing %s: %s (%d)!'%(self.name, mode, self.num_data))

  def histogram(self):
    # ipdb.set_trace()
    values = np.zeros(len(self.cls2idx))
    for line in self.lines:
      _cls = self.cls2idx[line.split('/')[-2]]
      values[_cls] += 1
    # ipdb.set_trace()
    keys_sorted = [key for key,value in sorted(self.cls2idx.iteritems(), key=lambda (k,v): (v,k))]
    self.hist={}
    for key, value in zip(keys_sorted, values):
      self.hist[key] = value      
    total = 0
    print('All attributes: '+str(keys_sorted))
    with open('datasets/{}_histogram_attributes.txt'.format(self.name), 'w') as f:
      for key,value in sorted(self.hist.iteritems(), key=lambda (k,v): (v,k), reverse=True):
        print(key, value)
        print>>f, '{}\t{}'.format(key,value)
        total+=value
      print>>f, 'TOTAL\t{}'.format(total)

    self.hist = {key:0 for key in self.selected_attrs}
    for line in self.lines:
      _cls = line.split('/')[-2]
      if _cls not in self.selected_attrs: continue
      self.hist[_cls] += 1      

  def preprocess(self):
    if self.all_attr==1: #ALL OF THEM
      self.selected_attrs = ['cloudy', 'foggy', 'rain', 'snow', 'sunny']

    elif self.all_attr==0:
      self.selected_attrs = ['cloudy', 'rain', 'snow', 'sunny'] #TNo Foggy due to scarce data
    self.histogram()
    # ipdb.set_trace()
    self.filenames = []
    self.labels = []

    lines = self.lines
    balanced = {key:0 for key in self.selected_attrs}
    if self.shuffling: random.shuffle(lines) 
    for i, line in enumerate(lines):
      _class = line.split('/')[-2]
      if _class not in self.selected_attrs: continue
      if self.all_attr==0 and balanced[_class]>=min(self.hist.values()): continue #Balancing all classes to the minimum
      label = []
      for idx, attr in enumerate(self.selected_attrs):
        balanced[attr]+=1
        if attr == _class:
          label.append(1)
        else:
          label.append(0)

      self.filenames.append(line)
      self.labels.append(label)
    # ipdb.set_trace()
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