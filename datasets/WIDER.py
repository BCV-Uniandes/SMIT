import torch
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import ipdb, json
import numpy as np
import glob 
from misc.utils import _horovod
hvd = _horovod()   
 
######################################################################################################
###                                              CelebA                                            ###
######################################################################################################
class WIDER(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, shuffling=False, all_attr=-1, **kwargs):
    self.transform = transform
    self.image_size = image_size
    self.shuffling = shuffling
    self.name = 'WIDER'
    self.all_attr = all_attr
    self.metadata_path = metadata_path
    _mode = 'trainval' if mode=='train' else 'test'
    json_file = 'data/WIDER/wider_attribute_annotation/wider_attribute_{}.json'.format(_mode)
    self.data_root = 'data/WIDER/Image'
    with open(json_file) as f: self.data = json.load(f)
    self.ru = lambda i: i.encode('ascii', 'ignore')
    self.attr2idx = {self.ru(j):int(self.ru(i)) for i,j in self.data['attribute_id_map'].iteritems()}    
    self.idx2attr = {int(self.ru(i)):self.ru(j) for i,j in self.data['attribute_id_map'].iteritems()}       
    if mode!='val' and hvd.rank() == 0: print ('Start preprocessing %s: %s!'%(self.name, mode))
    random.seed(1234)
    self.preprocess()
    if mode!='val' and hvd.rank() == 0: print ('Finished preprocessing %s: %s (%d)!'%(self.name, mode, self.num_data))

  def histogram(self):
    values = np.array(self.labels).sum(axis=0)
    values = {i:v for i,v in zip(self.selected_attrs, values)}
    total=0
    with open('datasets/{}_histogram_attributes.txt'.format(self.name), 'w') as f:
      for key,value in sorted(values.items(), key = lambda kv: (kv[1],kv[0]), reverse=True):
        total+=value
        PRINT(f, '{} {}'.format(key,value))
      PRINT(f, 'TOTAL {}'.format(total))

  def preprocess(self):
    self.all_filenames = []
    self.all_labels = []
    self.all_bbox = []
    for line in self.data['images']:
    	filename = self.ru(line['file_name'])
    	for target in line['targets']:
    		bbox = target['bbox']
        label = np.array(target['attribute']).clip(min=0) # 0 and -1 are negative labels
        self.all_filenames.append(filename)
        self.all_labels.append(label)
        self.all_bbox.append(bbox)

    if self.all_attr==1:
      self.selected_attrs = [key for key,value in sorted(self.attr2idx.items(),  key = lambda kv: (kv[1],kv[0]))]#self.attr2idx.keys()
      # ['Male', 'longHair', 'sunglass', 'Hat', 'Tshiirt', 'longSleeve', 'formal', 'shorts',
      #  'jeans', 'longPants', 'skirt', 'faceMask', 'logo', 'stripe']
    else:
      self.selected_attrs = ['Male', 'sunglass', 'Tshiirt', 'shorts', 'faceMask']
    self.filenames = []
    self.labels = []
    self.bbox = []
    _range = range(len(self.all_filenames))
    if self.shuffling: random.shuffle(_range) 
    for i in _range:
      filename = os.path.abspath(os.path.join(self.data_root, self.all_filenames[i]))
      key = self.all_labels[i]
      label = []
      for attr in self.selected_attrs:
        label.append(key[self.attr2idx[attr]])
      # ipdb.set_trace()
      self.filenames.append(filename)
      self.labels.append(label)
      self.bbox.append(self.all_bbox[i]) # [x, y, width, height]

    self.num_data = len(self.filenames)
    self.histogram()

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