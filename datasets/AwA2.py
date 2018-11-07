import torch
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import ipdb
import numpy as np
import glob  
from misc.utils import _horovod, PRINT
hvd = _horovod()   

######################################################################################################
###                                              AwA2                                              ###
######################################################################################################
class AwA2(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, shuffling=False, all_attr=-1, continuous=True, **kwargs):
    self.transform = transform
    self.image_size = image_size
    self.shuffling = shuffling
    self.name = 'AwA2'
    self.all_attr = all_attr
    self.metadata_path = metadata_path
    data_root = os.path.join('data','AwA2', 'Animals_with_Attributes2')
    self.lines = sorted(glob.glob(os.path.abspath(os.path.join(data_root,'JPEGImages','*', '*.jpg'))))
    _replace = lambda line: line.strip().replace('   ',' ').replace('  ',' ').split(' ')
    if continuous: self.cls2attr = np.array([[float (i) for i in _replace(line)] for line in open(os.path.join(data_root, 'predicate-matrix-continuous.txt')).readlines()])
    else: self.cls2attr = np.array([[int(i) for i in _replace(line)] for line in open(os.path.join(data_root, 'predicate-matrix-binary.txt')).readlines()])
    key = lambda line: (int(line.strip().split('\t')[0])-1, line.strip().split('\t')[1])
    self.idx2cls = {key(line)[0]: key(line)[1] for line in open(os.path.join(data_root, 'classes.txt')).readlines()}
    self.cls2idx = {key(line)[1]: key(line)[0] for line in open(os.path.join(data_root, 'classes.txt')).readlines()}
    self.idx2attr = {key(line)[0]: key(line)[1] for line in open(os.path.join(data_root, 'predicates.txt')).readlines()}
    self.attr2idx = {key(line)[1]: key(line)[0] for line in open(os.path.join(data_root, 'predicates.txt')).readlines()}    

    if mode!='val' and hvd.rank() == 0: print ('Start preprocessing %s: %s!'%(self.name, mode))
    random.seed(1234)
    self.preprocess()
    if mode!='val' and hvd.rank() == 0: print ('Finished preprocessing %s: %s (%d)!'%(self.name, mode, self.num_data))

  def histogram(self):
    # ipdb.set_trace()
    values = np.zeros(len(self.attr2idx))
    for line in self.lines:
      _cls = self.cls2idx[line.split('/')[-2]]
      attr = self.cls2attr[_cls]
      values += attr
    # ipdb.set_trace()
    keys_sorted = [key for key,value in sorted(self.attr2idx.items(), key= lambda kv: (kv[1],kv[0]))]
    dict_={}
    for key, value in zip(keys_sorted, values):
      dict_[key] = value      
    total = 0
    with open('datasets/{}_histogram_attributes.txt'.format(self.name), 'w') as f:
      for key,value in sorted(dict_.iteritems(), key = lambda kv: (kv[1],kv[0]), reverse=True):
        total+=value
        PRINT(f, '{} {}'.format(key,value))
      PRINT(f, 'TOTAL {}'.format(total))

  def preprocess(self):
    self.histogram()
    if self.all_attr==1: #ALL OF THEM
      self.selected_attrs = [
        'black', 'white', 'blue', 'brown', 'gray', 'orange', 'red', 'yellow', 'patches', 
        'spots', 'stripes', 'furry', 'hairless', 'toughskin', 'big', 'small', 'bulbous', 
        'lean', 'flippers', 'hands', 'hooves', 'pads', 'paws', 'longleg', 'longneck', 'tail', 
        'chewteeth', 'meatteeth', 'buckteeth', 'strainteeth', 'horns', 'claws', 'tusks', 
        'smelly', 'flys', 'hops', 'swims', 'tunnels', 'walks', 'fast', 'slow', 'strong', 
        'weak', 'muscle', 'bipedal', 'quadrapedal', 'active', 'inactive', 'nocturnal', 
        'hibernate', 'agility', 'fish', 'meat', 'plankton', 'vegetation', 'insects', 
        'forager', 'grazer', 'hunter', 'scavenger', 'skimmer', 'stalker', 'newworld', 
        'oldworld', 'arctic', 'coastal', 'desert', 'bush', 'plains', 'forest', 'fields', 
        'jungle', 'mountains', 'ocean', 'ground', 'water', 'tree', 'cave', 'fierce', 'timid', 
        'smart', 'group', 'solitary', 'nestspot', 'domestic'
      ]

    elif self.all_attr==0:
      self.selected_attrs = ['black', 'white', 'brown', 'stripes', 'water', 'tree'] 

    self.filenames = []
    self.labels = []

    lines = self.lines
    if self.shuffling: random.shuffle(lines) 
    for i, line in enumerate(lines):
      _class = os.path.basename(line).split('_')[0]
      _class_idx = self.cls2idx[_class]
      values = self.cls2attr[_class_idx]
      label = []
      for idx, value in enumerate(values):
        attr = self.idx2attr[idx]
        if attr in self.selected_attrs:
          if value == -1:
            label.append(0)
          else:
            label.append(value)
      self.filenames.append(line)
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