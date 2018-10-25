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
###                                              RafD                                              ###
######################################################################################################
class RafD(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, shuffling=False, RafD_EMOTIONS=False, RafD_FRONTAL=False, **kwargs):

    self.transform = transform
    self.image_size = image_size
    self.shuffling = shuffling
    self.name = 'RafD'
    self.FRONTAL = RafD_FRONTAL
    self.EMOTIONS = RafD_EMOTIONS
    data_root = os.path.join('data', 'RafD', '{}')
    data_root = data_root.format('faces') if 'faces' in metadata_path else data_root.format('data')
    self.lines = sorted(glob.glob(os.path.abspath(os.path.join(data_root,'*.jpg'))))
    if mode!='val' and hvd.rank() == 0: print ('Start preprocessing %s: %s!'%(self.name, mode))
    random.seed(1234)
    self.preprocess()
    if mode!='val' and hvd.rank() == 0: print ('Finished preprocessing %s: %s (%d)!'%(self.name, mode, self.num_data))

  def preprocess(self):
    self.pose = [0,45,90,135,180]

    if self.FRONTAL or self.EMOTIONS: 
      self.selected_attrs = ['neutral', 'angry', 'contemptuous', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']
      index = 0
    else:
      self.selected_attrs = ['pose_0', 'pose_45', 'pose_90', 'pose_135', 'pose_180',
                    'neutral', 'angry', 'contemptuous', 'disgusted', 'fearful', 'happy', 'sad', 'surprised']
      index = 5
    self.idx2cls = {idx:key for idx, key in enumerate(self.selected_attrs)}
    self.cls2idx = {key:idx for idx, key in enumerate(self.selected_attrs)}    
    self.filenames = []
    self.labels = []

    lines = self.lines
    if self.shuffling: random.shuffle(lines) 
    for i, line in enumerate(lines):
      _class = os.path.basename(line).split('_')[-2]
      pose = int(os.path.basename(line).split('_')[0].replace('Rafd',''))
      if self.FRONTAL and pose!=90 and not self.EMOTIONS: continue
      # label = [pose, self.cls2idx[_class]]
      label = []
      if not self.FRONTAL and not self.EMOTIONS:
        for _pose in self.pose:
          if _pose == pose:
            label.append(1)
          else:
            label.append(0)

      for value in self.selected_attrs[index:]:
        if _class == value:
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