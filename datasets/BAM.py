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
###                                              CelebA                                            ###
######################################################################################################
class BAM(Dataset):
  def __init__(self, image_size, metadata_path, transform, mode, shuffling=False, all_attr=0, **kwargs):
    self.transform = transform
    self.image_size = image_size
    self.shuffling = shuffling
    self.name = 'BAM'
    self.all_attr = all_attr
    self.metadata_path = metadata_path
    self.lines = open(os.path.abspath('data/{}/list_attr_{}.txt'.format(self.name, self.name.lower()))).readlines()
    self.attr2idx = {}
    self.idx2attr = {}
    if mode!='val' and hvd.rank() == 0: print ('Start preprocessing %s: %s!'%(self.name, mode))
    random.seed(1234)
    self.preprocess()
    if mode!='val' and hvd.rank() == 0: print ('Finished preprocessing %s: %s (%d)!'%(self.name, mode, self.num_data))

  def histogram(self):
    values = np.array([int(i) for i in self.lines[2].split()[1:]])*0
    for line in self.lines[2:]:
      value = np.array([int(i) for i in line.split()[1:]]).clip(min=0)
      values += value
    self.hist = {}
    for key, value in zip(self.lines[1].split(), values):
      self.hist[key] = value
    total = 0
    with open('datasets/{}_histogram_attributes.txt'.format(self.name), 'w') as f:
      for key,value in sorted(self.hist.iteritems(), key = lambda kv: (kv[1],kv[0]), reverse=True):
        total+=value
        PRINT(f, '{} {}'.format(key,value))
      PRINT(f, 'TOTAL {}'.format(total))



  def preprocess(self):
    attrs = self.lines[1].split()
    self.histogram()
    # ipdb.set_trace()
    for i, attr in enumerate(attrs):
      self.attr2idx[attr] = i
      self.idx2attr[i] = attr

    if self.all_attr==1:
      self.selected_attrs = attrs # Total: 20
    elif self.all_attr==2:
      self.selected_attrs = [
                        'emotion_gloomy', 'emotion_happy', 'emotion_peaceful', 'emotion_scary', 
                        'media_3d_graphics', 'media_comic', 'media_graphite', 'media_oilpaint', 
                        'media_pen_ink', 'media_vectorart', 'media_watercolor'      
                        ] #ALL OF THEM      
    else:
      self.selected_attrs = [
                        'content_bicycle', 'content_bird', 'content_building', 'content_cars', 
                        'content_cat', 'content_dog', 'content_flower', 'content_people', 'content_tree', 
                        'emotion_gloomy', 'emotion_happy', 'emotion_peaceful', 'emotion_scary', 
                        'media_3d_graphics', 'media_comic', 'media_graphite', 'media_oilpaint', 
                        'media_pen_ink', 'media_vectorart', 'media_watercolor'      
                        ] #ALL OF THEM
      # raise TypeError("Please specify attributes to train.")
    self.filenames = []
    self.labels = []

    lines = self.lines[2:]
    if self.shuffling: random.shuffle(lines) 
    for i, line in enumerate(lines):

      splits = line.split()
      filename = os.path.abspath('data/BAM/data2m/{}'.format(splits[0]))
      if not os.path.isfile(filename): continue#ipdb.set_trace()
      # try:
      #   success = Image.open(filename).convert('RGB')
      # except IOError:
      #   os.remove(filename)
      #   continue
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
    try:
      image = Image.open(self.filenames[index]).convert('RGB')
      label = self.labels[index]
    except:
      os.remove(self.filenames[index])
      image = Image.open(self.filenames[0]).convert('RGB')
      label = self.labels[0]
    
    return self.transform(image), torch.FloatTensor(label), self.filenames[index]

  def __len__(self):
    return self.num_data    

  def shuffle(self, seed):
    random.seed(seed)
    random.shuffle(self.filenames)
    random.seed(seed)
    random.shuffle(self.labels)

def create_img(url, img_file):
  try:
    img_data = requests.get(url).content
    with open(img_file, 'wb') as handler:
      handler.write(img_data)
    im1 = Image.open(img_file).convert('RGB')  
    im_small = im1.resize((256, 256), Image.ANTIALIAS)
    im_small.save(img_file)
    return True
  except:
    return False

def get_labels(all_attr, df, index):
  ru = lambda i: i.encode('ascii', 'ignore')
  text2idx = lambda text: 1 if text=='p' else 0
  labels = []
  for attr in all_attr:
    labels.append(text2idx(ru(df[attr][index])[0]))  
  return labels


if __name__ == '__main__':
  # mpirun -np 10 ipython datasets/BAM.py
  import pandas as pd, sqlite3, requests, random
  from PIL import Image
  from tqdm import tqdm  
  root = '/home/afromero/ssd2/BAM/'
  image_root = root+'data2m/'
  if not os.path.isdir(image_root): os.makedirs(image_root)
  sql_file = root+"20170509-bam-2.2m-Nja9G.sqlite"
  df = pd.read_sql('select * from modules, automatic_labels where modules.mid = automatic_labels.mid',\
                   sqlite3.connect(sql_file))
  
  ru = lambda i: i.encode('ascii', 'ignore')
  all_attr = pd.read_sql("select * from automatic_labels", sqlite3.connect(sql_file), index_col="mid")
  all_attr = sorted([ru(attr) for attr in all_attr.keys()])
  with open(root+'_list_attr_bam.txt', 'w')  as text:
    text.writelines('{}\t# Initially. There must be some broken link that were not included in this file\n'.format(len(df)))
    text.writelines('{}\n'.format('\t'.join(all_attr)))
    _range = range(len(df))
    random.shuffle(_range)

    for i in tqdm(_range, total=len(df), desc='Extracting images from BAM'):
      img_file = '{}.jpg'.format(image_root+str(i).zfill(len(str(len(df)))))
      if not os.path.isfile(img_file):
        # continue
        success = create_img(df['src'][i], img_file)
        if not success: continue

      txt_file = img_file.replace('jpg','txt')
      if not os.path.isfile(txt_file):
        # continue
        labels = get_labels(all_attr, df, i)
        string = '{}\n'.format('\t'.join([str(i) for i in labels]))
        with open(txt_file, 'w') as f: f.writelines(string)
      # else:
      #   labels = open(txt_file).readline().strip().split()
      #   if len(labels)!=20: continue
      #   string = '{}\t{}\n'.format(os.path.basename(img_file), '\t'.join(labels))
      #   text.writelines(string)


