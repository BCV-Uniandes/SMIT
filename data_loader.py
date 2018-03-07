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

class CelebDataset(Dataset):
    def __init__(self, image_path, metadata_path, transform, mode):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        self.lines = open(metadata_path, 'r').readlines()
        self.num_data = int(self.lines[0])
        self.attr2idx = {}
        self.idx2attr = {}

        print ('Start preprocessing dataset..!')
        random.seed(1234)
        self.preprocess()
        print ('Finished preprocessing dataset..!')

        if self.mode == 'train':
            self.num_data = len(self.train_filenames)
        elif self.mode == 'test':
            self.num_data = len(self.test_filenames)

    def preprocess(self):
        attrs = self.lines[1].split()
        for i, attr in enumerate(attrs):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr

        self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.train_filenames = []
        self.train_labels = []
        self.test_filenames = []
        self.test_labels = []

        lines = self.lines[2:]
        random.shuffle(lines)   # random shuffling
        for i, line in enumerate(lines):

            splits = line.split()
            filename = splits[0]
            values = splits[1:]

            label = []
            for idx, value in enumerate(values):
                attr = self.idx2attr[idx]

                if attr in self.selected_attrs:
                    if value == '1':
                        label.append(1)
                    else:
                        label.append(0)

            if (i+1) < 2000:
                self.test_filenames.append(filename)
                self.test_labels.append(label)
            else:
                self.train_filenames.append(filename)
                self.train_labels.append(label)

    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(os.path.join(self.image_path, self.train_filenames[index]))
            label = self.train_labels[index]
        elif self.mode in ['test']:
            image = Image.open(os.path.join(self.image_path, self.test_filenames[index]))
            label = self.test_labels[index]

        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_data

class MultiLabelAU(Dataset):
    def __init__(self, image_size, metadata_path, transform, mode, no_flipping = False):
        # ipdb.set_trace()
        self.transform = transform
        self.mode = mode
        self.no_flipping = no_flipping
        self.image_size = image_size
        self.lines = open(os.path.join(metadata_path, mode+'.txt'), 'r').readlines()
        # self.num_data = int(self.lines[0])
        # self.attr2idx = {}
        # self.idx2attr = {}

        print ('Start preprocessing dataset: %s!'%(mode))
        random.seed(1234)
        self.preprocess()
        print ('Finished preprocessing dataset: %s!'%(mode))
        
        self.num_data = len(self.filenames)

    def preprocess(self):
        # attrs = self.lines[1].split()
        # for i, attr in enumerate(attrs):
        #     self.attr2idx[attr] = i
        #     self.idx2attr[i] = attr

        # self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.filenames = []
        self.labels = []
        lines = [i.strip() for i in self.lines]
        random.shuffle(lines)   # random shuffling
        for i, line in enumerate(lines):
            splits = line.split()
            filename = splits[0]
            if self.no_flipping and 'flip' in filename: continue

            if self.image_size==512:
                filename_512 = filename.replace('256', str(self.image_size))
                if not os.path.isfile(filename_512): 
                    filename = get_resize(filename, self.image_size)
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

def get_loader(metadata_path, crop_size, image_size, batch_size, dataset='CelebA', mode='train', no_flipping=False):
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
            transforms.Scale(image_size, interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if dataset == 'au01_fold0':
        image_path = create_folder_au(metadata_path, image_size)
        dataset = ImageFolder(metadata_path, transform)        
    elif dataset == 'MultiLabelAU':
        dataset = MultiLabelAU(image_size, metadata_path, transform, mode, no_flipping = no_flipping)        

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader

def create_folder_rafd(image_path, image_size):
    import glob
    import scipy.misc
    imgs = glob.glob(os.path.join(image_path, '*.jpg'))
    emotions = set([i.split('_')[-2] for i in imgs])
    new_folder = os.path.join(image_path, 'sorted_em', str(image_size))
    for im in imgs:
        if '045' in im or '090' in im or '135' in im: 
            em = im.split('_')[-2]
            im = im.replace('Databases/RafD', 'Emotions/data/Faces/RafD')
            if not os.path.isfile(im): continue
            sorted_em = os.path.join(new_folder, em)
            if not os.path.isdir(sorted_em): os.makedirs(sorted_em)
            new_file = os.path.join(sorted_em, os.path.basename(im))
            if not os.path.isfile(new_file): 
                img_org = scipy.misc.imresize(scipy.misc.imread(im), [image_size, image_size])
                scipy.misc.imsave(new_file, img_org)
    return new_folder

def create_folder_au(image_path, image_size): 
    import scipy.misc    
    AU = os.path.basename(image_path).split('_')[-2].upper()
    train_path = os.path.join(image_path, 'train_'+AU+'.txt')
    test_path  = os.path.join(image_path, 'test_'+AU+'.txt')
    train_data = [i.strip().split(' ') for i in open(train_path).readlines()]
    test_data = [i.strip().split(' ') for i in open(test_path).readlines()]
    # ipdb.set_trace()
    train_img, train_label = balancing_labels(train_data)

    test_img   = [i[0] for i in test_data]
    test_label = [i[1] for i in test_data]    

    folder_mode = ['train', 'test']

    for idx in xrange(len(train_img)):
        if not os.path.isfile(train_img[idx]): continue
        folder_ = os.path.join(image_path, str(image_size), folder_mode[0], 'label_'+train_label[idx])
        if not os.path.isdir(folder_): os.makedirs(folder_)
        name = str(idx).zfill(6)+'.jpg'

        new_file = os.path.join(folder_, os.path.basename(name))
        if not os.path.isfile(new_file): 
            img_org = scipy.misc.imresize(scipy.misc.imread(train_img[idx]), [image_size, image_size])
            scipy.misc.imsave(new_file, img_org)   

    for idx in xrange(len(test_img)):
        if not os.path.isfile(test_img[idx]): continue
        folder_ = os.path.join(image_path, str(image_size), folder_mode[1], 'label_'+test_label[idx])
        if not os.path.isdir(folder_): os.makedirs(folder_)
        name = str(idx).zfill(6)+'.jpg'

        new_file = os.path.join(folder_, os.path.basename(name))
        if not os.path.isfile(new_file): 
            img_org = scipy.misc.imresize(scipy.misc.imread(test_img[idx]), [image_size, image_size])
            scipy.misc.imsave(new_file, img_org) 

    return os.path.join(image_path, str(image_size))                   

def balancing_labels(data, test=False):
    import numpy as np
    imgs   = [i[0] for i in data]
    labels = [i[1] for i in data]
    negative_count = 0
    positive_count = 0
    for idx in xrange(len(imgs)):
      if labels[idx]=='1': positive_count +=1
      if labels[idx]=='0': negative_count +=1

    print('Positive labels: '+str(positive_count))
    print('Negative labels: '+str(negative_count))      
    check = min(negative_count, positive_count)
    pos_check = 0
    neg_check = 0
    new_imgs = []
    new_labels = []
    for idx in xrange(len(imgs)):
      if labels[idx]=='1' and pos_check<check: 
        pos_check += 1
        new_imgs.append(imgs[idx])
        new_labels.append(labels[idx])

      if labels[idx]=='0' and neg_check<check: 
        neg_check += 1
        new_imgs.append(imgs[idx])
        new_labels.append(labels[idx])  

    print('Positive labels: '+str(pos_check))
    print('Negative labels: '+str(neg_check))   

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(new_imgs)
    np.random.seed(seed)
    np.random.shuffle(new_labels)

    return new_imgs, new_labels
