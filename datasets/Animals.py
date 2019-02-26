import torch
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
from misc.utils import PRINT

# ==================================================================#
# == AwA2
# ==================================================================#


class Animals(Dataset):
    def __init__(self,
                 image_size,
                 mode_data,
                 transform,
                 mode,
                 shuffling=False,
                 all_attr=-1,
                 continuous=True,
                 verbose=False,
                 **kwargs):
        self.transform = transform
        self.image_size = image_size
        self.shuffling = shuffling
        self.name = 'Animals'
        self.mode = mode
        self.all_attr = all_attr
        self.mode_data = mode_data
        self.verbose = verbose
        data_root = os.path.join('data', 'Animals', 'Animals_with_Attributes2')
        self.lines = sorted(
            glob.glob(
                os.path.abspath(
                    os.path.join(data_root, 'JPEGImages', '*', '*.jpg'))))

        def _replace(line):
            return line.strip().replace('   ', ' ').replace('  ',
                                                            ' ').split(' ')

        def key(line):
            return (int(line.strip().split('\t')[0]) - 1,
                    line.strip().split('\t')[1])

        self.idx2cls = {
            key(line)[0]: key(line)[1]
            for line in open(os.path.join(data_root, 'classes.txt')).
            readlines()
        }
        self.cls2idx = {
            key(line)[1]: key(line)[0]
            for line in open(os.path.join(data_root, 'classes.txt')).
            readlines()
        }

        if self.verbose:
            print('Start preprocessing %s: %s!' % (self.name, mode))
        random.seed(1234)
        self.preprocess()
        if self.verbose:
            print('Finished preprocessing %s: %s (%d)!' % (self.name, mode,
                                                           self.num_data))

    def histogram(self):
        values = np.zeros(len(self.cls2idx))
        for line in self.lines:
            _cls = self.cls2idx[line.split('/')[-2]]
            values[_cls] += 1
        keys_sorted = [
            key for key, value in sorted(
                self.cls2idx.items(), key=lambda kv: (kv[1], kv[0]))
        ]
        dict_ = {}
        for key, value in zip(keys_sorted, values):
            dict_[key] = value
        total = 0
        print('All attributes: ' + str(keys_sorted))
        with open('datasets/{}_histogram_attributes.txt'.format(self.name),
                  'w') as f:
            for key, value in sorted(
                    dict_.items(), key=lambda kv: (kv[1], kv[0]),
                    reverse=True):
                total += value
                PRINT(f, '{} {}'.format(key, value))
            PRINT(f, 'TOTAL {}'.format(total))

    def preprocess(self):
        if self.verbose:
            self.histogram()
        if self.all_attr == 1:  # ALL OF THEM
            self.selected_attrs = [
                'antelope', 'grizzly+bear', 'killer+whale', 'beaver',
                'dalmatian', 'persian+cat', 'horse', 'german+shepherd',
                'blue+whale', 'siamese+cat', 'skunk', 'mole', 'tiger',
                'hippopotamus', 'leopard', 'moose', 'spider+monkey',
                'humpback+whale', 'elephant', 'gorilla', 'ox', 'fox', 'sheep',
                'seal', 'chimpanzee', 'hamster', 'squirrel', 'rhinoceros',
                'rabbit', 'bat', 'giraffe', 'wolf', 'chihuahua', 'rat',
                'weasel', 'otter', 'buffalo', 'zebra', 'giant+panda', 'deer',
                'bobcat', 'pig', 'lion', 'mouse', 'polar+bear', 'collie',
                'walrus', 'raccoon', 'cow', 'dolphin'
            ]

        else:
            self.selected_attrs = [
                'dalmatian',
                'german+shepherd',
                'collie',
                'wolf',
                'grizzly+bear',
                'gorilla',
                'giant+panda',
                'polar+bear',
                'antelope',
                'horse',
                'ox',
                'buffalo',
                'zebra',
                'cow',
                'tiger',
                'leopard',
                'lion',
            ]  # 17

        self.filenames = []
        self.labels = []

        lines = self.lines
        if self.shuffling or self.mode == 'test':
            random.shuffle(lines)
        for i, line in enumerate(lines):
            _class = os.path.basename(line).split('_')[0]
            if _class not in self.selected_attrs:
                continue
            label = []
            for idx, attr in enumerate(self.selected_attrs):
                if attr == _class:
                    label.append(1)
                else:
                    label.append(0)
            self.filenames.append(line)
            self.labels.append(label)

        self.num_data = len(self.filenames)

    def get_data(self):
        return self.filenames, self.labels

    def __getitem__(self, index):
        image = Image.open(self.filenames[index]).convert('RGB')
        label = self.labels[index]
        return self.transform(image), torch.FloatTensor(
            label), self.filenames[index]

    def __len__(self):
        return self.num_data

    def shuffle(self, seed):
        random.seed(seed)
        random.shuffle(self.filenames)
        random.seed(seed)
        random.shuffle(self.labels)
