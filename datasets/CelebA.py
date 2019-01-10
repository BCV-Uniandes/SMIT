import torch
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from misc.utils import PRINT

# ==================================================================#
# == CelebA
# ==================================================================#


class CelebA(Dataset):
    def __init__(self,
                 image_size,
                 mode_data,
                 transform,
                 mode,
                 shuffling=False,
                 all_attr=0,
                 **kwargs):
        self.transform = transform
        self.image_size = image_size
        self.shuffling = shuffling
        self.mode = mode
        self.name = 'CelebA'
        self.all_attr = all_attr
        self.mode_data = mode_data
        self.lines = [
            line.strip().split(',') for line in open(
                os.path.abspath('data/CelebA/list_attr_celeba.txt')).
            readlines()
        ]
        self.splits = {
            line.split(',')[0]: int(line.strip().split(',')[1])
            for line in open(
                os.path.abspath('data/CelebA/train_val_test.txt')).readlines()
            [1:]
        }
        self.mode_allowed = [0, 1] if mode == 'train' else [2]
        self.all_attr2idx = {}
        self.all_idx2attr = {}
        self.attr2idx = {}
        self.idx2attr = {}

        if mode != 'val':
            print('Start preprocessing %s: %s!' % (self.name, mode))
        random.seed(1234)
        self.preprocess()
        if mode != 'val':
            print('Finished preprocessing %s: %s (%d)!' % (self.name, mode,
                                                           self.num_data))

    def histogram(self):
        values = np.array([int(i) for i in self.lines[1][1:]]) * 0
        for line in self.lines[1:]:
            value = np.array([int(i) for i in line[1:]]).clip(min=0)
            values += value
        dict_ = {}
        for key, value in zip(self.lines[0], values):
            dict_[key] = value
        total = 0
        with open('datasets/{}_histogram_attributes.txt'.format(self.name),
                  'w') as f:
            for key, value in sorted(
                    dict_.items(), key=lambda kv: (kv[1], kv[0]),
                    reverse=True):
                total += value
                if self.mode == 'train':
                    PRINT(f, '{} {}'.format(key, value))
            if self.mode == 'train':
                PRINT(f, 'TOTAL {}'.format(total))

    def preprocess(self):
        attrs = self.lines[0][1:]
        self.histogram()

        for i, attr in enumerate(attrs):
            self.all_attr2idx[attr] = i
            self.all_idx2attr[i] = attr

        if self.all_attr == 1:
            self.selected_attrs = [
                '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',
                'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
                'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
                'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                'Wearing_Necktie', 'Young'
            ]  # Total: 40

        else:
            self.selected_attrs = [
                'Eyeglasses', 'Bangs', 'Black_Hair', 'Blond_Hair',
                'Brown_Hair', 'Gray_Hair', 'Male', 'Pale_Skin', 'Smiling',
                'Young'
            ]

        for i, attr in enumerate(self.selected_attrs):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr
        self.filenames = []
        self.labels = []

        lines = self.lines[1:]
        # if self.shuffling: random.shuffle(lines)
        for i, line in enumerate(lines):
            if self.splits[line[0]] not in self.mode_allowed:
                continue
            filename = os.path.abspath(
                'data/CelebA/img_align_celeba/{}'.format(line[0]))
            if not os.path.isfile(filename):
                continue
            values = line[1:]

            label = []

            for attr in self.selected_attrs:
                selected_value = values[self.all_attr2idx[attr]]
                if selected_value == '1':
                    label.append(1)
                else:
                    label.append(0)

            self.filenames.append(filename)
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
