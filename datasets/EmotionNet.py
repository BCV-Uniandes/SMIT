import torch
import os
import random
from torch.utils.data import Dataset
from PIL import Image

# ==================================================================#
# == EmotionNet
# ==================================================================#


class EmotionNet(Dataset):
    def __init__(self,
                 image_size,
                 mode_data,
                 transform,
                 mode,
                 shuffling=False,
                 verbose=False,
                 **kwargs):
        self.transform = transform
        self.mode = mode
        self.shuffling = shuffling
        self.verbose = verbose
        self.name = 'EmotionNet'
        self.image_size = image_size if image_size >= 128 else 130
        if self.verbose:
            self.ssd = '/home/afromero/ssd2/EmotionNet2018/faces/{}'.format(
                mode)
        else:
            self.ssd = '/home/afromero/ssd2/EmotionNet2018/data_{}/{}'.format(
                self.image_size, mode)
        file_txt = os.path.abspath(
            os.path.join('data', 'EmotionNet', mode_data, mode + '.txt'))
        if self.verbose:
            print("Data from: " + file_txt)
        self.lines = open(file_txt, 'r').readlines()

        if self.verbose:
            print('Start preprocessing %s: %s!' % (self.name, mode))
        random.seed(1)
        # random.seed(1234)
        self.preprocess()
        self.num_data = len(self.filenames)
        if self.verbose:
            print('Finished preprocessing %s: %s (%d)!' % (self.name, mode,
                                                           self.num_data))

    def preprocess(self):
        self.filenames = []
        self.labels = []
        lines = [i.strip() for i in self.lines]
        if self.mode == 'train' or self.shuffling:
            random.shuffle(lines)  # random shuffling
        for i, line in enumerate(lines):
            splits = line.split()
            filename = os.path.join(self.ssd, splits[0])
            # or os.stat(filename).st_size==0:
            if not os.path.isfile(filename):
                continue
            values = splits[1:]

            label = []
            for value in values:
                label.append(int(value))

            self.filenames.append(filename)
            self.labels.append(label)

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
