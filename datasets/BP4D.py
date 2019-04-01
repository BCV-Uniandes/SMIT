import torch
import os
import random
from torch.utils.data import Dataset
from PIL import Image

# ==================================================================#
# == BP4D
# ==================================================================#


class BP4D(Dataset):
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
        self.image_size = image_size
        self.mode_data = mode_data
        self.verbose = verbose
        self.name = 'BP4D'
        file_txt = os.path.abspath(
            os.path.join('data', 'BP4D', mode_data, 'fold_0', mode + '.txt'))
        if self.verbose:
            print("Data from: " + file_txt)
        self.lines = open(file_txt, 'r').readlines()
        if self.verbose:
            print('Start preprocessing %s: %s!' % (self.name, mode))
        random.seed(1234)
        self.preprocess()
        self.num_data = len(self.filenames)
        if self.verbose:
            print('Finished preprocessing %s: %s (%d)!' % (self.name, mode,
                                                           self.num_data))

    def preprocess(self):
        self.filenames = []
        self.labels = []
        lines = [i.strip() for i in self.lines]
        random.shuffle(lines)
        for i, line in enumerate(lines):
            splits = line.split()
            filename = splits[0]
            if self.mode_data != 'faces':
                filename = filename.replace('Faces', 'Sequences_400')
            if not os.path.isfile(filename) or os.stat(filename).st_size == 0:
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
        image = Image.open(self.filenames[index])
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
