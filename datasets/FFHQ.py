import torch
import os
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import numpy as np
from misc.utils import PRINT
from solver import Solver

# ==================================================================#
# == FFHQ
# ==================================================================#


class FFHQ(Dataset):
    def __init__(self,
                 image_size,
                 mode_data,
                 transform,
                 mode,
                 shuffling=False,
                 all_attr=0,
                 verbose=False,
                 **kwargs):
        self.transform = transform
        self.image_size = image_size
        self.shuffling = shuffling
        self.mode = mode
        self.name = 'FFHQ'
        self.all_attr = all_attr
        self.mode_data = mode_data
        self.verbose = verbose
        self.lines = [
            line.strip().split(',') for line in open(
                'data/FFHQ/list_attr_{}.txt'.format(all_attr)).readlines()
        ]
        self.all_attr2idx = {}
        self.all_idx2attr = {}
        self.attr2idx = {}
        self.idx2attr = {}

        if self.verbose:
            print('Start preprocessing %s: %s!' % (self.name, mode))
        random.seed(123)
        self.preprocess()
        if self.verbose:
            print('Finished preprocessing %s: %s (%d)!' % (self.name, mode,
                                                           self.num_data))

    def histogram(self):
        values = np.array([int(i) for i in self.lines[1][1:]]) * 0
        for line in self.lines[1:]:
            value = np.array([int(i) for i in line[1:]]).clip(min=0)
            values += value
        dict_ = {}
        for key, value in zip(self.lines[0][1:], values):
            dict_[key] = value
        total = 0
        with open('datasets/{}_histogram_attributes.txt'.format(self.name),
                  'w') as f:
            for key, value in sorted(
                    dict_.items(), key=lambda kv: (kv[1], kv[0]),
                    reverse=True):
                total += value
                PRINT(f, '{} {}'.format(key, value))
            PRINT(f, 'TOTAL {}'.format(total))

    def preprocess(self):
        attrs = self.lines[0][1:]
        if self.verbose:
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
        self.metadata_path = os.path.join('data', 'FFHQ', 'images1024x1024')
        lines = self.lines[1:]
        if self.shuffling:
            random.shuffle(lines)
        for i, line in enumerate(lines):
            filename = os.path.abspath('{}/{}'.format(self.metadata_path,
                                                      line[0]))
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

            self.filenames.append(line[0])
            self.labels.append(label)

        self.num_data = len(self.filenames)

    def get_data(self):
        return self.filenames, self.labels

    def __getitem__(self, index):
        filename = os.path.join(self.metadata_path, self.filenames[index])
        image = Image.open(filename).convert('RGB')
        label = self.labels[index]
        return self.transform(image), torch.FloatTensor(label), filename

    def __len__(self):
        return self.num_data

    def shuffle(self, seed):
        random.seed(seed)
        random.shuffle(self.filenames)
        random.seed(seed)
        random.shuffle(self.labels)


class Test(Solver):
    def __init__(self, config, data_loader):
        super(Test, self).__init__(config, data_loader)
        self.__call__()

    def imshow(self, img):
        from misc.utils import denorm, to_data
        import matplotlib.pyplot as plt
        for im in img:
            im = denorm(to_data(im, cpu=True)).numpy().transpose(1, 2, 0)
            plt.imshow(im)
            plt.show()
            break

    def __call__(self):

        from misc.utils import to_var, to_data
        from torchvision import transforms
        import torch.nn.functional as F
        import torch
        import tqdm

        metadata_path = os.path.join('data', 'FFHQ')
        self.metadata_path = os.path.join(metadata_path, 'images1024x1024')
        # inception Norm

        image_size = 256
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform = [
            transforms.Resize((image_size, image_size),
                              interpolation=Image.ANTIALIAS)
        ]
        transform += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        transform = transforms.Compose(transform)
        data = datasets.ImageFolder(metadata_path, transform)
        data_loader = DataLoader(
            data,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers)
        all_attr = self.data_loader.dataset.all_attr
        selected_attrs = ['image_id'] + self.data_loader.dataset.selected_attrs
        file_ = os.path.join(metadata_path,
                             'list_attr_{}.txt'.format(all_attr))
        print("Saving labels to " + file_)
        self.D.eval()

        progress_bar = tqdm.tqdm(
            enumerate(data_loader), total=len(data_loader))

        with torch.no_grad() and open(file_, 'w') as f:
            f.writelines(','.join(selected_attrs) + '\n')
            for _iter, (img, _) in progress_bar:
                files = [
                    data.imgs[(_iter * i) + i][0] for i in range(img.size(0))
                ]
                # self.imshow(img)
                img = to_var(img, volatile=True)
                _cls = [d.unsqueeze(-1) for d in self.D(img)[1]]
                _cls = torch.cat(_cls, dim=-1).sum(-1)
                _cls = (F.sigmoid(_cls) > 0.5) * 1
                _cls = to_data(_cls, cpu=True).numpy()
                for i in range(img.size(0)):
                    filename = os.path.basename(files[i])
                    labels = ','.join(map(str, _cls[i].tolist()))
                    f.writelines(filename + ',' + labels + '\n')
                    f.flush()
