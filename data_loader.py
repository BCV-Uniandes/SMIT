from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import importlib
import torch
from misc.utils import horovod
hvd = horovod()
hvd.init()

# ==================================================================#
# ==                           LOADER                             ==#
# ==================================================================#


def get_loader(mode_data,
               image_size,
               batch_size,
               dataset='BP4D',
               mode='train',
               shuffling=False,
               num_workers=0,
               HOROVOD=False,
               **kwargs):

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    transform = []
    if mode_data == 'faces' or mode != 'train':
        transform += [
            transforms.Resize((image_size, image_size),
                              interpolation=Image.ANTIALIAS)
        ]
    elif dataset == 'RafD' or dataset == 'EmotionNet':
        window = int(image_size / 10)
        transform += [
            transforms.Resize((image_size + window, image_size + window),
                              interpolation=Image.ANTIALIAS)
        ]
        transform += [
            transforms.RandomResizedCrop(
                image_size, scale=(0.7, 1.0), ratio=(0.8, 1.2))
        ]
    else:
        window = int(image_size / 10)
        transform += [
            transforms.Resize((image_size + window, image_size + window),
                              interpolation=Image.ANTIALIAS)
        ]
        transform += [
            transforms.RandomResizedCrop(
                image_size, scale=(0.7, 1.0), ratio=(0.8, 1.2))
        ]

    if dataset != 'RafD' and mode == 'train':
        transform += [transforms.RandomHorizontalFlip()]
    transform += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    transform = transforms.Compose(transform)
    dataset_module = getattr(
        importlib.import_module('datasets.{}'.format(dataset)), dataset)
    dataset = dataset_module(
        image_size,
        mode_data,
        transform,
        mode,
        shuffling=shuffling or mode == 'train',
        verbose=mode == 'train' and hvd.rank() == 0,
        **kwargs)
    if hvd.size() == 1:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
    elif hvd.size() != 1:
        if mode == 'train':
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=hvd.size(), rank=hvd.rank())
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=1, rank=0)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler)
    return data_loader
