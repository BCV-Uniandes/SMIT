#!/usr/bin/ipython
from __future__ import print_function
import os
from data_loader import get_loader
import config as cfg
import warnings
import torch
import horovod.torch as hvd
from misc.utils import config_yaml
warnings.filterwarnings('ignore')


def _PRINT(config):
    string = '------------ Options -------------'
    print(string)
    for k, v in sorted(vars(config).items()):
        string = '%s: %s' % (str(k), str(v))
        print(string)
    string = '-------------- End ----------------'
    print(string)


def main(config):
    from torch.backends import cudnn
    # For fast training
    cudnn.benchmark = True

    data_loader = get_loader(
        config.mode_data,
        config.image_size,
        config.batch_size,
        config.dataset_fake,
        config.mode,
        num_workers=config.num_workers,
        all_attr=config.ALL_ATTR)

    from datasets.FFHQ import Test
    Test(config, data_loader)


if __name__ == '__main__':
    from options import base_parser
    config = base_parser()

    if config.GPU == '-1':
        # Horovod
        torch.cuda.set_device(hvd.local_rank())
        config.GPU = [int(i) for i in range(hvd.size())]
        config.g_lr *= hvd.size()
        config.d_lr *= hvd.size()

    else:
        if config.GPU == 'NO_CUDA':
            config.GPU = '-1'
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
        config.GPU = [int(i) for i in config.GPU.split(',')]
        config.batch_size *= len(config.GPU)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    config.dataset_fake = 'CelebA'
    config_yaml(config, 'datasets/{}.yaml'.format(config.dataset_fake))
    config = cfg.update_config(config)
    main(config)
