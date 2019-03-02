#!/usr/bin/ipython
from __future__ import print_function
import os
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--color_dim', type=int, default=3)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--GPU', type=str, default='-1')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_epochs_decay', type=int, default=40)
    parser.add_argument(
        '--save_epoch', type=int, default=1)  # Save samples how many epochs
    parser.add_argument('--pretrained_model', type=str, default=None)

    # Misc
    parser.add_argument('--DELETE', action='store_true', default=False)
    parser.add_argument('--ALL_ATTR', type=int, default=0)

    # Generative
    parser.add_argument('--MultiDis', type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--g_conv_dim', type=int, default=32)
    parser.add_argument('--d_conv_dim', type=int, default=32)
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--style_dim', type=int, default=20, choices=[20])
    parser.add_argument('--dc_dim', type=int, default=256, choices=[256])

    # Path
    parser.add_argument('--log_path', type=str, default='./snapshot/logs')
    parser.add_argument(
        '--model_save_path', type=str, default='./snapshot/models')
    parser.add_argument(
        '--sample_path', type=str, default='./snapshot/samples')

    config = parser.parse_args()

    if config.GPU == '-1':
        # Horovod
        torch.cuda.set_device(hvd.local_rank())
        config.GPU = [int(i) for i in range(hvd.size())]

    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU
        config.num_workers = 0
        config.GPU = [int(i) for i in config.GPU.split(',')]
        config.batch_size *= len(config.GPU)

    config.dataset_fake = 'CelebA'
    config_yaml(config, 'datasets/{}.yaml'.format(config.dataset_fake))
    config = cfg.update_config(config)
    main(config)
