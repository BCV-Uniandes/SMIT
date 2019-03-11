#!/usr/bin/ipython
from __future__ import print_function
import os
import argparse
from data_loader import get_loader
import glob
import config as cfg
import warnings
import sys
import torch
import horovod.torch as hvd
from misc.utils import PRINT, config_yaml
warnings.filterwarnings('ignore')

__DATASETS__ = [
    os.path.basename(line).split('.py')[0]
    for line in glob.glob('datasets/*.py')
]


def _PRINT(config):
    string = '------------ Options -------------'
    PRINT(config.log, string)
    for k, v in sorted(vars(config).items()):
        string = '%s: %s' % (str(k), str(v))
        PRINT(config.log, string)
    string = '-------------- End ----------------'
    PRINT(config.log, string)


def main(config):
    from torch.backends import cudnn
    # For fast training
    cudnn.benchmark = True
    cudnn.deterministic = True

    data_loader = get_loader(
        config.mode_data,
        config.image_size,
        config.batch_size,
        config.dataset_fake,
        config.mode,
        num_workers=config.num_workers,
        all_attr=config.ALL_ATTR,
        c_dim=config.c_dim)

    if config.mode == 'train':
        from train import Train
        Train(config, data_loader)
        from test import Test
        test = Test(config, data_loader)
        test(dataset=config.dataset_real)

    elif config.mode == 'test':
        from test import Test
        test = Test(config, data_loader)
        if config.DEMO_PATH:
            test.DEMO(config.DEMO_PATH)
        else:
            test(dataset=config.dataset_real)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument(
        '--dataset_fake', type=str, default='CelebA', choices=__DATASETS__)
    parser.add_argument(
        '--dataset_real', type=str, default='', choices=[''] + __DATASETS__)
    parser.add_argument(
        '--mode', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--color_dim', type=int, default=3)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=22)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=80)
    parser.add_argument(
        '--save_epoch', type=int, default=1)  # Save samples how many epochs
    parser.add_argument(
        '--model_epoch', type=int,
        default=2)  # Save models and weights every how many epochs
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--pretrained_model', type=str, default=None)

    parser.add_argument('--seed', type=int, default=10)

    # Path
    parser.add_argument('--log_path', type=str, default='./snapshot/logs')
    parser.add_argument(
        '--model_save_path', type=str, default='./snapshot/models')
    parser.add_argument(
        '--sample_path', type=str, default='./snapshot/samples')
    parser.add_argument('--DEMO_PATH', type=str, default='')
    parser.add_argument('--DEMO_LABEL', type=str, default='')

    # Generative
    parser.add_argument(
        '--MultiDis', type=int, default=3, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--g_conv_dim', type=int, default=32)
    parser.add_argument('--d_conv_dim', type=int, default=32)
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--lambda_cls', type=float, default=1.0)
    parser.add_argument('--lambda_rec', type=float, default=10.0)
    parser.add_argument('--lambda_idt', type=float, default=10.0)
    parser.add_argument('--lambda_mask', type=float, default=0.1)
    parser.add_argument('--lambda_mask_smooth', type=float, default=0.00001)

    parser.add_argument('--style_dim', type=int, default=20, choices=[20])
    parser.add_argument('--dc_dim', type=int, default=256, choices=[256])

    parser.add_argument('--DETERMINISTIC', action='store_true', default=False)
    parser.add_argument('--STYLE_ENCODER', action='store_true', default=False)
    parser.add_argument('--DC_TRAIN', action='store_true', default=False)
    parser.add_argument('--INIT_DC', action='store_true', default=False)
    parser.add_argument(
        '--SPLIT_DC', type=int, default=0, choices=[0, 2, 3, 4, 6, 8, 12])
    parser.add_argument(
        '--SPLIT_DC_REVERSE',
        type=int,
        default=0,
        choices=[0, 2, 3, 4, 6, 8, 12])
    parser.add_argument(
        '--upsample',
        type=str,
        default='bilinear',
        choices=['bilinear', 'nearest'])

    # Misc
    parser.add_argument('--DELETE', action='store_true', default=False)
    parser.add_argument('--NO_ATTENTION', action='store_true', default=False)
    parser.add_argument('--ALL_ATTR', type=int, default=0)
    parser.add_argument('--GPU', type=str, default='-1')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=10000)

    # Debug options
    parser.add_argument('--style_debug', type=int, default=4)
    parser.add_argument('--style_train_debug', type=int, default=9)
    parser.add_argument(
        '--style_label_debug', type=int, default=2, choices=[0, 1, 2])

    config = parser.parse_args()

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
        config.g_lr *= len(config.GPU)
        config.d_lr *= len(config.GPU)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    config_yaml(config, 'datasets/{}.yaml'.format(config.dataset_fake))
    config = cfg.update_config(config)
    if config.mode == 'train':
        if hvd.rank() == 0:
            PRINT(config.log, ' '.join(sys.argv))
            _PRINT(config)
        main(config)
        config.log.close()

    else:
        main(config)
