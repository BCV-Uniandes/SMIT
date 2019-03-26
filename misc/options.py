import os
import glob

__DATASETS__ = [
    os.path.basename(line).split('.py')[0]
    for line in glob.glob('datasets/*.py')
]


def base_parser():
    import argparse
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
    parser.add_argument('--num_epochs', type=int, default=70)
    parser.add_argument('--num_epochs_decay', type=int, default=30)
    parser.add_argument(
        '--save_epoch', type=int, default=1)  # Save samples how many epochs
    parser.add_argument(
        '--model_epoch', type=int,
        default=5)  # Save models and weights every how many epochs
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--pretrained_model', type=str, default=None)

    parser.add_argument('--seed', type=int, default=1)

    # Path
    parser.add_argument('--log_path', type=str, default='./snapshot/logs')
    parser.add_argument(
        '--model_save_path', type=str, default='./snapshot/models')
    parser.add_argument(
        '--sample_path', type=str, default='./snapshot/samples')
    parser.add_argument('--DEMO_PATH', type=str, default='')
    parser.add_argument('--DEMO_LABEL', type=str, default='')

    # Generative
    parser.add_argument('--MultiDis', type=int, default=3, choices=[1, 2, 3])
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

    parser.add_argument('--DECONV', action='store_true', default=False)
    parser.add_argument('--INIT_DC', action='store_true', default=False)
    parser.add_argument('--DETERMINISTIC', action='store_true', default=False)

    # Misc
    parser.add_argument('--DELETE', action='store_true', default=False)
    parser.add_argument('--ALL_ATTR', type=int, default=0)
    parser.add_argument('--GPU', type=str, default='-1')

    # Scores
    parser.add_argument('--LPIPS_REAL', action='store_true', default=False)
    parser.add_argument('--LPIPS_UNIMODAL', action='store_true', default=False)
    parser.add_argument(
        '--LPIPS_MULTIMODAL', action='store_true', default=False)
    parser.add_argument('--INCEPTION', action='store_true', default=False)
    parser.add_argument('--INCEPTION_REAL', action='store_true', default=False)

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
    return config
