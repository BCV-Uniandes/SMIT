import os
import argparse
# import tensorflow as tf
from data_loader import get_loader
from torch.backends import cudnn
import glob
import math
def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    # Data loader
    celebA_loader = None
    rafd_loader = None
    au_loader = None

    if config.multi_binary:
        img_size = 224
    else:
        img_size = config.image_size

    if config.dataset in ['MultiLabelAU']:
        MultiLabelAU_loader = get_loader(config.metadata_path, img_size,
                                   img_size, config.batch_size, 'MultiLabelAU', config.mode)        

    if config.dataset in ['au01_fold0']:
        au_loader = get_loader(config.metadata_path, img_size,
                                         img_size, config.batch_size, 'au01_fold0', config.mode)        

    # Solver
    if config.CLS:
        from solver_cls import Solver
    else:
        from solver import Solver        
    solver = Solver(MultiLabelAU_loader, au_loader, config)

    if config.mode == 'train':
        solver.train()
        solver.test_cls()
    elif config.mode == 'test':
        solver.test_cls()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--c_dim', type=int, default=12)
    parser.add_argument('--c2_dim', type=int, default=8)
    parser.add_argument('--celebA_crop_size', type=int, default=178)
    parser.add_argument('--rafd_crop_size', type=int, default=256)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--lambda_cls', type=float, default=1)
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--d_train_repeat', type=int, default=5)

    # Training settings
    parser.add_argument('--dataset', type=str, default='MultiLabelAU', choices=['CelebA', 'MultiLabelAU', 'RaFD', 'au01_fold0', 'Both'])
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_epochs_decay', type=int, default=20)
    parser.add_argument('--num_iters', type=int, default=200000)
    parser.add_argument('--num_iters_decay', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--pretrained_model', type=str, default=None)    
    parser.add_argument('--FOCAL_LOSS', action='store_true', default=False)
    parser.add_argument('--JUST_REAL', action='store_true', default=False)
    parser.add_argument('--FAKE_CLS', action='store_true', default=False)
    parser.add_argument('--DENSENET', action='store_true', default=False)
    parser.add_argument('--CLS', action='store_true', default=False)

    # Training LSTM
    parser.add_argument('--LSTM', action='store_true', default=False)
    parser.add_argument('--batch_seq', type=int, default=24)        

    # Test settings
    parser.add_argument('--test_model', type=str, default='')

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--metadata_path', type=str, default='./data/MultiLabelAU')
    # parser.add_argument('--log_path', type=str, default='./stargan_MultiLabelAU_New/logs')
    # parser.add_argument('--model_save_path', type=str, default='./stargan_MultiLabelAU_New/models')
    # parser.add_argument('--sample_path', type=str, default='./stargan_MultiLabelAU_New/samples')
    # parser.add_argument('--result_path', type=str, default='./stargan_MultiLabelAU_New/results')
    parser.add_argument('--log_path', type=str, default='./stargan_MultiLabelAU/logs')
    parser.add_argument('--model_save_path', type=str, default='./stargan_MultiLabelAU/models')
    parser.add_argument('--sample_path', type=str, default='./stargan_MultiLabelAU/samples')
    parser.add_argument('--result_path', type=str, default='./stargan_MultiLabelAU/results')    
    parser.add_argument('--fold', type=str, default='0')
    parser.add_argument('--mode_data', type=str, default='aligned') 

    # Training Binary Classifier
    parser.add_argument('--multi_binary', action='store_true', default=False)
    parser.add_argument('--au', type=str, default='1')
    parser.add_argument('--au_model', type=str, default='aunet')
    parser.add_argument('--pretrained_model_generator', type=str, default='')


    # Step size
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=30000)

    config = parser.parse_args()

    config.metadata_path = os.path.join(config.metadata_path, config.mode_data, 'fold_'+config.fold, )

    config.log_path = os.path.join(config.log_path, config.mode_data, str(config.image_size), 'fold_'+config.fold)
    config.sample_path = os.path.join(config.sample_path, config.mode_data, str(config.image_size), 'fold_'+config.fold)
    config.model_save_path = os.path.join(config.model_save_path, config.mode_data, str(config.image_size), 'fold_'+config.fold)
    config.result_path = os.path.join(config.result_path, config.mode_data, str(config.image_size), 'fold_'+config.fold)

    if config.FOCAL_LOSS:
        config.log_path = os.path.join(config.log_path, 'Focal_Loss')
        config.sample_path = os.path.join(config.sample_path, 'Focal_Loss')
        config.model_save_path =os.path.join(config.model_save_path, 'Focal_Loss')
        config.result_path = os.path.join(config.result_path, 'Focal_Loss')

    if config.JUST_REAL:
        config.log_path = os.path.join(config.log_path, 'JUST_REAL')
        config.sample_path = os.path.join(config.sample_path, 'JUST_REAL')
        config.model_save_path =os.path.join(config.model_save_path, 'JUST_REAL')
        config.result_path = os.path.join(config.result_path, 'JUST_REAL')  

    if config.FAKE_CLS:
        config.log_path = os.path.join(config.log_path, 'FAKE_CLS')
        config.sample_path = os.path.join(config.sample_path, 'FAKE_CLS')
        config.model_save_path =os.path.join(config.model_save_path, 'FAKE_CLS')
        config.result_path = os.path.join(config.result_path, 'FAKE_CLS')    

    if config.CLS:
        config.pretrained_model_generator = sorted(glob.glob(os.path.join(config.model_save_path, '*_G.pth')))[-1]
        config.pretrained_model_discriminator = sorted(glob.glob(os.path.join(config.model_save_path, '*_D.pth')))[-1]

        config.log_path = config.log_path.replace('MultiLabelAU', 'MultiLabelAU_CLS')
        config.sample_path = config.sample_path.replace('MultiLabelAU', 'MultiLabelAU_CLS')
        config.model_save_path = config.model_save_path.replace('MultiLabelAU', 'MultiLabelAU_CLS')
        config.result_path = config.result_path.replace('MultiLabelAU', 'MultiLabelAU_CLS')
        

    if config.DENSENET:
        config.log_path = os.path.join(config.log_path, 'DENSENET')
        config.sample_path = os.path.join(config.sample_path, 'DENSENET')
        config.model_save_path =os.path.join(config.model_save_path, 'DENSENET')
        config.result_path = os.path.join(config.result_path, 'DENSENET')                        

    if config.lambda_cls!=1 or config.d_train_repeat!=5:
        config.log_path = os.path.join(config.log_path, 'lambda_cls_%d_d_repeat_%d'%(config.lambda_cls, config.d_train_repeat))
        config.sample_path = os.path.join(config.sample_path, 'lambda_cls_%d_d_repeat_%d'%(config.lambda_cls, config.d_train_repeat))
        config.model_save_path =os.path.join(config.model_save_path, 'lambda_cls_%d_d_repeat_%d'%(config.lambda_cls, config.d_train_repeat))
        config.result_path = os.path.join(config.result_path, 'lambda_cls_%d_d_repeat_%d'%(config.lambda_cls, config.d_train_repeat))   


    config.g_repeat_num = int(math.log(config.image_size,2)-1)
    config.d_repeat_num = int(math.log(config.image_size,2)-1)

    if config.pretrained_model is None:
        if config.LSTM:
            try:
                config.pretrained_model = sorted(glob.glob(os.path.join(config.model_save_path, '*_LSTM.pth')))[-1]
                config.pretrained_model = '_'.join(os.path.basename(config.pretrained_model).split('_')[:2])
            except:
                pass
        else:            
            try:
                config.pretrained_model = sorted(glob.glob(os.path.join(config.model_save_path, '*_D.pth')))[-1]
                config.pretrained_model = '_'.join(os.path.basename(config.pretrained_model).split('_')[:2])
            except:
                pass

    try:
        config.test_model = sorted(glob.glob(os.path.join(config.model_save_path, '*_D.pth')))[-1]
        config.test_model = '_'.join(os.path.basename(config.test_model).split('_')[:2])
    except:
        config.test_model = ''


    print(config)
    main(config)