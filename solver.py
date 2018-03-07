import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import sys
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms
from model import Generator
from model import Discriminator
from PIL import Image
import ipdb
import config as cfg
import glob
import pickle
from utils import f1_score, f1_score_max

class Solver(object):

    def __init__(self, MultiLabelAU_loader, au_loader, config):
        # Data loader
        self.MultiLabelAU_loader = MultiLabelAU_loader
        self.au_loader = au_loader

        # Model hyper-parameters
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.d_train_repeat = config.d_train_repeat

        # Hyper-parameteres
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.dataset = config.dataset
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.batch_size = config.batch_size
        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = config.pretrained_model

        self.FOCAL_LOSS = config.FOCAL_LOSS
        self.JUST_REAL = config.JUST_REAL
        self.FAKE_CLS = config.FAKE_CLS
        self.DENSENET = config.DENSENET        

        #Training Binary Classifier Settings
        self.au_model = config.au_model
        self.au = config.au
        self.multi_binary = config.multi_binary
        self.pretrained_model_generator = config.pretrained_model_generator

        # Test settings
        self.test_model = config.test_model
        self.metadata_path = config.metadata_path

        # Path
        self.log_path = config.log_path
        self.sample_path = config.sample_path
        self.model_save_path = config.model_save_path
        self.result_path = config.result_path
        self.fold = config.fold
        self.mode_data = config.mode_data

        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        # Define a generator and a discriminator
        if self.dataset == 'Both':
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)
        else:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 

        # Optimizers
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        # Print networks
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        # ipdb.set_trace()
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def threshold(self, x):
        x = x.clone()
        x = (x >= 0.5).float()
        return x

    def compute_accuracy(self, x, y, dataset):
        if dataset == 'CelebA' or dataset=='MultiLabelAU':
            x = F.sigmoid(x)
            predicted = self.threshold(x)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct, dim=0) * 100.0
        else:
            _, predicted = torch.max(x, dim=1)
            correct = (predicted == y).float()
            accuracy = torch.mean(correct) * 100.0
        return accuracy

    def one_hot(self, labels, dim):
        """Convert label indices to one-hot vector"""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        # ipdb.set_trace()
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def get_fixed_c_list(self):
        fixed_x = []
        real_c = []
        if self.image_size==512:
            n = 0
        else:
            n = 1
        for i, (images, labels, _) in enumerate(self.data_loader):
            fixed_x.append(images)
            real_c.append(labels)
            if i == n:
                break

        # Fixed inputs and target domain labels for debugging
        fixed_x = torch.cat(fixed_x, dim=0)
        fixed_x = self.to_var(fixed_x, volatile=True)
        real_c = torch.cat(real_c, dim=0)
        # ipdb.set_trace()
        if self.dataset == 'CelebA':
            fixed_c_list = self.make_celeb_labels(real_c)
        # elif self.dataset == 'MultiLabelAU':
        #     fixed_c_list = [self.to_var(torch.FloatTensor(np.random.randint(0,2,[self.batch_size*4,self.c_dim])), volatile=True)]*4
        elif self.dataset == 'RaFD' or self.dataset=='au01_fold0' or self.dataset == 'MultiLabelAU':
            fixed_c_list = []
            for i in range(self.c_dim):
                # ipdb.set_trace()
                fixed_c = self.one_hot(torch.ones(fixed_x.size(0)) * i, self.c_dim)
                fixed_c_list.append(self.to_var(fixed_c, volatile=True))
        return fixed_x, fixed_c_list  

    def show_img(self, img, real_label, fake_label):                  
        import matplotlib.pyplot as plt
        fake_image_list=[img]

        for fl in fake_label:
            # ipdb.set_trace()
            fake_image_list.append(self.G(img, self.to_var(fl.data, volatile=True)))
        fake_images = torch.cat(fake_image_list, dim=3)        
        shape0 = min(8, fake_images.data.cpu().shape[0])
        # ipdb.set_trace()
        save_image(self.denorm(fake_images.data.cpu()[:shape0,:,:,self.image_size:]), 'tmp_fake.jpg',nrow=1, padding=0)
        save_image(self.denorm(fake_images.data.cpu()[:shape0,:,:,:self.image_size]), 'tmp_real.jpg',nrow=1, padding=0)
        print("Real Label: \n"+str(real_label.data.cpu()[:shape0].numpy()))
        for fl in fake_label:
            print("Fake Label: \n"+str(fl.data.cpu()[:shape0].numpy()))        
        os.system('eog tmp_real.jpg')
        os.system('eog tmp_fake.jpg')
        os.remove('tmp_real.jpg')
        os.remove('tmp_fake.jpg')

    def train(self):
        """Train StarGAN within a single dataset."""

        # Set dataloader
        if self.dataset == 'MultiLabelAU':
            self.data_loader = self.MultiLabelAU_loader            
        elif self.dataset == 'au01_fold0':
            self.data_loader = self.au_loader      

        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        fixed_x = []
        real_c = []
        for i, (images, labels) in enumerate(self.data_loader):
            fixed_x.append(images)
            real_c.append(labels)
            if i == 1:
                break

        # Fixed inputs and target domain labels for debugging
        fixed_x = torch.cat(fixed_x, dim=0)
        fixed_x = self.to_var(fixed_x, volatile=True)
        real_c = torch.cat(real_c, dim=0)
        # ipdb.set_trace()
        if self.dataset == 'CelebA':
            fixed_c_list = self.make_celeb_labels(real_c)
        # elif self.dataset == 'MultiLabelAU':
        #     fixed_c_list = [self.to_var(torch.FloatTensor(np.random.randint(0,2,[self.batch_size*4,self.c_dim])), volatile=True)]*4
        elif self.dataset == 'RaFD' or self.dataset=='au01_fold0' or self.dataset == 'MultiLabelAU':
            fixed_c_list = []
            for i in range(self.c_dim):
                # ipdb.set_trace()
                fixed_c = self.one_hot(torch.ones(fixed_x.size(0)) * i, self.c_dim)
                fixed_c_list.append(self.to_var(fixed_c, volatile=True))


        # lr cache for decaying
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0

        last_model_step = len(self.data_loader)

        # Start training
        start_time = time.time()
        for e in range(start, self.num_epochs):
            E = str(e+1).zfill(2)
            for i, (real_x, real_label) in enumerate(self.data_loader):
                
                # Generat fake labels randomly (target domain labels)
                rand_idx = torch.randperm(real_label.size(0))
                fake_label = real_label[rand_idx]
                # ipdb.set_trace()
                if self.dataset == 'CelebA' or self.dataset=='MultiLabelAU':
                    real_c = real_label.clone()
                    fake_c = fake_label.clone()
                else:
                    real_c = self.one_hot(real_label, self.c_dim)
                    fake_c = self.one_hot(fake_label, self.c_dim)

                # Convert tensor to variable
                real_x = self.to_var(real_x)
                real_c = self.to_var(real_c)           # input for the generator
                fake_c = self.to_var(fake_c)
                real_label = self.to_var(real_label)   # this is same as real_c if dataset == 'CelebA'
                fake_label = self.to_var(fake_label)
                
                # ================== Train D ================== #

                # Compute loss with real images
                out_src, out_cls = self.D(real_x)#image -1,1
                d_loss_real = - torch.mean(out_src)
                # ipdb.set_trace()
                if self.dataset == 'CelebA' or self.dataset=='MultiLabelAU':
                    d_loss_cls = F.binary_cross_entropy_with_logits(
                        out_cls, real_label, size_average=False) / real_x.size(0)
                else:
                    d_loss_cls = F.cross_entropy(out_cls, real_label)

                # Compute classification accuracy of the discriminator
                if (i+1) % self.log_step == 0:
                    accuracies = self.compute_accuracy(out_cls, real_label, self.dataset)
                    log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                    if self.dataset == 'CelebA':
                        print('Classification Acc (Black/Blond/Brown/Gender/Aged): ')#, end='')
                    elif self.dataset=='MultiLabelAU':
                        print('Classification Acc (12 AUs): ')#, end='')
                    else:
                        print('Classification Acc (8 emotional expressions): ')#, end='')
                    print(log)

                # Compute loss with fake images
                fake_x = self.G(real_x, fake_c)
                
                # fake_list = []
                # fake_c=real_label.clone()*0
                # fake_list.append(fake_c.clone())
                # fake_c[:,0]=1
                # fake_list.append(fake_c.clone())
                # fake_c[:,6]=1
                # fake_list.append(fake_c.clone())
                # fake_c[:,-1]=1
                # fake_list.append(fake_c.clone())
                # fake_c[:]=1
                # fake_list.append(fake_c.clone())                
                # ipdb.set_trace()
                # self.show_img(real_x, real_c, fake_list)
                fake_x = Variable(fake_x.data)
                out_src, out_cls = self.D(fake_x)
                d_loss_fake = torch.mean(out_src)

                # Backward + Optimize
                
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Compute gradient penalty
                alpha = torch.rand(real_x.size(0), 1, 1, 1).cuda().expand_as(real_x)
                interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
                out, out_cls = self.D(interpolated)

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1)**2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging
                loss = {}
                loss['D/loss_real'] = d_loss_real.data[0]
                loss['D/loss_fake'] = d_loss_fake.data[0]
                loss['D/loss_cls'] = d_loss_cls.data[0]
                loss['D/loss_gp'] = d_loss_gp.data[0]

                # ================== Train G ================== #
                if (i+1) % self.d_train_repeat == 0:

                    # Original-to-target and target-to-original domain
                    fake_x = self.G(real_x, fake_c)
                    rec_x = self.G(fake_x, real_c)

                    # Compute losses
                    out_src, out_cls = self.D(fake_x)
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_rec = torch.mean(torch.abs(real_x - rec_x))

                    if self.dataset == 'CelebA' or self.dataset=='MultiLabelAU':
                        g_loss_cls = F.binary_cross_entropy_with_logits(
                            out_cls, fake_label, size_average=False) / fake_x.size(0)
                    else:
                        g_loss_cls = F.cross_entropy(out_cls, fake_label)

                    # Backward + Optimize
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging
                    loss['G/loss_fake'] = g_loss_fake.data[0]
                    loss['G/loss_rec'] = g_loss_rec.data[0]
                    loss['G/loss_cls'] = g_loss_cls.data[0]

                # Print out log info
                if (i+1) % self.log_step == 0 or (i+1)==last_model_step:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, E, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        print("Log path: "+self.log_path)
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)

                # Translate fixed images for debugging
                if (i+1) % self.sample_step == 0 or (i+1)==last_model_step:
                    fake_image_list = [fixed_x]
                    # ipdb.set_trace()
                    for fixed_c in fixed_c_list:
                        fake_image_list.append(self.G(fixed_x, fixed_c))
                    fake_images = torch.cat(fake_image_list, dim=3)
                    # ipdb.set_trace()
                    shape0 = min(64, fake_images.data.cpu().shape[0])
                    save_image(self.denorm(fake_images.data.cpu()[:shape0]),
                        os.path.join(self.sample_path, '{}_{}_fake.png'.format(E, i+1)),nrow=1, padding=0)
                    print('Translated images and saved into {}..!'.format(self.sample_path))

                # Save model checkpoints
                if (i+1) % self.model_save_step == 0 or (i+1)==last_model_step:
                    torch.save(self.G.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_G.pth'.format(E, i+1)))
                    torch.save(self.D.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_D.pth'.format(E, i+1)))

            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                d_lr -= (self.d_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decay learning rate to g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))
        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()
        # ipdb.set_trace()
        if self.dataset == 'MultiLabelAU':
            data_loader = self.MultiLabelAU_loader            
        elif dataset == 'au01_fold0':
            data_loader = self.au_loader            

        for i, (real_x, org_c) in enumerate(data_loader):
            real_x = self.to_var(real_x, volatile=True)
            if self.dataset == 'CelebA':
                target_c_list = self.make_celeb_labels(org_c)
            else:
                target_c_list = []
                for j in range(self.c_dim):
                    target_c = self.one_hot(torch.ones(real_x.size(0)) * j, self.c_dim)
                    target_c_list.append(self.to_var(target_c, volatile=True))

            # Start translations
            fake_image_list = [real_x]

            for target_c in target_c_list:
                fake_x = self.G(real_x, target_c)
                #out_src_temp, out_cls_temp = self.D(fake_x)
                #F.sigmoid(out_cls_temp)
                #accuracies = self.compute_accuracy(out_cls_temp, target_c, self.dataset)
                fake_image_list.append(fake_x)
            fake_images = torch.cat(fake_image_list, dim=3)
            save_path = os.path.join(self.result_path, '{}_real.png'.format(i+1))
            # ipdb.set_trace()
            # save_image(self.denorm(fake_images.data), save_path, nrow=1, padding=0)
            print('Translated test images and saved into "{}"..!'.format(save_path))

    def test_cls(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        from data_loader import get_loader
        if self.test_model=='':
            last_file = sorted(glob.glob(os.path.join(self.model_save_path,  '*_D.pth')))[-1]
            last_name = '_'.join(last_file.split('/')[-1].split('_')[:2])
        else:
            last_name = self.test_model

        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(last_name))
        D_path = os.path.join(self.model_save_path, '{}_D.pth'.format(last_name))
        txt_path = os.path.join(self.model_save_path, '{}_{}.txt'.format(last_name,'{}'))
        self.pkl_data = os.path.join(self.model_save_path, '{}_{}.pkl'.format(last_name, '{}'))
        self.lstm_path = os.path.join(self.model_save_path, '{}_lstm'.format(last_name))
        if not os.path.isdir(self.lstm_path): os.makedirs(self.lstm_path)
        print(" [!!] {} model loaded...".format(D_path))
        self.G.load_state_dict(torch.load(G_path))
        self.D.load_state_dict(torch.load(D_path))
        self.G.eval()
        self.D.eval()
        # ipdb.set_trace()
        if self.dataset == 'MultiLabelAU':
            data_loader_train = get_loader(self.metadata_path, self.image_size,
                                   self.image_size, self.batch_size, 'MultiLabelAU', 'train', no_flipping = True)
            data_loader_test = get_loader(self.metadata_path, self.image_size,
                                   self.image_size, self.batch_size, 'MultiLabelAU', 'test')
        elif dataset == 'au01_fold0':
            data_loader = self.au_loader    


        if not hasattr(self, 'output_txt'):
            # ipdb.set_trace()
            self.output_txt = txt_path
            try:
                self.output_txt  = sorted(glob.glob(self.output_txt.format('*')))[-1]
                number_file = len(glob.glob(self.output_txt))
            except:
                number_file = 0
            self.output_txt = self.output_txt.format(str(number_file).zfill(2)) 
        
        self.f=open(self.output_txt, 'a')     
        self.thresh = np.linspace(0.01,0.99,200).astype(np.float32)
        # ipdb.set_trace()
        F1_real, F1_max, max_thresh_train  = self.F1_TEST(data_loader_train, mode = 'TRAIN')
        _ = self.F1_TEST(data_loader_test, thresh = max_thresh_train)
     
        self.f.close()

    def F1_TEST(self, data_loader, mode = 'TEST', thresh = [0.5]*len(cfg.AUs)):

        PREDICTION = []
        GROUNDTRUTH = []
        total_idx=int(len(data_loader)/self.batch_size)  
        count = 0
        for i, (real_x, org_c, files) in enumerate(data_loader):
            if os.path.isfile(self.pkl_data.format(mode.lower())): 
                PREDICTION, GROUNDTRUTH = pickle.load(open(self.pkl_data.format(mode.lower())))
                break
            # ipdb.set_trace()
            real_x = self.to_var(real_x, volatile=True)
            labels = org_c
            
            
            # ipdb.set_trace()
            _, out_cls_temp, lstm_input = self.D(real_x, lstm=True)
            self.save_lstm(lstm_input.data.cpu().numpy(), files)
            # output = ((F.sigmoid(out_cls_temp)>=0.5)*1.).data.cpu().numpy()
            output = F.sigmoid(out_cls_temp)
            if i==0:
                print(mode.upper())
                print("Predicted:   "+str((output>=0.5)*1))
                print("Groundtruth: "+str(org_c))

            count += org_c.shape[0]
            string_ = str(count)+' / '+str(len(data_loader)*self.batch_size)
            sys.stdout.write("\r%s" % string_)
            sys.stdout.flush()        
            # ipdb.set_trace()

            PREDICTION.append(output.data.cpu().numpy().tolist())
            GROUNDTRUTH.append(labels.cpu().numpy().astype(np.uint8).tolist())

        if not os.path.isfile(self.pkl_data.format(mode.lower())): 
            pickle.dump([PREDICTION, GROUNDTRUTH], open(self.pkl_data.format(mode.lower()), 'w'))
        print("")
        print >>self.f, ""
        # print("[Min and Max predicted: "+str(min(prediction))+ " " + str(max(prediction))+"]")
        # print >>self.f, "[Min and Max predicted: "+str(min(prediction))+ " " + str(max(prediction))+"]"
        print("")

        PREDICTION = np.vstack(PREDICTION)
        GROUNDTRUTH = np.vstack(GROUNDTRUTH)

        F1_real5 = [0]*len(cfg.AUs); F1_Thresh5 = [0]*len(cfg.AUs); F1_real = [0]*len(cfg.AUs)
        F1_Thresh = [0]*len(cfg.AUs); F1_0 = [0]*len(cfg.AUs); F1_1 = [0]*len(cfg.AUs)
        F1_Thresh_0 = [0]*len(cfg.AUs); F1_Thresh_1 = [0]*len(cfg.AUs); F1_MAX = [0]*len(cfg.AUs)
        F1_Thresh_max = [0]*len(cfg.AUs); F1_median5 = [0]*len(cfg.AUs); F1_median7 = [0]*len(cfg.AUs)
        F1_median3 = [0]*len(cfg.AUs)
        # ipdb.set_trace()
        for i in xrange(len(cfg.AUs)):
            prediction = PREDICTION[:,i]
            groundtruth = GROUNDTRUTH[:,i]
            if mode=='TEST':
                _, F1_real5[i], F1_Thresh5[i], F1_median3[i], F1_median5[i], F1_median7[i] = f1_score(groundtruth, prediction, 0.5, median=True)     
            _, F1_real[i], F1_Thresh[i] = f1_score(np.array(groundtruth), np.array(prediction), thresh[i])
            _, F1_0[i], F1_Thresh_0[i] = f1_score(np.array(groundtruth), np.array(prediction)*0, thresh[i])
            _, F1_1[i], F1_Thresh_1[i] = f1_score(np.array(groundtruth), (np.array(prediction)*0)+1, thresh[i])
            _, F1_MAX[i], F1_Thresh_max[i] = f1_score_max(np.array(groundtruth), np.array(prediction), self.thresh)     


        if mode=='TEST':
            for i, au in enumerate(cfg.AUs):
                string = "---> [%s] AU%s F1: %.4f, Threshold: %.4f <---" % (mode, str(au).zfill(2), F1_real5[i], F1_Thresh5[i])
                print(string)
                print >>self.f, string
            string = "F1 Mean: %.4f"%np.mean(F1_real5)
            print(string)
            print("")
            print >>self.f, string
            print >>self.f, ""

            for i, au in enumerate(cfg.AUs):
                string = "---> [%s] AU%s F1_median3: %.4f, Threshold: %.4f <---" % (mode, str(au).zfill(2), F1_median3[i], F1_Thresh5[i])
                print(string)
                print >>self.f, string
            string = "F1_median3 Mean: %.4f"%np.mean(F1_median3)
            print(string)
            print("")
            print >>self.f, string
            print >>self.f, ""

            for i, au in enumerate(cfg.AUs):
                string = "---> [%s] AU%s F1_median5: %.4f, Threshold: %.4f <---" % (mode, str(au).zfill(2), F1_median5[i], F1_Thresh5[i])
                print(string)
                print >>self.f, string
            string = "F1_median5 Mean: %.4f"%np.mean(F1_median5)
            print(string)
            print("")
            print >>self.f, string
            print >>self.f, ""

            for i, au in enumerate(cfg.AUs):
                string = "---> [%s] AU%s F1_median7: %.4f, Threshold: %.4f <---" % (mode, str(au).zfill(2), F1_median7[i], F1_Thresh5[i])
                print(string)
                print >>self.f, string
            string = "F1_median7 Mean: %.4f"%np.mean(F1_median7)
            print(string)
            print("")
            print >>self.f, string
            print >>self.f, ""            

        for i, au in enumerate(cfg.AUs):
            string = "---> [%s] AU%s F1: %.4f, Threshold: %.4f <---" % (mode, str(au).zfill(2), F1_real[i], F1_Thresh_0[i])
            print(string)
            print >>self.f, string
        string = "F1 Mean: %.4f"%np.mean(F1_real)
        print(string)
        print("")
        print >>self.f, string
        print >>self.f, ""

        for i, au in enumerate(cfg.AUs):
            string = "---> [%s - 0] AU%s F1: %.4f, Threshold: %.4f <---" % (mode, str(au).zfill(2), F1_0[i], F1_Thresh[i])
            print(string)
            print >>self.f, string
        string = "F1 Mean: %.4f"%np.mean(F1_0)
        print(string)
        print("")
        print >>self.f, string
        print >>self.f, ""

        for i, au in enumerate(cfg.AUs):
            string = "---> [%s - 1] AU%s F1: %.4f, Threshold: %.4f <---" % (mode, str(au).zfill(2), F1_1[i], F1_Thresh_1[i])
            print(string)
            print >>self.f, string
        string = "F1 Mean: %.4f"%np.mean(F1_1)
        print(string)
        print("")
        print >>self.f, string
        print >>self.f, ""

        for i, au in enumerate(cfg.AUs):
            #REAL F1_MAX
            string = "---> [%s] AU%s F1_MAX: %.4f, Threshold: %.4f <---" % (mode, str(au).zfill(2), F1_MAX[i], F1_Thresh_max[i])
            print(string)
            print >>self.f, string
        string = "F1 Mean: %.4f"%np.mean(F1_MAX)
        print(string)
        print("")
        print >>self.f, string
        print >>self.f, ""

        return F1_real, F1_MAX, F1_Thresh_max     

    def save_lstm(self, data, files):
        assert data.shape[0]==len(files)
        for i in range(len(files)):
            name = os.path.join(self.lstm_path, '/'.join(files[i].split('/')[-6:]))
            name = name.replace('jpg', 'npy')
            folder = os.path.dirname(name)
            if not os.path.isdir(folder): os.makedirs(folder)
            np.save(name, data[i])
