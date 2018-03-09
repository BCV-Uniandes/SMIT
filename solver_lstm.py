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
import random

class LSTM(nn.Module):
    """Discriminator. PatchGAN."""
    # def __init__(self, batch_size = 8, input_size=4096*2*2, hidden_size=256, num_layers=1, c_dim=12):
    def __init__(self, batch_size = 8, input_size=2048*4*4, hidden_size=256, num_layers=1, c_dim=12):
        super(LSTM, self).__init__()
        self.batch_size = batch_size
        self.fc0 = nn.Linear(input_size, 2048)
        self.lstm = nn.LSTM(2048, hidden_size, num_layers)
        # ipdb.set_trace()
        # self.conv = nn.Conv2d(hidden_size, c_dim, kernel_size=2, stride=1, bias=False)
        self.fc1 = nn.Linear(hidden_size, c_dim)

    def forward(self, x):
        # ipdb.set_trace()
        data = x.view(self.batch_size, -1)
        hh = self.fc0(data)
        hh = hh.view(self.batch_size, 1, -1)
        hh = self.lstm(hh)
        hh = torch.squeeze(hh[0],1)
        hh = self.fc(hh)

        return hh

class LSTM2(nn.Module):
    """Discriminator. PatchGAN."""
    # def __init__(self, input_size=4096*2*2, hidden_size=256, num_layers=1, c_dim=12):
    def __init__(self, input_size=4096*2*2, hidden_size=256, num_layers=1, c_dim=12):
        super(LSTM2, self).__init__()
        self.fc0 = nn.Linear(input_size, 2048)
        self.dropout = nn.Dropout()
        self.lstm = nn.LSTM(2048, hidden_size, num_layers)
        self.fc1 = nn.Linear(hidden_size, c_dim)

    def forward(self, x):
        # ipdb.set_trace()
        hh = x.transpose(0,1)
        hh = self.dropout(self.fc0(hh))
        # ipdb.set_trace()
        hh = self.lstm(hh)[0][-1]
        hh = self.fc1(hh)

        return hh        

class Solver(object):

    def __init__(self, MultiLabelAU_loader, config):
        # Data loader
        self.MultiLabelAU_loader = MultiLabelAU_loader

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
        self.c_lr = config.d_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.dataset = config.dataset
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.batch_size = config.batch_size
        self.batch_seq = config.batch_seq
        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = config.pretrained_model
        self.FOCAL_LOSS = config.FOCAL_LOSS

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
        #self.C = LSTM(self.batch_size)
        self.C = LSTM2()
        # ipdb.set_trace()
        # Optimizers
        # self.c_optimizer = torch.optim.Adam(self.C.parameters(), self.c_lr, [self.beta1, self.beta2])
        self.c_optimizer = torch.optim.SGD(self.C.parameters(), self.c_lr, momentum=0.9, nesterov=True)

        # Print networks
        self.print_network(self.C, 'C')

        if torch.cuda.is_available():
            self.C.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        self.C.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_LSTM.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, c_lr):
        for param_group in self.c_optimizer.param_groups:
            param_group['lr'] = c_lr

    def reset_grad(self):
        self.c_optimizer.zero_grad()

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

    def focal_loss(self, out, label):
        alpha=2
        gamma=1
        max_val = (-out).clamp(min=0)
        pt = out - out * label + max_val + ((-max_val).exp() + (-out - max_val).exp()).log()

        FL = alpha*torch.pow(1-(-pt).exp(),gamma)*pt
        FL = FL.sum()
        return FL    

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
        for i, (images, labels) in enumerate(self.data_loader):
            fixed_x.append(images)
            real_c.append(labels)
            if i == 3:
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

    def get_id_filename(self, filename):
        return '_'.join(filename.split('/')[-3:-1])

    def get_num_filename(self, filename):
        return filename.split('/')[-1].split('.')[0]

    def get_dict_filenames(self, all_filenames):
        dict_names = {}
        for line in all_filenames:
            subj_id=self.get_id_filename(line)
            frame_num=int(self.get_num_filename(line))
            if subj_id not in dict_names:
                dict_names[subj_id]=frame_num
            else:
                if frame_num<dict_names[subj_id]:
                    dict_names[subj_id]=frame_num
        return dict_names        

    def get_ind_filename(self, filename, dict_names, num_files=24):
        subj_id=self.get_id_filename(filename)
        frame_last  = self.get_num_filename(filename)        
        frame_first = dict_names[subj_id]
        count = 1
        files_seq = [filename]
        range_idx = range(int(frame_last)-1, frame_first-1, -1)
        random.seed(1234)
        random.shuffle(range_idx)
        # ipdb.set_trace()
        for i in range_idx:
            temp0 = filename.replace(frame_last, str(i).zfill(2))
            temp1 = filename.replace(frame_last, str(i).zfill(3))
            temp2 = filename.replace(frame_last, str(i).zfill(4))
            if os.path.isfile(temp0): temp=temp0
            elif os.path.isfile(temp1): temp=temp1
            elif os.path.isfile(temp2): temp=temp2
            else: continue
            npy_file = os.path.join(self.lstm_path, '/'.join(temp.split('/')[-6:])).replace('jpg', 'npy')
            if not os.path.isfile(npy_file): continue
            files_seq.append(temp)
            count+=1
            if count==num_files: break

        files_seq = files_seq[::-1]

        while count<num_files:
            files_seq.append(files_seq[-1])
            count+=1
        return files_seq
            

    def train(self):
        """Train StarGAN within a single dataset."""

        # Set dataloader
        self.data_loader, self.all_filenames = self.MultiLabelAU_loader            

        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        self.lstm_path = os.path.join(self.model_save_path, '{}_lstm'.format(self.test_model))

        try:
            last_file = sorted(glob.glob(os.path.join(self.lstm_path, '*_LSTM.pth')))[-1]
            last_name = '_'.join(last_file.split('/')[-1].split('_')[:2])

            C_path = os.path.join(self.lstm_path,'{}_LSTM.pth'.format(last_name))
            print("[*] Loading from: "+C_path)
            self.C.load_state_dict(torch.load(C_path))
            start = int(last_name.split('_')[0])
        except:
            start = 0

        # lr cache for decaying
        c_lr = self.c_lr
        

        for _ in range(start):
            c_lr -= (self.c_lr / 20.)#float(self.num_epochs_decay))
            self.update_lr(c_lr)
            print ('Decay learning rate to c_lr: {}.'.format(c_lr))

        #fake CLS loss
        d_loss_cls_fake = self.to_var(torch.zeros((1)))
        iter_fake = 2
        epoch_stop_generator = 9
        epoch_start_fake_cls = 7

        last_model_step = len(self.data_loader)

        # Start training
        start_time = time.time()
        print(" [!] Starting Epoch {}".format(start+1))
        #50 batch size, 24 sequences
        dict_names = self.get_dict_filenames(self.all_filenames)

        for e in range(start, self.num_epochs):
            E = str(e+1).zfill(2)
            for i, (real_x, real_label_, files) in enumerate(self.data_loader):
                # ipdb.set_trace()
                data_lstm = []#np.zeros((real_x.size(0), self.batch_seq, 4096*4), dtype=np.float32)
                all_files_lstm = []
                for bs in range(real_x.size(0)):
                    # ipdb.set_trace()
                    imgs_lstm = self.get_ind_filename(files[bs], dict_names, num_files=self.batch_seq)

                    files_lstm = [os.path.join(self.lstm_path, '/'.join(imgs_lstm[j].split('/')[-6:])) \
                                        for j in range(len(imgs_lstm))]
                    all_files_lstm.append(imgs_lstm)

                    data_lstm_ = [np.load(j.replace('jpg', 'npy')).reshape(1,-1) for j in files_lstm]
                    try:data_lstm_ = np.expand_dims(np.concatenate(data_lstm_, axis=0), axis=0)
                    except:ipdb.set_trace()
                    data_lstm.append(data_lstm_)
                data_lstm = np.concatenate(data_lstm, axis=0)
                # ipdb.set_trace()
                data_lstm = self.to_var(torch.from_numpy(data_lstm))

                # if data_lstm.size()[0]<self.batch_size:continue

                # ipdb.set_trace()
                real_c = real_label_.clone()

                # Convert tensor to variable
                real_x = self.to_var(real_x)
                real_c = self.to_var(real_c)   
                real_label = self.to_var(real_label_)  
                
                # ================== Train D ================== #
                # ipdb.set_trace()
                # Compute loss with real images
                out_cls = self.C(data_lstm)

                if self.FOCAL_LOSS:
                    c_loss_cls = self.focal_loss(
                        out_cls, real_label) / real_x.size(0)
                else:
                    # c_loss_cls = F.binary_cross_entropy_with_logits(
                    #     out_cls, real_label, size_average=False) / real_x.size(0)
                    c_loss_cls = F.binary_cross_entropy_with_logits(
                        out_cls, real_label, size_average=True)


                # Compute classification accuracy of the discriminator
                if (i+1) % self.log_step == 0:
                    accuracies = self.compute_accuracy(out_cls, real_label, self.dataset)
                    log = ["{:.2f}".format(acc) for acc in accuracies.data.cpu().numpy()]
                    print('Classification Acc (12 AUs): ')#, end='')
                    print(log)

                c_loss = c_loss_cls
                self.reset_grad()
                c_loss.backward()
                self.c_optimizer.step()

                # Logging
                loss = {}
                loss['loss_cls_real'] = c_loss_cls.data[0]



                # Print out log info
                if (i+1) % self.log_step == 0 or (i+1)==last_model_step:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}] [fold{}] [{}] [LSTM]".format(
                        elapsed, E, self.num_epochs, i+1, iters_per_epoch, self.fold, self.image_size)   

                    for tag, value in sorted(loss.items(), key= lambda x:x[0]):
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        print("Log path: "+self.log_path)
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)


                # Save model checkpoints
                if (i+1) % self.model_save_step == 0 or (i+1)==last_model_step:
                    print("Saving model at: "+os.path.join(self.lstm_path, '{}_{}_LSTM.pth'.format(E, i+1)))
                    torch.save(self.C.state_dict(),
                        os.path.join(self.lstm_path, '{}_{}_LSTM.pth'.format(E, i+1)))

          # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                c_lr -= (self.c_lr / float(self.num_epochs_decay))
                self.update_lr(c_lr)
                print ('Decay learning rate to c_lr: {}.'.format(c_lr))

    def test_cls(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        from data_loader import get_loader
       
        self.lstm_path = os.path.join(self.model_save_path, '{}_lstm'.format(self.test_model))

        last_file = sorted(glob.glob(os.path.join(self.lstm_path, '*_LSTM.pth')))[-1]
        last_name = '_'.join(last_file.split('/')[-1].split('_')[:2])

        C_path = os.path.join(self.lstm_path,'{}_LSTM.pth'.format(last_name))
        txt_path = os.path.join(self.lstm_path,'{}_{}_LSTM.txt'.format(last_name,'{}'))
        self.pkl_data = os.path.join(self.lstm_path,'{}_{}_LSTM.pkl'.format(last_name, '{}'))
        print("[*] Loading from: "+C_path)
        self.C.load_state_dict(torch.load(C_path))
        self.C.eval()
        # ipdb.set_trace()
        if self.dataset == 'MultiLabelAU':
            data_loader_train, all_filenames_train = get_loader(self.metadata_path, self.image_size,
                                   self.image_size, self.batch_size, 'MultiLabelAU', 'train', no_flipping = True, LSTM=True)
            data_loader_test, all_filenames_test = get_loader(self.metadata_path, self.image_size,
                                   self.image_size, self.batch_size, 'MultiLabelAU', 'test', LSTM=True)
        elif dataset == 'au01_fold0':
            data_loader = self.au_loader    

        self.dict_names = {}
        self.dict_names['TRAIN'] = self.get_dict_filenames(all_filenames_train)
        self.dict_names['TEST'] = self.get_dict_filenames(all_filenames_test)

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
        # F1_real, F1_max, max_thresh_train  = self.F1_TEST(data_loader_train, mode = 'TRAIN')
        # _ = self.F1_TEST(data_loader_test, thresh = max_thresh_train)
        _ = self.F1_TEST(data_loader_test, thresh = [0.5]*12)
     
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

            data_lstm = []#np.zeros((real_x.size(0), self.batch_seq, 4096*4), dtype=np.float32)
            all_files_lstm = []
            for bs in range(real_x.size(0)):
                # ipdb.set_trace()
                imgs_lstm = self.get_ind_filename(files[bs], self.dict_names[mode], num_files=self.batch_seq)

                files_lstm = [os.path.join(self.lstm_path, '/'.join(imgs_lstm[j].split('/')[-6:])) \
                                    for j in range(len(imgs_lstm))]
                all_files_lstm.append(imgs_lstm)

                data_lstm_ = [np.load(j.replace('jpg', 'npy')).reshape(1,-1) for j in files_lstm]
                # ipdb.set_trace()
                try:data_lstm_ = np.expand_dims(np.concatenate(data_lstm_, axis=0), axis=0)
                except:ipdb.set_trace()
                data_lstm.append(data_lstm_)
            data_lstm = np.concatenate(data_lstm, axis=0)
            data_lstm = self.to_var(torch.from_numpy(data_lstm))

            # real_x = self.to_var(real_x, volatile=True)
            labels = org_c
            
            
            # ipdb.set_trace()
            out_cls_temp = self.C(data_lstm)
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
