import torch
import os
import glob
import warnings
import torch.nn.functional as F
import numpy as np
import time
import datetime
from torchvision.utils import save_image
from misc.utils import color_frame
from misc.utils import create_dir, denorm, get_labels, get_torch_version
from misc.utils import Modality, PRINT, single_source, target_debug_list
from misc.utils import to_cuda, to_data, to_var
import torch.utils.data.distributed
warnings.filterwarnings('ignore')


class Solver(object):
    def __init__(self, config, data_loader=None):
        # Data loader
        self.data_loader = data_loader
        self.config = config

        self.build_model()

        # Start with trained model
        if self.config.pretrained_model:
            self.load_pretrained_model()

    # ==================================================================#
    # ==================================================================#
    def build_model(self):
        # Define a generator and a discriminator
        from model import MultiDiscriminator as Discriminator
        from model import AdaInGEN as Generator

        self.D = Discriminator(self.config, debug=self.config.mode == 'train')
        to_cuda(self.D)
        D_parameters = filter(lambda p: p.requires_grad, self.D.parameters())
        self.d_optimizer = torch.optim.Adam(
            D_parameters, self.config.d_lr,
            [self.config.beta1, self.config.beta2])

        self.G = Generator(self.config, debug=self.config.mode == 'train')
        G_parameters = self.G.generator.parameters()
        G_parameters = filter(lambda p: p.requires_grad, G_parameters)
        self.g_optimizer = torch.optim.Adam(
            G_parameters, self.config.g_lr,
            [self.config.beta1, self.config.beta2])
        to_cuda(self.G)

        if self.config.mode == 'train':
            self.print_network(self.D, 'Discriminator')
            self.print_network(self.G, 'Generator')

    # ==================================================================#
    # ==================================================================#

    def print_network(self, model, name):
        if name == 'Generator':
            choices = ['generator', 'adain_net']
            for m in choices:
                submodel = getattr(model, m)
                num_params = 0
                for p in submodel.parameters():
                    num_params += p.numel()
                self.PRINT("{} number of parameters: {}".format(
                    m.upper(), num_params))
        else:
            num_params = 0
            for p in model.parameters():
                num_params += p.numel()
            self.PRINT("{} number of parameters: {}".format(
                name.upper(), num_params))
        # self.PRINT(name)
        # self.PRINT(model)
        # self.PRINT("{} number of parameters: {}".format(name, num_params))
        # self.display_net(name)

    # ============================================================#
    # ============================================================#
    def output_sample(self, epoch, iter):
        return os.path.join(
            self.config.sample_path, '{}_{}_fake.jpg'.format(
                str(epoch).zfill(4),
                str(iter).zfill(len(str(len(self.data_loader))))))

    # ============================================================#
    # ============================================================#
    def output_model(self, epoch, iter):
        return os.path.join(
            self.config.model_save_path, '{}_{}_{}.pth'.format(
                str(epoch).zfill(4),
                str(iter).zfill(len(str(len(self.data_loader)))), '{}'))

    # ==================================================================#
    # ==================================================================#
    def save(self, Epoch, iter):
        name = self.output_model(Epoch, iter)
        torch.save(self.G.state_dict(), name.format('G'))
        torch.save(self.g_optimizer.state_dict(), name.format('G_optim'))
        torch.save(self.D.state_dict(), name.format('D'))
        torch.save(self.d_optimizer.state_dict(), name.format('D_optim'))

        if self.config.model_epoch != 1 and int(
                Epoch) % self.config.model_epoch == 0:
            for _epoch in range(
                    int(Epoch) - self.config.model_epoch + 1, int(Epoch)):
                name_1 = os.path.join(
                    self.config.model_save_path, '{}_{}_{}.pth'.format(
                        str(_epoch).zfill(4), iter, '{}'))
                if os.path.isfile(name_1.format('G')):
                    os.remove(name_1.format('G'))
                if os.path.isfile(name_1.format('G_optim')):
                    os.remove(name_1.format('G_optim'))
                if os.path.isfile(name_1.format('D')):
                    os.remove(name_1.format('D'))
                if os.path.isfile(name_1.format('D_optim')):
                    os.remove(name_1.format('D_optim'))

    # ==================================================================#
    # ==================================================================#
    def load_pretrained_model(self):
        self.PRINT('Resuming model (step: {})...'.format(
            self.config.pretrained_model))
        self.name = os.path.join(
            self.config.model_save_path, '{}_{}.pth'.format(
                self.config.pretrained_model, '{}'))
        self.PRINT('Model: {}'.format(self.name))
        # if self.config.mode=='train': self.PRINT('Model: {}'.format(name))
        # self.G.load_state_dict(torch.load(self.name.format('G')))
        # self.D.load_state_dict(torch.load(self.name.format('D')))
        self.G.load_state_dict(
            torch.load(
                self.name.format('G'),
                map_location=lambda storage, loc: storage))
        self.D.load_state_dict(
            torch.load(
                self.name.format('D'),
                map_location=lambda storage, loc: storage))

        if self.config.mode == 'train':
            try:
                # self.g_optimizer.load_state_dict(torch.load(self.name.format('G_optim')))
                self.g_optimizer.load_state_dict(
                    torch.load(
                        self.name.format('G_optim'),
                        map_location=lambda storage, loc: storage))
                self.optim_cuda(self.g_optimizer)
                # self.d_optimizer.load_state_dict(torch.load(self.name.format('D_optim')))
                self.d_optimizer.load_state_dict(
                    torch.load(
                        self.name.format('D_optim'),
                        map_location=lambda storage, loc: storage))
                self.optim_cuda(self.d_optimizer)
                print("Success!!")
            except BaseException:
                print("Loading Failed!!")
        else:
            print("Success!!")

    # ==================================================================#
    # ==================================================================#
    def optim_cuda(self, optimizer):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = to_cuda(v)  # .cuda()

    # ==================================================================#
    # ==================================================================#
    def resume_name(self):
        if self.config.pretrained_model in ['', None]:
            try:
                last_file = sorted(
                    glob.glob(
                        os.path.join(self.config.model_save_path,
                                     '*_G.pth')))[-1]
            except IndexError:
                raise IndexError("No model found at " +
                                 self.config.model_save_path)
            last_name = '_'.join(os.path.basename(last_file).split('_')[:2])
        else:
            last_name = self.config.pretrained_model
        return last_name

    # ==================================================================#
    # ==================================================================#
    def get_labels(self):
        return get_labels(
            self.config.image_size,
            self.config.dataset_fake,
            attr=self.data_loader.dataset)

    # ==================================================================#
    # ==================================================================#
    def PRINT_LOG(self, batch_size):
        from termcolor import colored
        Log = "---> batch size: {}, img: {}, GPU: {}, !{} |".format(
            batch_size, self.config.image_size, self.config.GPU,
            self.config.mode_data)
        if self.config.ALL_ATTR != 0:
            Log += ' [*ALL_ATTR={}]'.format(self.config.ALL_ATTR)
        if self.config.MultiDis:
            Log += ' [*MultiDisc={}]'.format(self.config.MultiDis)
        if self.config.Identity:
            Log += ' [*Identity]'
        if self.config.Slim_Generator:
            Log += ' [*SlimGen]'
        if self.config.Deterministic:
            Log += ' [*Deterministic]'
        dataset_string = colored(self.config.dataset_fake, 'red')
        Log += ' [*{}]'.format(dataset_string)
        self.PRINT(Log)
        return Log

    # ==================================================================#
    # ==================================================================#
    def PRINT(self, str):
        if self.config.mode == 'train':
            PRINT(self.config.log, str)
        else:
            print(str)

    # ==================================================================#
    # ==================================================================#
    def PLOT(self, Epoch):
        from misc.utils import plot_txt
        LOSS = {
            key: np.array(value).mean()
            for key, value in self.LOSS.items()
        }
        if not os.path.isfile(self.config.loss_plot):
            with open(self.config.loss_plot, 'w') as f:
                f.writelines('{}\n'.format(
                    '\t'.join(['Epoch'] + list(LOSS.keys()))))
        with open(self.config.loss_plot, 'a') as f:
            f.writelines('{}\n'.format(
                '\t'.join([str(Epoch)] + [str(i)
                                          for i in list(LOSS.values())])))
        plot_txt(self.config.loss_plot)

    # ==================================================================#
    # ==================================================================#
    def _CLS(self, data):
        data = to_var(data, volatile=True)
        out_label = self.D(data)[1]
        if len(out_label) > 1:
            out_label = torch.cat(
                [F.sigmoid(out.unsqueeze(-1)) for out in out_label],
                dim=-1).mean(dim=-1)
        else:
            out_label = F.sigmoid(out_label[0])
        out_label = (out_label > 0.5).float()
        return out_label

    # ==================================================================#
    # ==================================================================#
    def _SAVE_IMAGE(self, save_path, fake_list, Attention=False, mode='fake'):
        fake_images = to_data(torch.cat(fake_list, dim=3), cpu=True)
        if 'fake' not in os.path.basename(save_path):
            save_path = save_path.replace('.jpg', '_fake.jpg')
        if Attention:
            mode = mode + '_attn'
        else:
            fake_images = denorm(fake_images)

        save_path = save_path.replace('fake', mode)
        fake_images = torch.cat((self.get_labels(), fake_images), dim=0)
        save_image(fake_images, save_path, nrow=1, padding=0)
        return save_path

    # ==================================================================#
    # ==================================================================#
    def target_multiAttr(self, target, index):
        # if self.config.dataset_fake == 'CelebA' and \
        #         self.config.c_dim == 10 and \
        #         k >= 3 and k <= 6:
        #     target_c[:, 2:5] = 0
        #     target_c[:, k - 1] = 1
        if self.config.dataset_fake == 'CelebA':
            all_attr = self.data_loader.dataset.selected_attrs
            attr2idx = self.data_loader.dataset.attr2idx
            color_hair = [
                'Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'
            ]
            style_hair = ['Bald', 'Straight_Hair', 'Wavy_Hair']
            ammount_hair = ['Bald', 'Bangs']
            if all_attr[index] in color_hair:
                color_hair.remove(all_attr[index])
                for attr in color_hair:
                    if attr in all_attr:
                        target[:, attr2idx[attr]] = 0
                target[:, index] = 1
            if all_attr[index] in style_hair:
                style_hair.remove(all_attr[index])
                for attr in style_hair:
                    if attr in all_attr:
                        target[:, attr2idx[attr]] = 0
                target[:, index] = 1
            if all_attr[index] in ammount_hair:
                ammount_hair.remove(all_attr[index])
                for attr in ammount_hair:
                    if attr in all_attr:
                        target[:, attr2idx[attr]] = 0
        return target

    # ==================================================================#
    # ==================================================================#
    def Create_Visual_List(self, batch):
        fake_image_list = [single_source(batch)]
        fake_attn_list = [
            single_source(to_var(denorm(batch.data), volatile=True))
        ]
        fake_image_list[0] = color_frame(
            fake_image_list[0], thick=5, color='green', first=True)
        fake_attn_list[0] = color_frame(
            fake_attn_list[0], thick=5, color='green', first=True)
        return fake_image_list, fake_attn_list

    # ==================================================================#
    # ==================================================================#
    @property
    def MultiLabel_Datasets(self):
        return ['BP4D', 'CelebA', 'EmotionNet', 'DEMO']

    # ==================================================================#
    # ==================================================================#

    def get_batch_inference(self, batch, Multimodal):
        if Multimodal:
            batch = [
                img.unsqueeze(0).repeat(self.config.style_debug, 1, 1, 1)
                for img in batch
            ]
        else:
            batch = [batch]
        return batch

    # ==================================================================#
    # ==================================================================#

    def generate_SMIT(self,
                      batch,
                      save_path,
                      Multimodal=0,
                      label=None,
                      output=False,
                      training=False,
                      fixed_style=None,
                      TIME=False):
        self.G.eval()
        self.D.eval()
        modal = 'Multimodal' if Multimodal else 'Unimodal'
        Output = []
        opt = torch.no_grad() if get_torch_version() > 0.3 else open(
            '/tmp/null.txt', 'w')
        flag_time = True

        with opt:
            batch = self.get_batch_inference(batch, Multimodal)
            label = self.get_batch_inference(label, Multimodal)
            for idx, real_x in enumerate(batch):
                if training and Multimodal and \
                        idx == self.config.style_train_debug:
                    break
                real_x = to_var(real_x, volatile=True)
                target_list = target_debug_list(
                    real_x.size(0), self.config.c_dim, config=self.config)

                # Start translations
                fake_image_list = [real_x]
                fake_attn_list = [to_var(denorm(real_x.data), volatile=True)]
                fake_image_list, fake_attn_list = self.Create_Visual_List(
                    real_x)

                if self.config.dataset_fake in self.MultiLabel_Datasets \
                        and label is None:
                    out_label = self._CLS(real_x)
                elif label is not None:

                    out_label = to_var(label[idx].squeeze(), volatile=True)
                else:
                    out_label = torch.zeros(real_x.size(0), self.config.c_dim)
                    out_label = to_var(out_label, volatile=True)

                if fixed_style is None:
                    style = self.G.random_style(real_x.size(0))
                    style = to_var(style, volatile=True)
                else:
                    style = to_var(
                        fixed_style[:real_x.size(0)], volatile=True)

                for k, target in enumerate(target_list):
                    if self.config.dataset_fake in self.MultiLabel_Datasets:
                        target = (out_label - target)**2  # Swap labels
                        target = self.target_multiAttr(target, k)
                    start_time = time.time()
                    target, style = Modality(target, style, Multimodal)
                    fake_x = self.G(real_x, target, style)
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    if TIME and flag_time:
                        print("[{}] Time/batch x forward (bs:{}): {}".format(
                            modal, real_x.size(0), elapsed))
                        flag_time = False

                    fake_image_list.append(fake_x[0])
                    fake_attn_list.append(fake_x[1].repeat(1, 3, 1, 1))

                # Create Folder
                if training:
                    _name = '' if fixed_style is not None \
                            and Multimodal else '_Random'
                    _save_path = save_path.replace('.jpg', _name + '.jpg')
                else:
                    _name = '' if fixed_style is not None else '_Random'
                    _save_path = os.path.join(
                        save_path.replace('.jpg', ''), '{}_{}{}.jpg'.format(
                            Multimodal,
                            str(idx).zfill(4), _name))
                    create_dir(_save_path)

                mode = 'fake' if not Multimodal else chr(65 + idx)
                Output.extend(
                    self._SAVE_IMAGE(_save_path, fake_image_list, mode=mode))
                Output.extend(
                    self._SAVE_IMAGE(
                        _save_path, fake_attn_list, Attention=True, mode=mode))

        self.G.train()
        self.D.train()
        if output:
            return Output
