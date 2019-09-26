from mpi4py import MPI
import torch.utils.data.distributed
from misc.utils import to_cuda, to_data, to_var, to_numpy, interpolation
from misc.utils import PRINT, single_source, target_debug_list
from misc.utils import create_arrow, create_circle, create_dir, denorm
from misc.utils import color_frame, get_labels, get_torch_version
from torchvision.utils import save_image
import datetime
import time
import numpy as np
import torch.nn.functional as F
import warnings
import glob
import os
import torch
from misc.utils import horovod
hvd = horovod()
comm = MPI.COMM_WORLD
warnings.filterwarnings('ignore')


class Solver(object):
    def __init__(self, config, data_loader=None):
        # Data loader
        self.data_loader = data_loader
        self.config = config
        self.verbose = 1 if hvd.rank() == 0 else 0
        self.build_model()

    # ==================================================================#
    # ==================================================================#
    def build_model(self):
        # Define a generator and a discriminator
        from models import Discriminator
        from models import AdaInGEN as Generator
        self.count = 0
        self.D = Discriminator(
            self.config, debug=self.config.mode == 'train' and self.verbose)
        self.D = to_cuda(self.D)
        self.G = Generator(
            self.config, debug=self.config.mode == 'train' and self.verbose)
        self.G = to_cuda(self.G)

        if self.config.mode == 'train':
            self.d_optimizer = self.set_optimizer(
                self.D, self.config.d_lr, self.config.beta1, self.config.beta2)
            self.g_optimizer = self.set_optimizer(
                self.G, self.config.g_lr, self.config.beta1, self.config.beta2)

        # Start with trained model
        if self.config.pretrained_model and self.verbose:
            self.load_pretrained_model()

        if self.config.mode == 'train' and self.verbose:
            self.print_network(self.D, 'Discriminator')
            self.print_network(self.G, 'Generator')

    # ==================================================================#
    # ==================================================================#
    def set_optimizer(self, model, lr, beta1=0.5, beta2=0.999):
        if torch.cuda.device_count() > 1 and hvd.size() == 1:
            model = model.module
        optimizer = torch.optim.Adam(model.parameters(), lr, [beta1, beta2])

        if hvd.size() > 1:
            optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=model.named_parameters())
            # Horovod: broadcast parameters & optimizer state.
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        return optimizer

    # ============================================================#
    # ============================================================#
    def imshow(self, img):
        import matplotlib.pyplot as plt
        img = to_data(denorm(img), cpu=True).numpy()
        img = img.transpose(1, 2, 0)
        plt.imshow(img)
        plt.show()

    # ==================================================================#
    # ==================================================================#

    def print_network(self, model, name):
        if torch.cuda.device_count() > 1 and hvd.size() == 1:
            model = model.module

        submodel = []
        if name == 'Generator':
            choices = ['generator', 'Domain_Embedding']
            for m in choices:
                submodel.append((m, getattr(model, m)))
        else:
            submodel.append((name, model))

        for name, model in submodel:
            num_params = 0
            num_learns = 0
            for p in model.parameters():
                num_params += p.numel()
                if p.requires_grad:
                    num_learns += p.numel()
            self.PRINT(
                "{} number of parameters (TOTAL): {}\t(LEARNABLE): {}.".format(
                    name.upper(), num_params, num_learns))
        # self.PRINT(name)
        # self.PRINT(model)

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
        torch.save(self.D.state_dict(), name.format('D'))

        def remove(name_1, mode):
            if os.path.isfile(name_1.format(mode)):
                os.remove(name_1.format(mode))

        if self.config.model_epoch != 1 and int(
                Epoch) % self.config.model_epoch == 0:
            for _epoch in range(
                    int(Epoch) - self.config.model_epoch + 1, int(Epoch)):
                name_1 = os.path.join(
                    self.config.model_save_path, '{}_{}_{}.pth'.format(
                        str(_epoch).zfill(4), iter, '{}'))
                for mode in ['G', 'D']:
                    remove(name_1, mode)

    # ==================================================================#
    # ==================================================================#
    def load_pretrained_model(self):
        self.PRINT('Resuming model (step: {})...'.format(
            self.config.pretrained_model))
        self.name = os.path.join(
            self.config.model_save_path, '{}_{}.pth'.format(
                self.config.pretrained_model, '{}'))
        self.PRINT('Model: {}'.format(self.name))

        self.name = comm.bcast(self.name, root=0)

        def load(model, name='G', replace=False):
            weights = torch.load(
                self.name.format(name),
                map_location=lambda storage, loc: storage)
            if replace:
                weights = {
                    key.replace('adain_net', 'Domain_Embedding'): item
                    for key, item in weights.items()
                }
            model.load_state_dict(weights)

        try:
            load(self.G, 'G')
        except RuntimeError:
            load(self.G, 'G', replace=True)
        load(self.D, 'D')

        print("Success!!")

    # ==================================================================#
    # ==================================================================#
    def resume_name(self, model_path=None):
        if model_path is None:
            model_path = self.config.model_save_path
        if self.config.pretrained_model in ['', None]:
            try:
                last_file = sorted(
                    glob.glob(os.path.join(model_path, '*_G.pth')))[-1]
            except IndexError:
                raise IndexError("No model found at " + model_path)
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
        dataset_string = colored(self.config.dataset_fake, 'red')
        Log += ' [*{}]'.format(dataset_string)
        self.PRINT(Log)
        return Log

    # ==================================================================#
    # ==================================================================#
    def PRINT(self, str):
        if self.verbose:
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

    # ============================================================#
    # ============================================================#
    def random_style(self, data, seed=None):
        if torch.cuda.device_count() > 1 and hvd.size() == 1:
            return self.G.module.random_style(data, seed=seed)
        else:
            return self.G.random_style(data, seed=seed)

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
    def _SAVE_IMAGE(self,
                    save_path,
                    fake_list,
                    Attention=False,
                    mode='fake',
                    circle=False,
                    arrow=False,
                    no_label=False):
        fake_images = torch.cat(fake_list, dim=3)
        if 'fake' not in os.path.basename(save_path):
            save_path = save_path.replace('.jpg', '_fake.jpg')
        if Attention:
            mode = mode + '_attn'
        else:
            fake_images = denorm(fake_images)

        save_path = save_path.replace('fake', mode)
        if circle:
            fake_images = create_circle(fake_images, self.config.image_size)
        if not no_label:
            fake_images = torch.cat((self.get_labels(), fake_images), dim=0)
        save_image(fake_images, save_path, nrow=1, padding=0)
        # if arrow or no_label:
        if arrow:
            create_arrow(
                save_path,
                arrow,
                image_size=self.config.image_size,
                horizontal=no_label)
        return save_path

    # ==================================================================#
    # ==================================================================#
    def target_multiAttr(self, target, index):
        if self.config.dataset_fake == 'CelebA':
            all_attr = self.data_loader.dataset.selected_attrs
            attr2idx = self.data_loader.dataset.attr2idx

            def replace(attrs):
                if all_attr[index] in attrs:
                    for attr in attrs:
                        if attr in all_attr:
                            target[:, attr2idx[attr]] = 0
                    target[:, index] = 1

            color_hair = [
                'Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'
            ]
            style_hair = ['Bald', 'Straight_Hair', 'Wavy_Hair']
            # ammount_hair = ['Bald', 'Bangs']
            replace(color_hair)
            replace(style_hair)
        return target

    # ==================================================================#
    # ==================================================================#
    def Create_Visual_List(self, batch, Multimodal=False):
        batch = to_data(batch)
        if Multimodal:
            fake_image_list = single_source(batch)
            fake_attn_list = single_source(denorm(batch))
            fake_image_list = color_frame(
                fake_image_list, thick=5, color='green', first=True)
            fake_attn_list = color_frame(
                fake_attn_list, thick=5, color='green', first=True)
            fake_image_list = [fake_image_list.cpu()]
            fake_attn_list = [fake_attn_list.cpu()]
        else:
            fake_image_list = [batch.cpu()]
            fake_attn_list = [denorm(batch).cpu()]

        return fake_image_list, fake_attn_list

    # ==================================================================#
    # ==================================================================#
    @property
    def MultiLabel_Datasets(self):
        return ['BP4D', 'CelebA', 'EmotionNet', 'DEMO']

    # ==================================================================#
    # ==================================================================#
    @property
    def Binary_Datasets(self):
        return ['Image2Edges', 'Yosemite']

    # ==================================================================#
    # ==================================================================#
    def get_batch_inference(self, batch, Multimodal):
        if Multimodal:
            if Multimodal > 1:
                n_rows = self.config.n_interpolation
            else:
                n_rows = self.config.style_debug
            batch = [img.unsqueeze(0).repeat(n_rows, 1, 1, 1) for img in batch]
        else:
            batch = [batch]
        return batch

    # ==================================================================#
    # ==================================================================#
    def label2embedding(self, target, style, _torch=False):
        assert target.max() == 1 and target.min() == 0
        in_de = self.G.preprocess(target, style)
        in_de = self.G.Domain_Embedding(in_de)
        if not _torch:
            in_de = to_numpy(in_de, data=True, cpu=True)
        return in_de

    # ==================================================================#
    # ==================================================================#
    def MMInterpolation(self, targets, styles, n_interp=None):
        assert len(targets) == 2 and len(styles) == 2
        if n_interp is None:
            n_interp = self.config.n_interpolation
        in_de0 = self.label2embedding(targets[0], styles[0])
        in_de1 = self.label2embedding(targets[1], styles[1])
        domain_interp = torch.zeros((n_interp, targets[0].size(0),
                                     in_de0.shape[-1]))
        domain_interp = to_var(domain_interp, volatile=True)
        for i in range(targets[0].size(0)):
            domain_interp[:, i] = interpolation(in_de0[i], in_de1[i], n_interp)
        return domain_interp

    # ==================================================================#
    # ==================================================================#
    def Modality(self, target, style, Multimodality, idx=0):
        _size = target.size(0)
        if self.config.dataset_fake in self.MultiLabel_Datasets:
            target = (self.org_label - target)**2  # Swap labels
            target = self.target_multiAttr(target, idx)
            target = to_var(target, volatile=True)

        if Multimodality == 1:
            # Random Styles
            domain_embedding = self.label2embedding(target, style, _torch=True)

        elif Multimodality == 2:
            # Style interpolation | Fixed Labels
            # The batch belongs to the same image
            style0 = style[0].repeat(_size, 1)
            style1 = style[1].repeat(_size, 1)
            targets = [target, target]
            styles = [style0, style1]
            domain_embedding = self.MMInterpolation(targets, styles)[:, 0]

        elif Multimodality == 3:
            # Style constant | Progressive swap label
            n_interp = self.config.n_interpolation + 5
            target0 = self.org_label
            target1 = target
            style = style[0].repeat(_size, 1)
            targets = [target0, target1]
            styles = [style, style]
            domain_embedding = self.MMInterpolation(targets, styles,
                                                    n_interp)[5:, 0]

        else:
            # Unimodal
            style = style[0].repeat(_size, 1)
            domain_embedding = self.label2embedding(target, style, _torch=True)

        return domain_embedding

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
                      TIME=False,
                      **kwargs):
        self.G.eval()
        self.D.eval()
        modal = 'Multimodal' if Multimodal else 'Unimodal'
        Output = []
        flag_time = True
        no_grad = open('/var/tmp/null.txt',
                       'w') if get_torch_version() < 1.0 else torch.no_grad()
        with no_grad:
            batch = self.get_batch_inference(batch, Multimodal)
            _label = self.get_batch_inference(label, Multimodal)
            for idx, real_x in enumerate(batch):
                if training and Multimodal and \
                        idx == self.config.style_train_debug:
                    break
                real_x = to_var(real_x, volatile=True)
                label = _label[idx]
                target_list = target_debug_list(
                    real_x.size(0), self.config.c_dim, config=self.config)

                # Start translations
                fake_image_list, fake_attn_list = self.Create_Visual_List(
                    real_x, Multimodal=Multimodal)
                if self.config.dataset_fake in self.MultiLabel_Datasets \
                        and label is None:
                    self.org_label = self._CLS(real_x)
                elif label is not None:
                    self.org_label = to_var(label.squeeze(), volatile=True)
                else:
                    self.org_label = torch.zeros(
                        real_x.size(0), self.config.c_dim)
                    self.org_label = to_var(self.org_label, volatile=True)

                if fixed_style is None:
                    style = self.random_style(real_x.size(0))
                    style = to_var(style, volatile=True)
                else:
                    style = to_var(fixed_style[:real_x.size(0)], volatile=True)

                for k, target in enumerate(target_list):
                    start_time = time.time()
                    embeddings = self.Modality(
                        target, style, Multimodal, idx=k)
                    fake_x = self.G(real_x, target, style, DE=embeddings)
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    if TIME and flag_time:
                        print("[{}] Time/batch x forward (bs:{}): {}".format(
                            modal, real_x.size(0), elapsed))
                        flag_time = False

                    fake_image_list.append(to_data(fake_x[0], cpu=True))
                    fake_attn_list.append(
                        to_data(fake_x[1].repeat(1, 3, 1, 1), cpu=True))

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

                mode = 'fake' if not Multimodal else 'style_' + chr(65 + idx)
                Output.extend(
                    self._SAVE_IMAGE(
                        _save_path, fake_image_list, mode=mode, **kwargs))
                Output.extend(
                    self._SAVE_IMAGE(
                        _save_path,
                        fake_attn_list,
                        Attention=True,
                        mode=mode,
                        **kwargs))

        self.G.train()
        self.D.train()
        if output:
            return Output
