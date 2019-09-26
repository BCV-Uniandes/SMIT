from solver import Solver
import torch
import os
import warnings
from misc.utils import create_dir, get_torch_version
from misc.utils import TimeNow_str
from misc.utils import to_data, to_var

warnings.filterwarnings('ignore')


class Test(Solver):
    def __init__(self, config, data_loader):
        super(Test, self).__init__(config, data_loader)

    # ==================================================================#
    # ==================================================================#
    def save_multimodal_output(self,
                               real_x,
                               label,
                               save_path,
                               interpolation=False,
                               **kwargs):
        self.G.eval()
        self.D.eval()
        n_rep = 4
        no_label = self.config.dataset_fake in self.Binary_Datasets
        no_grad = open('/var/tmp/null.txt',
                       'w') if get_torch_version() < 1.0 else torch.no_grad()
        with no_grad:
            real_x = to_var(real_x, volatile=True)
            out_label = to_var(label, volatile=True)
            # target_c_list = [out_label] * 7

            for idx, (real_x0, real_c0) in enumerate(zip(real_x, out_label)):
                _name = 'multimodal'
                if interpolation == 1:
                    _name += '_interp'
                elif interpolation == 2:
                    _name = 'multidomain_interp'

                _save_path = os.path.join(
                    save_path.replace('.jpg', ''), '{}_{}.jpg'.format(
                        _name,
                        str(idx).zfill(4)))
                create_dir(_save_path)
                real_x0 = real_x0.repeat(n_rep, 1, 1, 1)
                real_c0 = real_c0.repeat(n_rep, 1)

                fake_image_list, fake_attn_list = self.Create_Visual_List(
                    real_x0, Multimodal=True)

                target_c_list = [real_c0] * 7
                for _, target_c in enumerate(target_c_list):
                    if interpolation == 0:
                        style_ = to_var(
                            self.G.random_style(n_rep), volatile=True)
                        embeddings = self.label2embedding(
                            target_c, style_, _torch=True)
                    elif interpolation == 1:
                        style_ = to_var(self.G.random_style(1), volatile=True)
                        style1 = to_var(self.G.random_style(1), volatile=True)
                        _target_c = target_c[0].unsqueeze(0)
                        styles = [style_, style1]
                        targets = [_target_c, _target_c]
                        embeddings = self.MMInterpolation(
                            targets, styles, n_interp=n_rep)[:, 0]
                    elif interpolation == 2:
                        style_ = to_var(self.G.random_style(1), volatile=True)
                        target0 = 1 - target_c[0].unsqueeze(0)
                        target1 = target_c[0].unsqueeze(0)
                        styles = [style_, style_]
                        targets = [target0, target1]
                        # import ipdb; ipdb.set_trace()
                        embeddings = self.MMInterpolation(
                            targets, styles, n_interp=n_rep)[:, 0]
                    else:
                        raise ValueError(
                            "There are only 2 types of interpolation:\
                            Multimodal and Multi-domain")
                    fake_x = self.G(real_x0, target_c, style_, DE=embeddings)
                    fake_image_list.append(to_data(fake_x[0], cpu=True))
                    fake_attn_list.append(
                        to_data(fake_x[1].repeat(1, 3, 1, 1), cpu=True))
                self._SAVE_IMAGE(
                    _save_path,
                    fake_image_list,
                    mode='style_' + chr(65 + idx),
                    no_label=no_label,
                    arrow=interpolation,
                    circle=False)
                self._SAVE_IMAGE(
                    _save_path,
                    fake_attn_list,
                    Attention=True,
                    mode='style_' + chr(65 + idx),
                    arrow=interpolation,
                    no_label=no_label,
                    circle=False)
        self.G.train()
        self.D.train()

    # ==================================================================#
    # ==================================================================#
    def save_multidomain_output(self, real_x, label, save_path, **kwargs):
        self.G.eval()
        self.D.eval()
        no_grad = open('/var/tmp/null.txt',
                       'w') if get_torch_version() < 1.0 else torch.no_grad()
        with no_grad:
            real_x = to_var(real_x, volatile=True)
            n_style = self.config.style_debug
            n_interp = self.config.n_interpolation + 10
            _name = 'domain_interpolation'
            no_label = True
            for idx in range(n_style):
                dirname = save_path.replace('.jpg', '')
                filename = '{}_style{}.jpg'.format(_name,
                                                   str(idx + 1).zfill(2))
                _save_path = os.path.join(dirname, filename)
                create_dir(_save_path)
                fake_image_list, fake_attn_list = self.Create_Visual_List(
                    real_x)
                style = self.G.random_style(1).repeat(real_x.size(0), 1)
                style = to_var(style, volatile=True)
                label0 = to_var(label, volatile=True)
                opposite_label = self.target_multiAttr(1 - label,
                                                       2)  # 2: black hair
                opposite_label[:, 7] = 0  # Pale skin
                label1 = to_var(opposite_label, volatile=True)
                labels = [label0, label1]
                styles = [style, style]
                domain_interp = self.MMInterpolation(
                    labels, styles, n_interp=n_interp)
                for target_de in domain_interp[5:]:
                    # target_de = target_de.repeat(real_x.size(0), 1)
                    target_de = to_var(target_de, volatile=True)
                    fake_x = self.G(real_x, target_de, style, DE=target_de)
                    fake_image_list.append(to_data(fake_x[0], cpu=True))
                    fake_attn_list.append(
                        to_data(fake_x[1].repeat(1, 3, 1, 1), cpu=True))
                self._SAVE_IMAGE(
                    _save_path,
                    fake_image_list,
                    no_label=no_label,
                    arrow=False,
                    circle=False)
                self._SAVE_IMAGE(
                    _save_path,
                    fake_attn_list,
                    Attention=True,
                    arrow=False,
                    no_label=no_label,
                    circle=False)
        self.G.train()
        self.D.train()

    # ==================================================================#
    # ==================================================================#
    def DEMO(self, path):
        from data_loader import get_loader
        last_name = self.resume_name()
        save_folder = os.path.join(self.config.sample_path,
                                   '{}_test'.format(last_name))
        create_dir(save_folder)
        batch_size = 1
        no_label = self.config.dataset_fake in self.Binary_Datasets
        data_loader = get_loader(
            path,
            self.config.image_size,
            batch_size,
            shuffling=False,
            dataset='DEMO',
            Detect_Face=True,
            mode='test')
        label = self.config.DEMO_LABEL
        if self.config.DEMO_LABEL != '':
            label = torch.FloatTensor([int(i) for i in label.split(',')]).view(
                1, -1)
        else:
            label = None
        _debug = range(self.config.style_label_debug + 1)
        style_all = self.G.random_style(max(self.config.batch_size, 50))

        name = TimeNow_str()
        for i, real_x in enumerate(data_loader):
            save_path = os.path.join(save_folder, 'DEMO_{}_{}.jpg'.format(
                name, i + 1))
            self.PRINT('Translated test images and saved into "{}"..!'.format(
                save_path))
            for k in _debug:
                self.generate_SMIT(
                    real_x,
                    save_path,
                    label=label,
                    Multimodal=k,
                    fixed_style=style_all,
                    TIME=not i,
                    no_label=no_label,
                    circle=True)
                self.generate_SMIT(
                    real_x,
                    save_path,
                    label=label,
                    Multimodal=k,
                    no_label=no_label,
                    circle=True)

    # ==================================================================#
    # ==================================================================#
    def __call__(self, dataset='', load=False):
        import os
        from data_loader import get_loader
        last_name = self.resume_name()
        save_folder = os.path.join(self.config.sample_path,
                                   '{}_test'.format(last_name))
        create_dir(save_folder)
        if dataset == '':
            dataset = self.config.dataset_fake
            data_loader = self.data_loader
            self.dataset_real = dataset
        else:
            data_loader = get_loader(
                self.config.mode_data,
                self.config.image_size,
                self.config.batch_size,
                shuffling=True,
                dataset=dataset,
                mode='test')

        _debug = range(1, self.config.style_label_debug + 1)
        style_all = self.G.random_style(self.config.batch_size)

        string = '{}'.format(TimeNow_str())
        for i, (real_x, org_c, _) in enumerate(data_loader):
            save_path = os.path.join(
                save_folder, '{}_{}_{}.jpg'.format(dataset, '{}', i + 1))
            name = os.path.abspath(save_path.format(string))
            if self.config.dataset_fake == dataset:
                label = org_c
            else:
                label = None
            self.PRINT(
                'Translated test images and saved into "{}"..!'.format(name))

            if self.config.dataset_fake in ['Image2Edges', 'Yosemite']:
                for k in range(self.config.style_label_debug):
                    self.save_multimodal_output(
                        real_x, 1 - org_c, name, interpolation=k)

            else:
                if self.config.dataset_fake in ['CelebA']:
                    self.save_multidomain_output(real_x, label, name)
                self.generate_SMIT(
                    real_x,
                    name,
                    label=label,
                    fixed_style=style_all,
                    TIME=not i)
                for k in _debug:
                    self.generate_SMIT(
                        real_x,
                        name,
                        label=label,
                        Multimodal=k,
                        TIME=not i and k == 1)
                    self.generate_SMIT(
                        real_x,
                        name,
                        label=label,
                        Multimodal=k,
                        fixed_style=style_all)
