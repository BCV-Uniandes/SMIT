from solver import Solver
import torch
import os
import warnings
import numpy as np
from misc.utils import color_frame, create_dir, get_torch_version, slerp
from misc.utils import single_source, TimeNow_str, to_data, to_var
import torch.utils.data.distributed

warnings.filterwarnings('ignore')


class Test(Solver):
    def __init__(self, config, data_loader):
        super(Test, self).__init__(config, data_loader)

    def save_multimodal_output(self,
                               real_x,
                               label,
                               save_path,
                               interpolation=False,
                               **kwargs):
        self.G.eval()
        self.D.eval()
        n_rep = 4
        opt = torch.no_grad() if get_torch_version() > 0.3 else open(
            '/tmp/_null.txt', 'w')

        with opt:
            real_x = to_var(real_x, volatile=True)
            out_label = to_var(label, volatile=True)
            target_c_list = [out_label] * 7

            for idx, real_x0 in enumerate(real_x):
                _name = 'multimodal'
                if interpolation:
                    _name = _name + '_interp'
                _save_path = os.path.join(
                    save_path.replace('.jpg', ''), '{}_{}.jpg'.format(
                        _name,
                        str(idx).zfill(4)))
                create_dir(_save_path)
                real_x0 = real_x0.repeat(n_rep, 1, 1, 1)  # .unsqueeze(0)
                fake_image_list = [
                    color_frame(
                        single_source(real_x0),
                        thick=5,
                        color='green',
                        first=True)
                ]
                fake_attn_list = [
                    color_frame(
                        single_source(real_x0),
                        thick=5,
                        color='green',
                        first=True)
                ]

                for n_label, _target_c in enumerate(target_c_list):
                    target_c = _target_c[0].repeat(n_rep, 1)
                    if not interpolation:
                        style_ = self.G.random_style(n_rep)
                    else:
                        z0 = to_data(
                            self.G.random_style(1), cpu=True).numpy()[0]
                        z1 = to_data(
                            self.G.random_style(1), cpu=True).numpy()[0]
                        style_ = self.G.random_style(n_rep)
                        style_[:] = torch.FloatTensor(
                            np.array([
                                slerp(sz, z0, z1)
                                for sz in np.linspace(0, 1, n_rep)
                            ]))
                    style = to_var(style_, volatile=True)
                    fake_x = self.G(real_x0, target_c, stochastic=style)
                    fake_image_list.append(fake_x[0])
                    fake_attn_list.append(fake_x[1].repeat(1, 3, 1, 1))
                self._SAVE_IMAGE(
                    _save_path,
                    fake_image_list,
                    im_size=self.config.image_size,
                    mode='style_' + chr(65 + idx),
                    no_labels=True)
                self._SAVE_IMAGE(
                    _save_path.replace(_name, _name + '_attn'),
                    fake_attn_list,
                    im_size=self.config.image_size,
                    mode='style_' + chr(65 + idx),
                    no_labels=True)
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
        data_loader = get_loader(
            path,
            self.config.image_size,
            batch_size,
            shuffling=False,
            dataset='DEMO',
            mode='test',
            many_faces=self.config.many_faces)
        label = self.config.DEMO_LABEL
        if self.config.DEMO_LABEL != '':
            label = torch.FloatTensor([int(i) for i in label.split(',')]).view(
                1, -1)
        else:
            label = None
        if self.config.Deterministic:
            _debug = range(self.config.style_label_debug + 1)
            style_all = self.G.random_style(max(self.config.batch_size, 50))
        else:
            style_all = None
            _debug = range(1)

        name = TimeNow_str()
        Output = []
        if not self.config.many_faces:
            for i, real_x in enumerate(data_loader):
                save_path = os.path.join(save_folder, 'DEMO_{}_{}.jpg'.format(
                    name, i + 1))
                self.PRINT('Translated test images and saved into "{}"..!'.
                           format(save_path))
                for k in _debug:
                    output = self.save_fake_output(
                        real_x,
                        save_path,
                        gif=False,
                        label=label,
                        output=True,
                        Style=k,
                        fixed_style=style_all,
                        TIME=not i)
                    if self.config.many_faces:
                        Output.append(output)
                        break
                    if self.config.Deterministic:
                        output = self.save_fake_output(
                            real_x,
                            save_path,
                            gif=False,
                            label=label,
                            output=True,
                            Style=k)

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
        else:
            data_loader = get_loader(
                self.config.metadata_path,
                self.config.image_size,
                self.config.batch_size,
                shuffling=True,
                dataset=dataset,
                mode='test')

        if not self.config.Deterministic:
            _debug = range(self.config.style_label_debug + 1)
            style_all = self.G.random_style(self.config.batch_size)
        else:
            style_all = None
            _debug = range(1)

        string = '{}'.format(TimeNow_str())

        for i, (real_x, org_c, files) in enumerate(data_loader):
            save_path = os.path.join(
                save_folder, '{}_{}_{}.jpg'.format(dataset, '{}', i + 1))
            name = os.path.abspath(save_path.format(string))
            if self.config.dataset_fake == self.config.dataset_real:
                label = org_c
            else:
                label = None
            self.PRINT(
                'Translated test images and saved into "{}"..!'.format(name))

            if self.config.dataset_fake in ['Image2Edges', 'Yosemite']:
                self.save_multimodal_output(
                    real_x, 1 - org_c, name, interpolation=True)
                self.save_multimodal_output(real_x, 1 - org_c, name)

            self.generate_SMIT(
                real_x,
                name,
                label=label,
                output=True,
                fixed_style=style_all,
                TIME=not i)

            if not self.config.Deterministic:
                for k in _debug:
                    self.generate_SMIT(real_x, name, label=label, Multimodal=k)
