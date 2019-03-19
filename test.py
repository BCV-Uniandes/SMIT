from solver import Solver
import torch
import os
import warnings
import imageio
import numpy as np
from misc.utils import color_frame, create_dir, denorm
from misc.utils import slerp, single_source, TimeNow_str, to_data, to_var
import torch.utils.data.distributed

warnings.filterwarnings('ignore')


class Test(Solver):
    def __init__(self, config, data_loader):
        super(Test, self).__init__(config, data_loader)

    # ==================================================================#
    # ==================================================================#
    def folder_fid(self, data_loader):
        from scores import FID
        self.G.eval()
        n_rep = 5
        save_folder = os.path.join(self.config.sample_path, self.resume_name())
        self.PRINT('FID Folder at "{}"..!'.format(save_folder))
        _dirs = [[os.path.join(save_folder, 'real_label0')]]
        _dirs[-1].append(os.path.join(save_folder, 'real_label1'))
        for i in range(1, n_rep + 1):
            _dirs.append([
                os.path.join(save_folder, 'fake%s_label0' % (str(i).zfill(2)))
            ])
            _dirs[-1].append(
                os.path.join(save_folder, 'fake%s_label1' % (str(i).zfill(2))))

        # for _dir in _dirs:
        #     for _di in _dir:
        #         os.system('rm -rf {}'.format(_di))
        #         create_dir(_di)

        def save_img(data, idx, pos, iter):
            path = os.path.join(_dirs[idx][pos], '{}_{}.jpg'.format(
                idx,
                str(iter).zfill(4)))
            if not os.path.isfile(path):
                imageio.imwrite(path, (data * 255).astype(np.uint8))

        # iter = 0
        # with torch.no_grad():
        #     for i, (real_x, label, _) in enumerate(data_loader):
        #         for _, (real_x0, label0) in enumerate(zip(real_x, label)):
        #             real_x0 = real_x0.repeat(n_rep, 1, 1, 1)  # .unsqueeze(0)
        #             label0 = (1 - label0.repeat(n_rep, 1))**2
        #             real_x0 = to_var(real_x0, volatile=True)
        #             label0 = to_var(label0, volatile=True)

        #             style = self.G.random_style(n_rep)
        #             style = to_var(style, volatile=True)
        #             fake_x0 = self.G(real_x0, label0, stochastic=style)[0]

        #             fake_x0 = denorm(to_data(fake_x0, cpu=True)).numpy()
        #             real_x0 = denorm(to_data(real_x0, cpu=True)).numpy()
        #             real_x0 = real_x0.transpose(0, 2, 3, 1)[0]
        #             save_img(real_x0, 0, int(label0[0][0]), iter)
        #             fake_x0 = fake_x0.transpose(0, 2, 3, 1)
        #             for i, data in enumerate(fake_x0):
        #                 save_img(data, i + 1, int(1 - label0[0][0]), iter)
        #             iter += 1

        for j in range(2):
            fid = []
            self.PRINT('Calculating FID - label {}'.format(j))
            for i in range(1, n_rep + 1):
                real_folder = os.path.join(save_folder,
                                           'real_label{}'.format(j))
                fake_folder = os.path.join(
                    save_folder, 'fake{}_label{}'.format(str(i).zfill(2), j))
                folder = [real_folder, fake_folder]
                fid.append(FID(folder, gpu=self.config.GPU[0]))
            self.PRINT('Mean FID: {}'.format(np.mean(fid)))

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

        with torch.no_grad():
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
                    to_data(
                        color_frame(
                            single_source(real_x0),
                            thick=5,
                            color='green',
                            first=True),
                        cpu=True)
                ]
                fake_attn_list = [
                    to_data(
                        color_frame(
                            single_source(real_x0),
                            thick=5,
                            color='green',
                            first=True),
                        cpu=True)
                ]

                for _, _target_c in enumerate(target_c_list):
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
                    fake_image_list.append(to_data(fake_x[0], cpu=True))
                    fake_attn_list.append(
                        to_data(fake_x[1].repeat(1, 3, 1, 1), cpu=True))
                self._SAVE_IMAGE(
                    _save_path,
                    fake_image_list,
                    mode='style_' + chr(65 + idx),
                    no_label=True)
                self._SAVE_IMAGE(
                    _save_path,
                    fake_attn_list,
                    Attention=True,
                    mode='style_' + chr(65 + idx),
                    no_label=True)
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
            Detect_Face=True,
            mode='test')
        label = self.config.DEMO_LABEL
        if self.config.DEMO_LABEL != '':
            label = torch.FloatTensor([int(i) for i in label.split(',')]).view(
                1, -1)
        else:
            label = None
        if not self.config.DETERMINISTIC:
            _debug = range(self.config.style_label_debug + 1)
            style_all = self.G.random_style(max(self.config.batch_size, 50))
        else:
            style_all = None
            _debug = range(1)

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
                    TIME=not i)
                if self.config.DETERMINISTIC:
                    self.generate_SMIT(
                        real_x, save_path, label=label, Multimodal=k)

    # ==================================================================#
    # ==================================================================#
    def __call__(self, dataset='', load=False):
        import os
        from data_loader import get_loader
        save_folder_fid = self.config.dataset_fake in [
            'Yosemite', 'Image2Edges'
        ]
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

        if not self.config.DETERMINISTIC:
            _debug = range(1, self.config.style_label_debug + 1)
            style_all = self.G.random_style(self.config.batch_size)
        else:
            style_all = None
            _debug = range(1)

        string = '{}'.format(TimeNow_str())
        if save_folder_fid:
            self.folder_fid(data_loader)
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
                self.save_multimodal_output(
                    real_x, 1 - org_c, name, interpolation=True)
                self.save_multimodal_output(real_x, 1 - org_c, name)

            self.generate_SMIT(
                real_x, name, label=label, fixed_style=style_all, TIME=not i)

            if not self.config.DETERMINISTIC:
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
