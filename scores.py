from solver import Solver
import torch
import os
import warnings
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from misc.utils import PRINT, to_cuda, to_data, to_var
from data_loader import get_loader

warnings.filterwarnings('ignore')


def FID(path, gpu=''):
    from misc.fid_score import calculate_fid_given_paths
    fid_value = calculate_fid_given_paths(path, 64, gpu, 2048)
    print('FID: ', fid_value)
    return fid_value


class Scores(Solver):
    def __init__(self, config):

        super(Scores, self).__init__(config)
        self.data_loader = get_loader(
            config.mode_data,
            config.image_size,
            1,
            config.dataset_fake,
            config.mode,
            num_workers=config.num_workers,
            all_attr=config.ALL_ATTR,
            c_dim=config.c_dim)

    def LPIPS(self):
        from misc.utils import compute_lpips
        data_loader = self.data_loader
        n_images = 100
        pair_styles = 20
        model = None
        DISTANCE = {0: [], 1: []}
        self.G.eval()
        for i, (real_x, org_c, files) in tqdm(
                enumerate(data_loader), desc='Calculating LPISP',
                total=n_images):
            for _real_x, _org_c in zip(real_x, org_c):
                _real_x = _real_x.unsqueeze(0)
                _org_c = _org_c.unsqueeze(0)
                if len(DISTANCE[_org_c[0, 0]]) >= i:
                    continue
                _real_x = to_var(_real_x, volatile=True)
                target_c = to_var(1 - _org_c, volatile=True)
                for _ in range(pair_styles):
                    style0 = to_var(
                        self.G.random_style(_real_x.size(0)), volatile=True)
                    style1 = to_var(
                        self.G.random_style(_real_x.size(0)), volatile=True)
                    fake_x0 = self.G(_real_x, target_c, stochastic=style0)
                    fake_x1 = self.G(_real_x, target_c, stochastic=style1)
                    distance, model = compute_lpips(
                        fake_x0, fake_x1, model=model)
                    DISTANCE[org_c[0, 0]].append(distance)
                if i == len(DISTANCE[0, 0]) == len(DISTANCE[1]):
                    break
        print("LPISP a-b: {}".format(np.array(DISTANCE[0]).mean()))
        print("LPISP b-a: {}".format(np.array(DISTANCE[1]).mean()))

    # ==================================================================#
    # ==================================================================#

    def LPIPS_REAL(self):
        from misc.utils import compute_lpips
        data_loader = self.data_loader
        model = None
        file_name = 'scores/{}_Attr_{}_LPIPS.txt'.format(
            self.config.dataset_fake, self.config.ALL_ATTR)
        if os.path.isfile(file_name):
            print(file_name)
            for line in open(file_name).readlines():
                print(line.strip())
            return

        DISTANCE = {
            i: []
            for i in range(len(data_loader.dataset.labels[0]) + 1)
        }  # 0:[], 1:[], 2:[]}
        for i, (real_x, org_c, files) in tqdm(
                enumerate(data_loader),
                desc='Calculating LPISP - {}'.format(file_name),
                total=len(data_loader)):
            for label in range(len(data_loader.dataset.labels[0])):
                for j, (_real_x, _org_c, _files) in enumerate(data_loader):
                    if j <= i:
                        continue
                    _org_label = torch.max(_org_c, 1)[1][0]
                    for _label in range(len(data_loader.dataset.labels[0])):
                        if _org_label == _label:
                            continue
                        distance, model = compute_lpips(
                            real_x, _real_x, model=model)
                        DISTANCE[len(data_loader.dataset.labels[0])].append(
                            distance[0])
                        if label == _label:
                            DISTANCE[_label].append(distance[0])

        file_ = open(file_name, 'w')
        DISTANCE = {k: np.array(v) for k, v in DISTANCE.items()}
        for key, values in DISTANCE.items():
            if key == len(data_loader.dataset.labels[0]):
                mode = 'All'
            else:
                mode = chr(65 + key)
            PRINT(
                file_, "LPISP {}: {} +/- {}".format(mode, values.mean(),
                                                    values.std()))
        file_.close()

    # ==================================================================#
    # ==================================================================#

    def LPIPS_UNIMODAL(self):
        from misc.utils import compute_lpips
        from shutil import copyfile
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        data_loader = self.data_loader
        model = None
        style_fixed = True
        style_str = 'fixed' if style_fixed else 'random'
        file_name = os.path.join(
            self.name.replace('{}.pth',
                              'LPIPS_UNIMODAL_{}.txt'.format(style_str)))
        copy_name = 'scores/{}_Attr_{}_LPIPS_UNIMODAL_{}.txt'.format(
            self.config.dataset_fake, self.config.ALL_ATTR, style_str)
        if os.path.isfile(file_name):
            print(file_name)
            for line in open(file_name).readlines():
                print(line.strip())
            return

        DISTANCE = {
            i: []
            for i in range(len(data_loader.dataset.labels[0]) + 1)
        }  # 0:[], 1:[], 2:[]}
        n_images = {i: 0 for i in range(len(data_loader.dataset.labels[0]))}

        style0 = to_var(
            self.G.random_style(1),
            volatile=True) if not self.config.DETERMINISTIC else None
        print(file_name)
        for i, (real_x, org_c, files) in tqdm(
                enumerate(data_loader),
                desc='Calculating LPISP ',
                total=len(data_loader)):
            org_label = torch.max(org_c, 1)[1][0]
            real_x = to_var(real_x, volatile=True)
            for label in range(len(data_loader.dataset.labels[0])):
                if org_label == label:
                    continue
                target_c = to_var(org_c * 0, volatile=True)
                target_c[:, label] = 1
                if not style_fixed:
                    style0 = to_var(
                        self.G.random_style(real_x.size(0)), volatile=True)
                real_x = self.G(
                    real_x, to_var(target_c, volatile=True),
                    stochastic=style0)[0]
                n_images[label] += 1
                for j, (_real_x, _org_c, _files) in enumerate(data_loader):
                    if j <= i:
                        continue
                    _org_label = torch.max(_org_c, 1)[1][0]
                    _real_x = to_var(_real_x, volatile=True)
                    for _label in range(len(data_loader.dataset.labels[0])):
                        if _org_label == _label:
                            continue
                        _target_c = to_var(_org_c * 0, volatile=True)
                        _target_c[:, _label] = 1
                        if not style_fixed:
                            style0 = to_var(
                                self.G.random_style(_real_x.size(0)),
                                volatile=True)
                        _real_x = self.G(
                            _real_x,
                            to_var(_target_c, volatile=True),
                            stochastic=style0)[0]
                        distance, model = compute_lpips(
                            real_x.data, _real_x.data, model=model)
                        DISTANCE[len(data_loader.dataset.labels[0])].append(
                            distance[0])
                        if label == _label:
                            DISTANCE[_label].append(distance[0])

        file_ = open(file_name, 'w')
        DISTANCE = {k: np.array(v) for k, v in DISTANCE.items()}
        for key, values in DISTANCE.items():
            if key == len(data_loader.dataset.labels[0]):
                mode = 'All'
            else:
                mode = chr(65 + key)
            PRINT(
                file_, "LPISP {}: {} +/- {}".format(mode, values.mean(),
                                                    values.std()))
        file_.close()
        copyfile(file_name, copy_name)

    # ==================================================================#
    # ==================================================================#

    def LPIPS_MULTIMODAL(self):
        from misc.utils import compute_lpips

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)

        data_loader = self.data_loader
        model = None
        n_images = 20

        file_name = os.path.join(
            self.name.replace('{}.pth', 'LPIPS_MULTIMODAL.txt'))
        if os.path.isfile(file_name):
            print(file_name)
            for line in open(file_name).readlines():
                print(line.strip())

        # DISTANCE = {0:[], 1:[], 2:[]}
        DISTANCE = {
            i: []
            for i in range(len(data_loader.dataset.labels[0]) + 1)
        }  # 0:[], 1:[], 2:[]}
        print(file_name)
        for i, (real_x, org_c, files) in tqdm(
                enumerate(data_loader),
                desc='Calculating LPISP ',
                total=len(data_loader)):
            org_label = torch.max(org_c, 1)[1][0]
            for label in range(len(data_loader.dataset.labels[0])):
                if org_label == label:
                    continue
                target_c = org_c * 0
                target_c[:, label] = 1
                target_c = target_c.repeat(n_images, 1)
                real_x_var = to_var(
                    real_x.repeat(n_images, 1, 1, 1), volatile=True)
                target_c = to_var(target_c, volatile=True)
                style = to_var(self.G.random_style(n_images), volatile=True)
                # ipdb.set_trace()
                fake_x = self.G(real_x_var, target_c, stochastic=style)[0].data
                fake_x = [f.unsqueeze(0) for f in fake_x]
                # ipdb.set_trace()
                _DISTANCE = []
                for ii, fake0 in enumerate(fake_x):
                    for jj, fake1 in enumerate(fake_x):
                        if jj <= ii:
                            continue
                        distance, model = compute_lpips(
                            fake0, fake1, model=model)
                        _DISTANCE.append(distance[0])
                DISTANCE[len(data_loader.dataset.labels[0])].append(
                    np.array(_DISTANCE).mean())
                DISTANCE[label].append(DISTANCE[len(
                    data_loader.dataset.labels[0])][-1])

        file_ = open(file_name, 'w')
        DISTANCE = {k: np.array(v) for k, v in DISTANCE.items()}
        for key, values in DISTANCE.items():
            if key == len(data_loader.dataset.labels[0]):
                mode = 'All'
            else:
                mode = chr(65 + key)
            PRINT(
                file_, "LPISP {}: {} +/- {}".format(mode, values.mean(),
                                                    values.std()))
        file_.close()

    # ==================================================================#
    # ==================================================================#

    def INCEPTION(self):
        from misc.utils import load_inception
        from scipy.stats import entropy
        n_styles = 20
        net = load_inception()
        net = to_cuda(net)
        net.eval()
        self.G.eval()
        inception_up = nn.Upsample(size=(299, 299), mode='bilinear')
        if not self.config.DETERMINISTIC:
            mode = 'SMIT'
        elif not self.config.NO_ATTENTION:
            mode = 'StarGAN'
        else:
            mode = 'GANimation'
        data_loader = self.data_loader
        file_name = 'scores/Inception_{}.txt'.format(mode)

        PRED_IS = {i: []
                   for i in range(len(data_loader.dataset.labels[0]))
                   }  # 0:[], 1:[], 2:[]}
        CIS = {i: [] for i in range(len(data_loader.dataset.labels[0]))}
        IS = {i: [] for i in range(len(data_loader.dataset.labels[0]))}

        for i, (real_x, org_c, files) in tqdm(
                enumerate(data_loader),
                desc='Calculating CIS/IS - {}'.format(file_name),
                total=len(data_loader)):
            PRED_CIS = {
                i: []
                for i in range(len(data_loader.dataset.labels[0]))
            }  # 0:[], 1:[], 2:[]}
            org_label = torch.max(org_c, 1)[1][0]
            real_x = real_x.repeat(n_styles, 1, 1, 1)  # .unsqueeze(0)
            real_x = to_var(real_x, volatile=True)

            target_c = (org_c * 0).repeat(n_styles, 1)
            target_c = to_var(target_c, volatile=True)
            for label in range(len(data_loader.dataset.labels[0])):
                if org_label == label:
                    continue
                target_c *= 0
                target_c[:, label] = 1
                style = to_var(
                    self.G.random_style(n_styles),
                    volatile=True) if mode == 'SMIT' else None

                fake = (self.G(real_x, target_c, style)[0] + 1) / 2

                pred = to_data(
                    F.softmax(net(inception_up(fake)), dim=1),
                    cpu=True).numpy()
                PRED_CIS[label].append(pred)
                PRED_IS[label].append(pred)

                # CIS for each image
                PRED_CIS[label] = np.concatenate(PRED_CIS[label], 0)
                py = np.sum(
                    PRED_CIS[label], axis=0
                )  # prior is computed from outputs given a specific input
                for j in range(PRED_CIS[label].shape[0]):
                    pyx = PRED_CIS[label][j, :]
                    CIS[label].append(entropy(pyx, py))

        for label in range(len(data_loader.dataset.labels[0])):
            PRED_IS[label] = np.concatenate(PRED_IS[label], 0)
            py = np.sum(
                PRED_IS[label], axis=0)  # prior is computed from all outputs
            for j in range(PRED_IS[label].shape[0]):
                pyx = PRED_IS[label][j, :]
                IS[label].append(entropy(pyx, py))

        total_cis = []
        total_is = []
        file_ = open(file_name, 'w')
        for label in range(len(data_loader.dataset.labels[0])):
            cis = np.exp(np.mean(CIS[label]))
            total_cis.append(cis)
            _is = np.exp(np.mean(IS[label]))
            total_is.append(_is)
            PRINT(file_, "Label {}".format(label))
            PRINT(file_, "Inception Score: {:.4f}".format(_is))
            PRINT(file_, "conditional Inception Score: {:.4f}".format(cis))
        PRINT(file_, "")
        PRINT(
            file_, "[TOTAL] Inception Score: {:.4f} +/- {:.4f}".format(
                np.array(total_is).mean(),
                np.array(total_is).std()))
        PRINT(
            file_,
            "[TOTAL] conditional Inception Score: {:.4f} +/- {:.4f}".format(
                np.array(total_cis).mean(),
                np.array(total_cis).std()))
        file_.close()

    def INCEPTION_REAL(self):
        from misc.utils import load_inception
        from scipy.stats import entropy
        net = load_inception()
        to_cuda(net)
        net.eval()
        inception_up = nn.Upsample(size=(299, 299), mode='bilinear')
        mode = 'Real'
        data_loader = self.data_loader
        file_name = 'scores/Inception_{}.txt'.format(mode)

        # 0:[], 1:[], 2:[]}
        PRED_IS = {i: [] for i in range(len(data_loader.dataset.labels[0]))}
        IS = {i: [] for i in range(len(data_loader.dataset.labels[0]))}

        for i, (real_x, org_c, files) in tqdm(
                enumerate(data_loader),
                desc='Calculating CIS/IS - {}'.format(file_name),
                total=len(data_loader)):
            label = torch.max(org_c, 1)[1][0]
            real_x = to_var((real_x + 1) / 2., volatile=True)
            pred = to_data(
                F.softmax(net(inception_up(real_x)), dim=1), cpu=True).numpy()
            PRED_IS[label].append(pred)

        for label in range(len(data_loader.dataset.labels[0])):
            PRED_IS[label] = np.concatenate(PRED_IS[label], 0)
            # prior is computed from all outputs
            py = np.sum(PRED_IS[label], axis=0)
            for j in range(PRED_IS[label].shape[0]):
                pyx = PRED_IS[label][j, :]
                IS[label].append(entropy(pyx, py))

        # ipdb.set_trace()
        total_is = []
        file_ = open(file_name, 'w')
        for label in range(len(data_loader.dataset.labels[0])):
            _is = np.exp(np.mean(IS[label]))
            total_is.append(_is)
            PRINT(file_, "Label {}".format(label))
            PRINT(file_, "Inception Score: {:.4f}".format(_is))
        PRINT(file_, "")
        PRINT(
            file_, "[TOTAL] Inception Score: {:.4f} +/- {:.4f}".format(
                np.array(total_is).mean(),
                np.array(total_is).std()))
        file_.close()
