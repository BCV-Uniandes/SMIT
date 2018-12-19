from solver import Solver
import torch
import os
import time
import warnings
import datetime
import numpy as np
from tqdm import tqdm
from misc.utils import get_labels, get_loss_value
from misc.utils import TimeNow, to_var
from misc.losses import _compute_loss_smooth, _GAN_LOSS
import torch.utils.data.distributed

warnings.filterwarnings('ignore')


class Train(Solver):
    def __init__(self, config, data_loader):
        super(Train, self).__init__(config, data_loader)
        self.run()

    # ============================================================#
    # ============================================================#
    def update_lr(self, g_lr, d_lr):
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    # ============================================================#
    # ============================================================#
    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    # ============================================================#
    # ============================================================#
    def update_loss(self, loss, value):
        try:
            self.LOSS[loss].append(value)
        except BaseException:
            self.LOSS[loss] = []
            self.LOSS[loss].append(value)

    # ============================================================#
    # ============================================================#
    def get_labels(self):
        return get_labels(
            self.config.image_size,
            self.config.dataset_fake,
            attr=self.data_loader.dataset)

    # ============================================================#
    # ============================================================#
    def color(self, dict, key, color='red'):
        from termcolor import colored
        dict[key] = colored('%.2f' % (dict[key]), color)

    # ============================================================#
    # ============================================================#
    def get_randperm(self, x):
        if x.size(0) > 2:
            rand_idx = to_var(torch.randperm(x.size(0)))
        elif x.size(0) == 2:
            rand_idx = to_var(torch.LongTensor([1, 0]))
        else:
            rand_idx = to_var(torch.LongTensor([0]))
        return rand_idx

    # ============================================================#
    # ============================================================#
    def debug_vars(self):
        opt = torch.no_grad() if int(
            torch.__version__.split('.')[1]) > 3 else open(
                '/tmp/null.txt', 'w')
        with opt:
            fixed_x = []
            fixed_label = []
            for i, (images, labels, files) in enumerate(self.data_loader):
                fixed_x.append(images)
                fixed_label.append(labels)
                if i == max(1, int(16 / self.config.batch_size)):
                    break
            fixed_x = torch.cat(fixed_x, dim=0)
            fixed_label = torch.cat(fixed_label, dim=0)
            if not self.config.Deterministic:
                fixed_style = self.G.random_style(fixed_x)
            else:
                fixed_style = None
        return fixed_x, fixed_label, fixed_style

    # ============================================================#
    # ============================================================#
    def _GAN_LOSS(self, real_x, fake_x, label, is_fake=False):
        cross_entropy = self.config.dataset_fake in [
            'painters_14', 'Animals', 'Image2Weather', 'Image2Season',
            'Image2Edges', 'Yosemite', 'RafD,'
        ]
        if cross_entropy:
            label = torch.max(label, dim=1)[1]
        return _GAN_LOSS(
            self.D,
            real_x,
            fake_x,
            label,
            is_fake=is_fake,
            cross_entropy=cross_entropy)

    # ============================================================#
    # ============================================================#
    def INFO(self, epoch, iter):
        # PRINT log info
        if (iter + 1) % self.config.log_step == 0 or iter + epoch == 0:
            self.progress_bar.set_postfix(**self.loss)
        if (iter + 1) == len(self.data_loader):
            self.progress_bar.set_postfix('')

    # ============================================================#
    # ============================================================#
    def Decay_lr(self):
        self.g_lr = self.g_lr / 10.
        self.d_lr = self.d_lr / 10.
        self.update_lr(self.g_lr, self.d_lr)
        self.PRINT('Decay learning rate to g_lr: {}, d_lr: {}.'.format(
            self.g_lr, self.d_lr))

    # ============================================================#
    # ============================================================#
    def RESUME_INFO(self):
        start = int(self.config.pretrained_model.split('_')[0]) + 1
        total_iter = start * int(self.config.pretrained_model.split('_')[1])
        for e in range(start):
            if e != 0 and e % self.config.num_epochs_decay == 0:
                self.Decay_lr()
        return start, total_iter

    # ============================================================#
    # ============================================================#
    def MISC(self, epoch, iter):
        if epoch % self.config.save_epoch == 0:
            # Save Weights
            self.save(epoch, iter + 1)

            # Save Translation
            if not self.config.Deterministic:
                self.generate_SMIT(
                    self.fixed_x,
                    self.output_sample(epoch, iter + 1),
                    Multimodal=1,
                    label=self.fixed_label,
                    training=True,
                    fixed_style=self.fixed_style)
                self.generate_SMIT(
                    self.fixed_x,
                    self.output_sample(epoch, iter + 1),
                    Multimodal=1,
                    label=self.fixed_label,
                    training=True)
            self.generate_SMIT(
                self.fixed_x,
                self.output_sample(epoch, iter + 1),
                label=self.fixed_label,
                training=True)

            # Debug INFO
            elapsed = time.time() - self.start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))
            log = '-> %s | Elapsed [Iter: %d] (%d/%d) : %s | %s\nTrain' % (
                TimeNow(), self.total_iter, epoch, self.config.num_epochs,
                elapsed, self.Log)
            for tag, value in sorted(self.LOSS.items()):
                log += ", {}: {:.4f}".format(tag, np.array(value).mean())
            self.PRINT(log)
            self.PLOT(epoch)

        # Decay learning rate
        if epoch != 0 and epoch % self.config.num_epochs_decay == 0:
            self.Decay_lr()

    # ============================================================#
    # ============================================================#

    def Disc_update(self, real_x0, real_c0):
        rand_idx0 = self.get_randperm(real_c0)
        fake_c0 = real_c0[rand_idx0]
        fake_c0 = to_var(fake_c0.data)
        style_fake0 = to_var(self.G.random_style(real_x0))
        fake_x0 = self.G(real_x0, fake_c0, style_fake0)[0]

        # ============================================================#
        # ======================= Train D ============================#
        # ============================================================#
        d_loss_src, d_loss_cls = self._GAN_LOSS(real_x0, fake_x0, real_c0)
        d_loss_cls = self.config.lambda_cls * d_loss_cls

        self.loss['Dsrc'] = get_loss_value(d_loss_src)
        self.loss['Dcls'] = get_loss_value(d_loss_cls)
        self.update_loss('Dsrc', self.loss['Dsrc'])
        self.update_loss('Dcls', self.loss['Dcls'])

        # Backward + Optimize
        d_loss = d_loss_src + d_loss_cls

        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

    # ============================================================#
    # ============================================================#
    def run(self):
        # lr cache for decaying
        self.g_lr = self.config.g_lr
        self.d_lr = self.config.d_lr
        self.PRINT('Training with learning rate g_lr: {}, d_lr: {}.'.format(
            self.g_optimizer.param_groups[0]['lr'],
            self.d_optimizer.param_groups[0]['lr']))

        # Start with trained info if exists
        if self.config.pretrained_model:
            start, self.total_iter = self.RESUME_INFO()
        else:
            start = 0
            self.total_iter = 0

        # Fixed inputs, target domain labels, and style for debugging
        self.fixed_x, self.fixed_label, self.fixed_style = self.debug_vars()
        if start == 0:
            if not self.config.Deterministic:
                self.generate_SMIT(
                    self.fixed_x,
                    self.output_sample(0, 0),
                    Multimodal=1,
                    label=self.fixed_label,
                    training=True,
                    fixed_style=self.fixed_style)
            self.generate_SMIT(
                self.fixed_x,
                self.output_sample(0, 0),
                label=self.fixed_label,
                training=True)

        self.PRINT("Current time: " + TimeNow())
        self.PRINT("Debug Log txt: " + os.path.realpath(self.config.log.name))

        # Log info
        # RaGAN uses different data for Dis and Gen
        self.Log = self.PRINT_LOG(self.config.batch_size // 2)
        self.start_time = time.time()

        criterion_l1 = torch.nn.L1Loss()

        # Start training
        for epoch in range(start, self.config.num_epochs):
            self.D.train()
            self.G.train()
            self.LOSS = {}
            desc_bar = '[Iter: %d] Epoch: %d/%d' % (self.total_iter, epoch,
                                                    self.config.num_epochs)
            self.progress_bar = tqdm(
                enumerate(self.data_loader),
                unit_scale=True,
                total=len(self.data_loader),
                desc=desc_bar,
                ncols=5)
            for _iter, (real_x, real_c, files) in self.progress_bar:
                self.loss = {}
                self.total_iter += 1

                # RaGAN uses different data for Dis and Gen
                try:

                    def split(x):
                        return (x[:x.size(0) // 2], x[x.size(0) // 2:])

                    real_x0, real_x1 = split(real_x)
                    real_c0, real_c1 = split(real_c)
                except ValueError:

                    def split(x):
                        return (x, x)

                    real_x0, real_x1 = split(real_x)
                    real_c0, real_c1 = split(real_c)

                # ============================================================#
                # ========================= DATA2VAR =========================#
                # ============================================================#
                # Convert tensor to variable
                real_x0 = to_var(real_x0)
                real_c0 = to_var(real_c0)

                real_x1 = to_var(real_x1)
                real_c1 = to_var(real_c1)

                rand_idx1 = self.get_randperm(real_c1)
                fake_c1 = real_c1[rand_idx1]
                fake_c1 = to_var(fake_c1.data)

                self.Disc_update(real_x0, real_c0)

                # ============================================================#
                # ======================== Train G ===========================#
                # ============================================================#
                if (_iter + 1) % self.config.d_train_repeat == 0:

                    style_fake1 = to_var(self.G.random_style(real_x1))
                    style_rec1 = to_var(self.G.random_style(real_x1))
                    fake_x1 = self.G(real_x1, fake_c1, style_fake1)

                    g_loss_src, g_loss_cls = self._GAN_LOSS(
                        fake_x1[0], real_x1, fake_c1, is_fake=True)

                    g_loss_cls = g_loss_cls * self.config.lambda_cls
                    self.loss['Gsrc'] = get_loss_value(g_loss_src)
                    self.loss['Gcls'] = get_loss_value(g_loss_cls)
                    self.update_loss('Gsrc', self.loss['Gsrc'])
                    self.update_loss('Gcls', self.loss['Gcls'])

                    # REC LOSS
                    rec_x1 = self.G(fake_x1[0], real_c1, style_rec1)

                    g_loss_rec = self.config.lambda_rec * criterion_l1(
                        rec_x1[0], real_x1)
                    self.loss['Grec'] = get_loss_value(g_loss_rec)
                    self.update_loss('Grec', self.loss['Grec'])

                    g_loss = g_loss_src + g_loss_rec + g_loss_cls

                    # ========== Attention Part ==========#
                    g_loss_mask = self.config.lambda_mask * (
                        torch.mean(rec_x1[1]) + torch.mean(fake_x1[1]))
                    g_loss_mask_smooth = self.config.lambda_mask_smooth * (
                        _compute_loss_smooth(rec_x1[1]) + _compute_loss_smooth(
                            fake_x1[1]))
                    self.loss['Gatm'] = get_loss_value(g_loss_mask)
                    self.loss['Gats'] = get_loss_value(g_loss_mask_smooth)
                    self.update_loss('Gatm', self.loss['Gatm'])
                    self.update_loss('Gats', self.loss['Gats'])
                    self.color(self.loss, 'Gatm', 'blue')
                    g_loss += g_loss_mask + g_loss_mask_smooth

                    # ========== Identity Part ==========#
                    if self.config.Identity:
                        style_identity = to_var(self.G.random_style(real_x1))
                        idt_x1 = self.G(real_x1, real_c1, style_identity)
                        g_loss_idt = self.config.lambda_idt * criterion_l1(
                            idt_x1[0], real_x1)
                        self.loss['Gidt'] = get_loss_value(g_loss_idt)
                        self.update_loss('Gidt', self.loss['Gidt'])
                        g_loss += g_loss_idt

                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                # ====================== DEBUG =====================#
                self.INFO(epoch, _iter)

            # ============================================================#
            # ======================= MISCELANEOUS =======================#
            # ============================================================#
            # Shuffling dataset each epoch
            self.data_loader.dataset.shuffle(epoch)
            self.MISC(epoch, _iter)
