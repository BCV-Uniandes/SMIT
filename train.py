from solver import Solver
import torch
import os
import time
import warnings
import datetime
import numpy as np
from tqdm import tqdm
from misc.utils import color, get_fake, get_labels, get_loss_value
from misc.utils import split, TimeNow, to_var
from misc.losses import _compute_loss_smooth, _GAN_LOSS
import torch.utils.data.distributed
import horovod.torch as hvd
from mpi4py import MPI
comm = MPI.COMM_WORLD
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
    def debug_vars(self, start):
        fixed_x = []
        fixed_label = []
        for i, (images, labels, _) in enumerate(self.data_loader):
            fixed_x.append(images)
            fixed_label.append(labels)
            if i == max(1, int(16 / self.config.batch_size)):
                break
        fixed_x = torch.cat(fixed_x, dim=0)
        fixed_label = torch.cat(fixed_label, dim=0)
        if not self.config.DETERMINISTIC:
            fixed_style = self.random_style(fixed_x)
        else:
            fixed_style = None

        if start == 0:
            if not self.config.DETERMINISTIC:
                self.generate_SMIT(
                    fixed_x,
                    self.output_sample(0, 0),
                    Multimodal=1,
                    label=fixed_label,
                    training=True,
                    fixed_style=fixed_style)
            self.generate_SMIT(
                fixed_x,
                self.output_sample(0, 0),
                label=fixed_label,
                training=True)

        return fixed_x, fixed_label, fixed_style

    # ============================================================#
    # ============================================================#
    def _GAN_LOSS(self, real_x, fake_x, label):
        cross_entropy = self.config.dataset_fake in [
            'painters_14', 'Animals', 'Image2Weather', 'Image2Season',
            'Image2Edges', 'Yosemite', 'RafD,'
        ]
        if cross_entropy:
            label = torch.max(label, dim=1)[1]
        return _GAN_LOSS(
            self.D, real_x, fake_x, label, cross_entropy=cross_entropy)

    # ============================================================#
    # ============================================================#
    def INFO(self, epoch, iter):
        # PRINT log info
        if self.verbose:
            if (iter + 1) % self.config.log_step == 0 or iter + epoch == 0:
                self.loss = {
                    key: get_loss_value(value)
                    for key, value in self.loss.items()
                }
                if not self.config.NO_ATTENTION:
                    color(self.loss, 'Gatm', 'blue')
                self.progress_bar.set_postfix(**self.loss)
            if (iter + 1) == len(self.data_loader):
                self.progress_bar.set_postfix('')

    # ============================================================#
    # ============================================================#
    def Decay_lr(self):
        self.g_lr = self.g_lr / 10.
        self.d_lr = self.d_lr / 10.
        self.update_lr(self.g_lr, self.d_lr)
        if self.verbose:
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
        if epoch % self.config.save_epoch == 0 and self.verbose:
            # Save Weights
            self.save(epoch, iter + 1)

            # Save Translation
            if not self.config.DETERMINISTIC:
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
            # self.PLOT(epoch)

        comm.Barrier()
        # Decay learning rate
        if epoch != 0 and epoch % self.config.num_epochs_decay == 0:
            self.Decay_lr()

    # ============================================================#
    # ============================================================#
    def reset_losses(self):
        # losses = ['Dsrc', 'Dcls']
        # losses += ['Gsrc', 'Gcls', 'Grec', 'Gatm', 'Gats']
        # if self.config.Identity:
        #     losses += ['Gidt']
        # losses = {key: 0 for key in losses}
        return {}

    # ============================================================#
    # ============================================================#
    def current_losses(self, mode, **kwargs):
        loss = 0
        for key, value in kwargs.items():
            if mode in key:
                loss += self.loss[key]
                self.update_loss(key, get_loss_value(self.loss[key]))
        return loss

    # ============================================================#
    # ============================================================#
    def train_model(self, generator=False, discriminator=False):
        # if hvd.size() > 1:
        # G = self.G.module
        if torch.cuda.device_count() > 1 and hvd.size() == 1:
            G = self.G.module
        else:
            G = self.G
        for p in G.generator.parameters():
            p.requires_grad_(generator)
        for p in self.D.parameters():
            p.requires_grad_(discriminator)

    # ============================================================#
    # ============================================================#

    def Dis_update(self, real_x0, real_c0, fake_c0):
        self.train_model(discriminator=True)
        style_fake0 = to_var(self.random_style(real_x0))
        fake_x0 = self.G(real_x0, fake_c0, style_fake0)[0]
        d_loss_src, d_loss_cls = self._GAN_LOSS(real_x0, fake_x0, real_c0)

        self.loss['Dsrc'] = d_loss_src
        self.loss['Dcls'] = d_loss_cls * self.config.lambda_cls
        d_loss = self.current_losses('D', **self.loss)
        d_loss.backward()

    # ============================================================#
    # ============================================================#
    def Gen_update(self, real_x1, real_c1, fake_c1):
        self.train_model(generator=True)
        criterion_l1 = torch.nn.L1Loss()
        style_fake1 = to_var(self.random_style(real_x1))
        style_rec1 = to_var(self.random_style(real_x1))
        style_identity = to_var(self.random_style(real_x1))

        fake_x1 = self.G(real_x1, fake_c1, style_fake1)

        g_loss_src, g_loss_cls = self._GAN_LOSS(fake_x1[0], real_x1, fake_c1)
        self.loss['Gsrc'] = g_loss_src
        self.loss['Gcls'] = g_loss_cls * self.config.lambda_cls

        # REC LOSS
        rec_x1 = self.G(fake_x1[0], real_c1, style_rec1)
        g_loss_rec = criterion_l1(rec_x1[0], real_x1)
        self.loss['Grec'] = self.config.lambda_rec * g_loss_rec

        # ========== Attention Part ==========#
        if not self.config.NO_ATTENTION:
            self.loss['Gatm'] = self.config.lambda_mask * (
                torch.mean(rec_x1[1]) + torch.mean(fake_x1[1]))
            self.loss['Gats'] = self.config.lambda_mask_smooth * (
                _compute_loss_smooth(rec_x1[1]) + _compute_loss_smooth(
                    fake_x1[1]))
            if self.config.ADJUST_SMOOTH and self.loss['Gats'] > 0.1:
                self.loss['Gats'] *= 10

        # ========== Identity Part ==========#
        if self.config.Identity:
            idt_x1 = self.G(real_x1, real_c1, style_identity)[0]
            g_loss_idt = criterion_l1(idt_x1, real_x1)
            self.loss['Gidt'] = self.config.lambda_idt * \
                g_loss_idt

        # ========== Style Recovery Part ==========#
        if self.config.STYLE_ENCODER:
            style_fake1_rec = self.G.style_encoder(fake_x1[0])
            style_rec1_rec = self.G.style_encoder(rec_x1[0])
            self.loss['Gsty'] = criterion_l1(style_fake1, style_fake1_rec)
            self.loss['Gstyr'] = criterion_l1(style_rec1, style_rec1_rec)

        g_loss = self.current_losses('G', **self.loss)
        g_loss.backward()

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
        self.fixed_x, self.fixed_label, self.fixed_style = self.debug_vars(
            start)

        self.PRINT("Current time: " + TimeNow())
        self.PRINT("Debug Log txt: " + os.path.realpath(self.config.log.name))

        # Log info
        # RaGAN uses different data for Dis and Gen
        self.Log = self.PRINT_LOG(self.config.batch_size // 2)

        self.start_time = time.time()

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
                disable=not self.verbose
                or ((epoch % self.config.save_epoch != 0) and epoch != 0),
                ncols=5)
            for _iter, (real_x, real_c, files) in self.progress_bar:
                self.loss = self.reset_losses()
                self.total_iter += 1 * hvd.size()
                # RaGAN uses different data for Dis and Gen
                real_x0, real_x1 = split(real_x)
                real_c0, real_c1 = split(real_c)
                files0, files1 = split(files)

                # ============================================================#
                # ========================= DATA2VAR =========================#
                # ============================================================#
                real_x0 = to_var(real_x0)
                real_c0 = to_var(real_c0)
                real_x1 = to_var(real_x1)
                real_c1 = to_var(real_c1)
                fake_c0 = get_fake(real_c0)
                fake_c0 = to_var(fake_c0.data)
                fake_c1 = get_fake(real_c1)
                fake_c1 = to_var(fake_c1.data)

                # ============================================================#
                # ======================== Train D ===========================#
                # ============================================================#
                self.reset_grad()
                self.Dis_update(real_x0, real_c0, fake_c0)
                self.d_optimizer.step()

                # ============================================================#
                # ======================== Train G ===========================#
                # ============================================================#
                self.reset_grad()
                self.Gen_update(real_x1, real_c1, fake_c1)
                self.g_optimizer.step()

                # ====================== DEBUG =====================#
                self.INFO(epoch, _iter)

            # ============================================================#
            # ======================= MISCELANEOUS =======================#
            # ============================================================#
            # Shuffling dataset each epoch
            self.data_loader.dataset.shuffle(epoch)
            self.MISC(epoch, _iter)
