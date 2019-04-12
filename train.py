from mpi4py import MPI
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
from misc.utils import horovod
hvd = horovod()
comm = MPI.COMM_WORLD
warnings.filterwarnings('ignore')


class Train(Solver):
    def __init__(self, config, data_loader):
        super(Train, self).__init__(config, data_loader)
        self.count_seed = 0
        self.step_seed = 4  # 1 disc - 3 gen
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
        fixed_style = self.random_style(fixed_x, seed=self.count_seed)

        if start == 0:
            self.generate_SMIT(
                fixed_x,
                self.output_sample(0, 0),
                Multimodal=1,
                label=fixed_label,
                training=True,
                fixed_style=fixed_style)
            if self.config.image_size == 256:
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
            'Image2Edges', 'Yosemite', 'RafD', 'BP4D_idt'
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
                color(self.loss, 'Gatm', 'blue')
                self.progress_bar.set_postfix(**self.loss)
            if (iter + 1) == len(self.data_loader):
                self.progress_bar.set_postfix('')

    # ============================================================#
    # ============================================================#
    def Decay_lr(self, current_epoch=0):
        self.d_lr -= (
            self.config.d_lr /
            float(self.config.num_epochs - self.config.num_epochs_decay))
        self.g_lr -= (
            self.config.g_lr /
            float(self.config.num_epochs - self.config.num_epochs_decay))
        self.update_lr(self.g_lr, self.d_lr)
        if self.verbose and current_epoch % self.config.save_epoch == 0:
            self.PRINT('Decay learning rate to g_lr: {}, d_lr: {}.'.format(
                self.g_lr, self.d_lr))

    # ============================================================#
    # ============================================================#
    def RESUME_INFO(self):
        if not self.config.pretrained_model:
            return 0, 0
        start = int(self.config.pretrained_model.split('_')[0]) + 1
        total_iter = start * int(self.config.pretrained_model.split('_')[1])
        self.count_seed = start * total_iter * self.step_seed
        for e in range(start):
            if e > self.config.num_epochs_decay:
                self.Decay_lr(e)
        return start, total_iter

    # ============================================================#
    # ============================================================#
    def MISC(self, epoch, iter):
        if epoch % self.config.save_epoch == 0 and self.verbose:
            # Save Weights
            self.save(epoch, iter + 1)

            # Save Translation
            self.generate_SMIT(
                self.fixed_x,
                self.output_sample(epoch, iter + 1),
                Multimodal=1,
                label=self.fixed_label,
                training=True,
                fixed_style=self.fixed_style)
            if self.config.image_size == 256:
                self.generate_SMIT(
                    self.fixed_x,
                    self.output_sample(epoch, iter + 1),
                    Multimodal=1,
                    label=self.fixed_label,
                    training=True)
            if self.config.image_size == 256:
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
        if epoch > self.config.num_epochs_decay:
            self.Decay_lr(epoch)

    # ============================================================#
    # ============================================================#
    def reset_losses(self):
        return {}

    # ============================================================#
    # ============================================================#
    def current_losses(self, mode, **kwargs):
        loss = 0
        for key, _ in kwargs.items():
            if mode in key:
                loss += self.loss[key]
                self.update_loss(key, get_loss_value(self.loss[key]))
        return loss

    # ============================================================#
    # ============================================================#
    def to_var(self, *args):
        vars = []
        for arg in args:
            vars.append(to_var(arg))
        return vars

    # ============================================================#
    # ============================================================#
    def train_model(self, generator=False, discriminator=False):
        if torch.cuda.device_count() > 1 and hvd.size() == 1:
            G = self.G.module
        else:
            G = self.G
        for p in G.generator.parameters():
            try:
                p.requires_grad_(generator)
            except AttributeError:
                p.requires_grad = generator
        for p in self.D.parameters():
            try:
                p.requires_grad_(discriminator)
            except AttributeError:
                p.requires_grad = discriminator

    # ============================================================#
    # ============================================================#

    def Dis_update(self, real_x, real_c, fake_c):
        self.train_model(discriminator=True)
        real_x, real_c, fake_c = self.to_var(real_x, real_c, fake_c)
        style_fake = to_var(self.random_style(real_x, seed=self.count_seed))
        self.count_seed += 1
        fake_x = self.G(real_x, fake_c, style_fake)[0]
        d_loss_src, d_loss_cls = self._GAN_LOSS(real_x, fake_x, real_c)

        self.loss['Dsrc'] = d_loss_src
        self.loss['Dcls'] = d_loss_cls * self.config.lambda_cls
        d_loss = self.current_losses('D', **self.loss)
        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

    # ============================================================#
    # ============================================================#
    def Gen_update(self, real_x, real_c, fake_c):
        self.train_model(generator=True)
        real_x, real_c, fake_c = self.to_var(real_x, real_c, fake_c)
        criterion_l1 = torch.nn.L1Loss()
        style_fake = to_var(self.random_style(real_x, seed=self.count_seed))
        style_rec = to_var(self.random_style(real_x, seed=self.count_seed + 1))
        style_identity = to_var(
            self.random_style(real_x, seed=self.count_seed + 2))
        self.count_seed += 3

        fake_x = self.G(real_x, fake_c, style_fake)

        g_loss_src, g_loss_cls = self._GAN_LOSS(fake_x[0], real_x, fake_c)
        self.loss['Gsrc'] = g_loss_src
        self.loss['Gcls'] = g_loss_cls * self.config.lambda_cls

        # REC LOSS
        rec_x = self.G(fake_x[0], real_c, style_rec)
        g_loss_rec = criterion_l1(rec_x[0], real_x)
        self.loss['Grec'] = self.config.lambda_rec * g_loss_rec

        # ========== Attention Part ==========#
        self.loss['Gatm'] = self.config.lambda_mask * (
            torch.mean(rec_x[1]) + torch.mean(fake_x[1]))
        self.loss['Gats'] = self.config.lambda_mask_smooth * (
            _compute_loss_smooth(rec_x[1]) + _compute_loss_smooth(fake_x[1]))

        # ========== Identity Part ==========#
        if self.config.Identity:
            idt_x = self.G(real_x, real_c, style_identity)[0]
            g_loss_idt = criterion_l1(idt_x, real_x)
            self.loss['Gidt'] = self.config.lambda_idt * \
                g_loss_idt

        g_loss = self.current_losses('G', **self.loss)
        self.reset_grad()
        g_loss.backward()
        self.g_optimizer.step()

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
        start, self.total_iter = self.RESUME_INFO()

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
            epoch_verbose = (epoch % self.config.save_epoch) and epoch != 0
            self.progress_bar = tqdm(
                enumerate(self.data_loader),
                unit_scale=True,
                total=len(self.data_loader),
                desc=desc_bar,
                disable=not self.verbose or epoch_verbose,
                ncols=5)
            for _iter, (real_x, real_c, files) in self.progress_bar:
                self.loss = self.reset_losses()
                self.total_iter += 1 * hvd.size()
                # RaGAN uses different data for Dis and Gen
                real_x0, real_x1 = split(real_x)
                real_c0, real_c1 = split(real_c)
                fake_c = get_fake(real_c, seed=_iter)
                fake_c0, fake_c1 = split(fake_c)
                # files0, files1 = split(files)
                # import ipdb; ipdb.set_trace()

                # ============================================================#
                # ======================== Train D ===========================#
                # ============================================================#
                self.Dis_update(real_x0, real_c0, fake_c0)

                # ============================================================#
                # ======================== Train G ===========================#
                # ============================================================#
                self.Gen_update(real_x1, real_c1, fake_c1)

                # ====================== DEBUG =====================#
                self.INFO(epoch, _iter)

            # ============================================================#
            # ======================= MISCELANEOUS =======================#
            # ============================================================#
            # Shuffling dataset each epoch
            self.data_loader.dataset.shuffle(epoch)
            self.MISC(epoch, _iter)
