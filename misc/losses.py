def _compute_loss_smooth(mat):
    import torch
    return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
        torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))


# ==================================================================#
# ==================================================================#
def _CLS_LOSS(output, target, cross_entropy=False):
    import torch.nn.functional as F
    if cross_entropy:
        return F.cross_entropy(output, target)
    else:
        return F.binary_cross_entropy_with_logits(
            output, target, size_average=False) / output.size(0)


# ==================================================================#
# ==================================================================#
def _CLS_L1(output, target):
    import torch.nn.functional as F
    return F.l1_loss(output, target) / output.size(0)


# ==================================================================#
# ==================================================================#
def _CLS_L2(output, target):
    import torch.nn.functional as F
    return F.mse_loss(output, target) / output.size(0)


# ==================================================================#
# ==================================================================#
def _GAN_LOSS(Disc, real_x, fake_x, label, cross_entropy=False):
    import torch
    import torch.nn.functional as F

    src_real, cls_real = Disc(real_x)
    src_fake, _ = Disc(fake_x)

    loss_src = 0
    loss_cls = 0
    for i in range(len(src_real)):
        # Relativistic GAN
        loss_real = torch.mean(
            F.relu(1.0 - (src_real[i] - torch.mean(src_fake[i]))))
        loss_fake = torch.mean(
            F.relu(1.0 + (src_fake[i] - torch.mean(src_real[i]))))
        loss_src += (loss_real + loss_fake) / 2

        loss_cls += _CLS_LOSS(cls_real[i], label, cross_entropy=cross_entropy)

    return loss_src, loss_cls
