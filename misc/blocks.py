import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================================#
# Normalization layers
# ==================================================================#
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, \
            "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight,
                           self.bias, True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


# ==================================================================#
# ==================================================================#
class ResidualBlock(nn.Module):
    """Residual Block."""

    def __init__(self, dim_in, dim_out, AdaIn=False):
        super(ResidualBlock, self).__init__()
        if AdaIn:
            norm1 = AdaptiveInstanceNorm2d(dim_out)
            norm2 = AdaptiveInstanceNorm2d(dim_out)
        else:
            norm1 = nn.InstanceNorm2d(dim_out, affine=True)
            norm2 = nn.InstanceNorm2d(dim_out, affine=True)

        self.main = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False), norm1, nn.ReLU(inplace=True),
            nn.Conv2d(
                dim_out,
                dim_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False), norm2)

    def forward(self, x):
        return x + self.main(x)


# ==================================================================#
# ==================================================================#
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
