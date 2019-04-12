import torch
import torch.nn as nn
from misc.utils import to_var
from models.domain_embedding import DE
from models.generator import Generator

# ==================================================================#
# ==================================================================#


class AdaInGEN(nn.Module):
    def __init__(self, config, debug=False):
        super(AdaInGEN, self).__init__()

        self.config = config
        self.color_dim = config.color_dim
        self.image_size = config.image_size
        self.style_dim = config.style_dim
        self.c_dim = config.c_dim
        self.generator = Generator(config, debug=False)
        in_dim = self.style_dim + self.c_dim

        adain_params = self.get_num_adain_params(self.generator)
        self.adain_net = DE(
            config, in_dim, adain_params, train=False, debug=debug)
        if debug:
            self.debug()

    def debug(self):
        feed = to_var(
            torch.ones(1, self.color_dim, self.image_size, self.image_size),
            volatile=True,
            no_cuda=True)
        label = to_var(torch.ones(1, self.c_dim), volatile=True, no_cuda=True)
        style = to_var(self.random_style(feed), volatile=True, no_cuda=True)
        self.apply_style(feed, label, style)
        self.generator.debug()

    def forward(self, x, c, stochastic):
        self.apply_style(x, c, stochastic)
        return self.generator(x)

    def random_style(self, x, seed=None):
        if isinstance(x, int):
            number = x
        else:
            number = x.size(0)
        if seed is not None:
            torch.manual_seed(seed)
        z = torch.randn(number, self.style_dim)
        return z

    def apply_style(self, image, label, style):
        label = label.view(label.size(0), -1)
        style = style.view(style.size(0), -1)
        input_adain = torch.cat([style, label], dim=-1)

        adain_params = self.adain_net(input_adain)
        self.assign_adain_params(adain_params, self.generator)

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params
