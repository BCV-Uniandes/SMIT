import torch
import torch.nn as nn
from models.utils import print_debug as _print_debug
from misc.utils import to_var
from models.generator import Generator
from models.domain_controller import DC


# ==================================================================#
# ==================================================================#
class AdaInGEN(nn.Module):
    def __init__(self, config, debug=False):
        super(AdaInGEN, self).__init__()

        dc_dim = config.dc_dim
        self.config = config
        self.color_dim = config.color_dim
        self.image_size = config.image_size
        self.style_dim = config.style_dim
        self.c_dim = config.c_dim
        self.Deterministic = config.Deterministic

        def print_debug(x, v):
            return _print_debug(x, v, file=config.log)

        self.generator = Generator(config, debug=False)
        if self.Deterministic:
            in_dim = self.c_dim
        else:
            in_dim = self.style_dim + self.c_dim
        self.adain_net = DC(
            config,
            in_dim,
            self.get_num_adain_params(self.generator),
            dc_dim,
            3,
            debug=debug)
        if debug:
            self.debug()

        if self.config.STYLE_ENCODER:
            from models.style_encoder import StyleEncoder
            self.style_encoder = StyleEncoder(config, debug=debug)

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

    def random_style(self, x):
        if isinstance(x, int):
            number = x
        else:
            number = x.size(0)
        z = torch.randn(number, self.style_dim)
        return z

    def apply_style(self, image, label, style):
        label = label.view(label.size(0), -1)
        style = style.view(style.size(0), -1)
        if self.Deterministic:
            input_adain = label
        else:
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
