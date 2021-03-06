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

        de_params = self.get_num_de_params(self.generator)
        self.Domain_Embedding = DE(
            config, in_dim, de_params, train=False, debug=debug)
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

    def forward(self, image, domain, style, DE=None):
        self.apply_style(image, domain, style, DE=DE)
        return self.generator(image)

    def random_style(self, x, seed=None):
        if isinstance(x, int):
            number = x
        else:
            number = x.size(0)
        if seed is not None:
            torch.manual_seed(seed)
        z = torch.randn(number, self.style_dim)
        return z

    def preprocess(self, label, style):
        label = label.view(label.size(0), -1)
        style = style.view(style.size(0), -1)
        input_de = torch.cat([style, label], dim=-1)
        return input_de

    def apply_style(self, image, label, style, DE=None):
        if DE is None:
            input_de = self.preprocess(label, style)
            de_params = self.Domain_Embedding(input_de)
        else:
            de_params = DE
        self.assign_de_params(de_params, self.generator)

    def assign_de_params(self, de_params, model):
        # assign the de_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = de_params[:, :m.num_features]
                std = de_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if de_params.size(1) > 2 * m.num_features:
                    de_params = de_params[:, 2 * m.num_features:]

    def get_num_de_params(self, model):
        # return the number of DE parameters needed by the model
        num_de_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_de_params += 2 * m.num_features
        return num_de_params
