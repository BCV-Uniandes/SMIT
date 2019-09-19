import torch
import torch.nn as nn
from misc.utils import PRINT, to_var
from models.utils import print_debug as _print_debug
from collections import OrderedDict

# ==================================================================#
# ==================================================================#


class DE(nn.Module):
    def __init__(self, config, input_dim, output_dim, train=False,
                 debug=False):

        super(DE, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.comment = 'Learn' if train else 'Fixed'

        linear = nn.Linear(input_dim, output_dim, bias=True)
        model = [('output', linear)]
        self.model = nn.Sequential(OrderedDict(model))

        if not train:
            for param in self.model.parameters():
                param.requires_grad = False

        if debug:
            self.debug()

    def debug(self):
        feed = to_var(
            torch.ones(1, self.input_dim), volatile=True, no_cuda=True)
        PRINT(self.config.log, '-- DE [*{}]:'.format(self.comment))
        self.print_debug(feed, self.model)

    def print_debug(self, x, v):
        return _print_debug(x, v, file=self.config.log)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
