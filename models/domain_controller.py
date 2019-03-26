import torch
import torch.nn as nn
from misc.utils import PRINT, to_var
# from models.utils import init_net
from collections import OrderedDict

# ==================================================================#
# ==================================================================#


class DC(nn.Module):
    def __init__(self,
                 config,
                 input_dim,
                 output_dim,
                 dim,
                 n_blk,
                 train=False,
                 debug=False):

        super(DC, self).__init__()
        self.config = config
        self.input_dim = input_dim

        self.comment = 'Learn' if train else 'Fixed'

        if not self.config.INIT_DC:
            use_bias = True
            _model = []
            linear = nn.Linear(input_dim, dim, bias=use_bias)
            _model += [('input', linear)]
            _model += [('relu_input', nn.ReLU(inplace=True))]
            for i in range(n_blk - 2):
                linear = nn.Linear(dim, dim, bias=use_bias)
                _model += [('linear_' + str(i), linear)]
                _model += [('relu_' + str(i), nn.ReLU(inplace=True))]
            linear = nn.Linear(dim, output_dim, bias=use_bias)
            _model += [('output', linear)]  # no output activations
            self.model = nn.Sequential(OrderedDict(_model))

        else:
            linear = nn.Linear(input_dim, output_dim, bias=True)
            _model = [('output', linear)]  # no output activations
            self.model = nn.Sequential(OrderedDict(_model))
            # init_net(
            # self.model, init_type='normal', init_gain=0.1, init_bias=-0.1)

        if not train:
            for param in self.model.parameters():
                param.requires_grad = False

        if debug:
            self.debug()

    def debug(self):
        feed = to_var(
            torch.ones(1, self.input_dim), volatile=True, no_cuda=True)
        PRINT(self.config.log, '-- DC [*{}]:'.format(self.comment))
        self.print_debug(feed, self.model)

    def print_debug(self, x, v):
        from models.utils import print_debug as _print_debug
        return _print_debug(x, v, file=self.config.log)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
