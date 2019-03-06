import torch
import torch.nn as nn
from models.utils import print_debug
from misc.utils import PRINT, to_var
from collections import OrderedDict

# ==================================================================#
# ==================================================================#


class DC(nn.Module):
    def __init__(self, config, input_dim, output_dim, dim, n_blk, debug=False):

        super(DC, self).__init__()
        self.config = config

        use_bias = True

        # self._model = [nn.Linear(input_dim, output_dim, bias=False)]
        # self.model += [nn.ReLU(inplace=True)]

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
        # init_net(self.model, 'normal', 0.02)

        if not self.config.DC_TRAIN:
            for param in self.model.parameters():
                param.requires_grad = False

        if debug:
            self.debug()

    def debug(self):
        feed = to_var(torch.ones(1, input_dim), volatile=True, no_cuda=True)
        PRINT(config.log, '-- DC:')
        self.print_debug(feed, self.model)

    def print_debug(self, x, v):
        from models.utils import print_debug as _print_debug
        return _print_debug(x, v, file=self.config.log)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
