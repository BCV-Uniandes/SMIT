import torch
import torch.nn as nn
from models.utils import print_debug, init_net
from misc.utils import PRINT, to_var

# from misc.blocks import LinearBlock

# ==================================================================#
# ==================================================================#


class DC(nn.Module):
    def __init__(self, config, input_dim, output_dim, dim, n_blk, debug=False):

        super(DC, self).__init__()
        self.config = config

        self._model = [nn.Linear(input_dim, output_dim, bias=False)]
        # self.model += [nn.ReLU(inplace=True)]

        # self._model = []
        # self._model += [
        #     LinearBlock(input_dim, dim, norm=norm, activation=activ)
        # ]
        # for i in range(n_blk - 2):
        #     self._model += [LinearBlock(dim, dim, norm=norm,
        #           activation=activ)]
        # self._model += [
        #     LinearBlock(dim, output_dim, norm='none', activation='none')
        # ]  # no output activations
        self.model = nn.Sequential(*self._model)
        # init_net(self.model, 'normal', 0.02)
        init_net(self.model, 'kaiming')

        for param in self.model.parameters():
            param.requires_grad = False

        if debug:
            feed = to_var(
                torch.ones(1, input_dim), volatile=True, no_cuda=True)
            PRINT(config.log, '-- DC:')
            print_debug(feed, self._model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
