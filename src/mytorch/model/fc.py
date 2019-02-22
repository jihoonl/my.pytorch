import numpy as np
import torch.nn.functional as F
from torch import nn

from ..utils import modules as M


def add_layer(size, in_f, out_f, relu=True):
    size, l = M.Linear(size, in_features=in_f, out_features=out_f)
    if relu:
        size, r = M.ReLU(out_f, inplace=True)
        return size, [l, r]
    return size, [l]


class FullyConnectedNet(nn.Module):

    def __init__(self, param):
        super(FullyConnectedNet, self).__init__()

        flatten = np.product(param.IN_SIZE)

        layer = []
        prev_dim = flatten
        size = param.IN_SIZE

        if param.EXTRA:
            p_layer = param.EXTRA[0]['LAYER']
            for dim in p_layer:
                size, l = add_layer(size, prev_dim, dim)
                prev_dim = dim
                layer.extend(l)
        size, l = add_layer(size, prev_dim, param.OUT_SIZE, False)
        layer.extend(l)
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for layer in self.layer:
            x = layer(x)
        return F.log_softmax(x, dim=1)
