from torch import nn
import torch.nn.functional as F
import numpy as np
from ..utils import modules as M


class FullyConnectedNet(nn.Module):

    def __init__(self, param):
        super(FullyConnectedNet, self).__init__()

        flatten = np.product(param.IN_SIZE)
        size, self._fc = M.Linear(
            img_size, in_features=flatten, out_features=param.OUT_SIZE)

    def forward(self, x):
        x = self._fc(x)
        return F.log_softmax(x, dim=1)
