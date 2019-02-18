from torch import nn
import torch.nn.functional as F
from ..utils import modules as M


class MnistBaseNet(nn.Module):

    def __init__(self, img_size):
        super(MnistBaseNet, self).__init__()
        """
        Conv1 -> Relu -> MaxPool2D -> Relu -> MaxPool2D -> FC -> Relu -> FC
        """

        # [28 x 28]
        size, self._conv1 = M.Conv2d(
            img_size,
            in_channels=1,
            out_channels=20,
            kernel_size=5,
            stride=1)
        size, self._relu1 = M.ReLU(size, inplace=True)

        size, self._max_pool2d1 = M.MaxPool2d(size, kernel_size=2, stride=2)
        size, self._conv2 = M.Conv2d(
            size, in_channels=20, out_channels=50, kernel_size=5, stride=1)
        size, self._relu2 = M.ReLU(size, inplace=True)
        size, self._max_pool2d2 = M.MaxPool2d(size, kernel_size=2, stride=2)
        size, self._fc1 = M.Linear(size, in_features=size[0] * size[1] * 50, out_features=500)
        size, self._relu3 = M.ReLU(size, inplace=True)
        size, self._fc2 = M.Linear(size, in_features=500, out_features=10)

    def forward(self, x):
        x = self._conv1(x)
        x = self._relu1(x)
        x = self._max_pool2d1(x)
        x = self._conv2(x)
        x = self._relu2(x)
        x = self._max_pool2d2(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self._fc1(x)
        x = self._relu3(x)
        x = self._fc2(x)
        return F.log_softmax(x, dim=1)