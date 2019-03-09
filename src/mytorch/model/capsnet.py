import torch
import torch.nn.functional as F
from torch import nn

from ..utils import modules as M


class Capsule(nn.Module):

    def __init__(self, in_channels, dimension, kernel_size=9, stride=2):
        super(Capsule, self).__init__()
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=dimension,
            kernel_size=kernel_size,
            stride=stride)

    def forward(self, x):
        out = self._conv(x)  # width, height, dimension
        out = self.squash(out)
        return out

    def squash(self, x):
        """
        Squash defined in Dynamic Routing Between Capsules
        v = ||s||^2 * s / ((1+||s||^2) * s)

        Args:
            x: [batch_size, d, w, h]
        """
        x = x.transpose(1, -1)
        squared_norm = (x**2).sum(-1, keepdim=True)
        output = squared_norm * x / (
            (1. + squared_norm) * torch.sqrt(squared_norm))
        output = output.transpose(1, -1)
        return output


class Conv2CapsuleLayer(nn.Module):

    def __init__(self, in_channels, out_capsules, vector_dim, num_routes=-1):
        super(CapsuleLayer, self).__init__()
        self._out_capsules = out_capsules
        self._num_routes = num_routes

        self._capsules = nn.ModuleList()
        for i in range(self._out_capsules):
            capsule = Capsule(in_channels=in_channels, dimension=vector_dim)
            self._capsules.append(capsule)

    def forward(self, x):
        out = [c(x) for c in self._capsules]


class CapsuleLayer(nn.Module):

    def __init__(self, in_capsules, out_capsules, vector_dim, num_routes=3):
        self._in = in_capsules
        self._out = out_capsules
        self._vectordim = vector_dim
        self._num_routes = num_routes

        #self.W = nn.Parameter(
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        return x


class CapsuleNet(nn.Module):

    def __init__(self, param):
        super(CapsuleNet, self).__init__()

        w, h, d = param.IN_SIZE

        size, self._conv1 = M.Conv2d((w, h),
                                     in_channels=d,
                                     out_channels=256,
                                     kernel_size=9,
                                     stride=1,
                                     bias=True)
        # size,  self._bn1 = M.BatchNorm2d(size, 256)
        size, self._relu1 = M.ReLU(size, inplace=True)

        self._primary_capsule = Conv2CapsuleLayer(
            in_channels=256, out_channels=32, vector_dim=8)
        self._digit_capsule = CapsuleLayer(
            in_capsules=32, out_capsules=10, vector_dim=16)

    def forward(self, x):
        out = self._conv1(x)
        out = self._relu1(out)
        out = self._primary_capsule(out)
        return out
