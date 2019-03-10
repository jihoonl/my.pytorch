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
        # out = self.move_vectordim_to_end(out)
        print('out', out.shape)
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

    def __init__(self, in_channels, out_channels, vector_dim, num_routes=-1):
        super(Conv2CapsuleLayer, self).__init__()
        self._out_channels = out_channels
        self._num_routes = num_routes

        self._capsules = nn.ModuleList()
        for i in range(self._out_channels):
            capsule = Capsule(in_channels=in_channels, dimension=vector_dim)
            self._capsules.append(capsule)

    def forward(self, x):
        out = torch.stack([c(x) for c in self._capsules], dim=2)
        out = self.move_vectordim_to_end(out)
        return out

    def move_vectordim_to_end(self, x):
        return x.permute(0, 2, 3, 4, 1)


class CapsuleLayer(nn.Module):

    def __init__(self, in_channels, out_channels, vector_dim, vector_indim, num_routes=3):
        super(CapsuleLayer, self).__init__()
        self._in = in_channels
        self._out = out_channels
        self._vectordim = vector_dim
        self._num_routes = num_routes

        # W = in layer's # of channel x vector -> out layer's # of channels x vectors
        # W is shared weight within a channel
        # In MNIST Capsnet: 32 x 6 x 6 x 8 -> 32 x 6 x 6 x 16
        self.W = nn.Parameter(torch.randn(in_channels, 6, 6, vector_dim, vector_indim))

    def forward(self, x):
        in_dim = x.shape[-1]
        W = torch.cat([self.W.unsqueeze(0)] * x.shape[0])
        u_hat = torch.matmul(W, x.unsqueeze(-1)).sqeeze(-1)
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
            in_channels=32, out_channels=10, vector_dim=16, vector_indim=8)


    def forward(self, x):
        out = self._conv1(x)
        out = self._relu1(out)
        print('relu',out.shape)
        out = self._primary_capsule(out)
        print('priv', out.shape)
        out = self._digit_capsule(out)
        return out
