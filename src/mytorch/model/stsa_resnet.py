from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from ..utils import modules as M
from ..utils.importer import import_class


class AttentionCell(nn.Module):
    """
    y_{ij} = \sum_{a,b \in\mathcal{N}}{softmax_{ab}(q_{ij}^{\top}k_{ab} + q_{ij}^{\top}r_{a-i,b-j})v_{ab}}
    """

    def __init__(self,
                 inplanes,
                 outplanes,
                 kernel_size=7,
                 stride=1,
                 padding=3,
                 groups=8):
        super(AttentionCell, self).__init__()
        self.outplanes = outplanes
        self.inplanes = inplanes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert outplanes % groups == 0, 'outplane must be a multiple of groups. {}/{}'.format(
            outplanes, groups)

        self.query = nn.Conv2d(inplanes,
                               outplanes,
                               kernel_size=1,
                               groups=groups,
                               bias=False)
        self.key = nn.Conv2d(inplanes,
                             outplanes,
                             kernel_size=1,
                             groups=groups,
                             bias=True)
        self.value = nn.Conv2d(inplanes,
                               outplanes,
                               kernel_size=1,
                               groups=groups,
                               bias=True)

        self.rel_size = (outplanes // groups) // 2
        self.rel_x = nn.Parameter(
            torch.randn(self.rel_size, self.kernel_size, 1))
        self.rel_y = nn.Parameter(
            torch.randn(self.rel_size, 1, self.kernel_size))

    def forward(self, x):
        """
        y_{ij} = \sum_{a,b \in\mathcal{N}}{softmax_{ab}(q_{ij}^{\top}k_{ab} + q_{ij}^{\top}r_{a-i,b-j})v_{ab}}
        """

        b, d, h, w = x.shape
        q = self.query(x)  # b, d, h, w
        x_pad = nn.functional.pad(
            x,  # b, d, h+3, w+3
            [self.padding, self.padding, self.padding, self.padding])
        k = self.key(x_pad)  # b, d, h+3, w+3
        v = self.value(x_pad)  # b, d, h+3, w+3

        # k_ab =  b, d, h+3, w+3 ->  b, d, h, w, kernel_h, kernel_w
        k = k.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        # k = batch, # groups, group_size, height, width, kernel_size^2
        k = k.contiguous().view(b, self.groups, self.outplanes // self.groups,
                                h, w, -1)
        # q = batch, # groups, group_size, height, width, 1
        q = q.contiguous().view(b, self.groups, self.outplanes // self.groups,
                                h, w, 1)

        # rel_x : d/2, k, 1
        # rel_y : d/2, 1, k
        # q_x : b, d/2, h, w
        # q_x * rel_x = b, d/2, h, w, k^2
        # qr = (q_x, q_y)
        # b, d, h, w => (b, d/2, h, w) * 2
        # q_{ij}^T * r_{a-i,b-j}
        q_x, q_y = q.split(self.rel_size, dim=2)
        rel_x = self.rel_x.expand(-1, self.kernel_size,
                                  self.kernel_size).contiguous().view(
                                      self.rel_size, -1)
        rel_y = self.rel_y.expand(-1, self.kernel_size,
                                  self.kernel_size).contiguous().view(
                                      self.rel_size, -1)
        qr_x = torch.einsum('bgdhwt,dk->bgdhwk', q_x, rel_x)
        qr_y = torch.einsum('bgdhwt,dk->bgdhwk', q_y, rel_y)
        qr = torch.cat((qr_x, qr_y), dim=2)
        # r = batch, # groups, group_size, height, width, kernel_size^2

        # q_{ij}^T * k_{ab}
        soft = q * k + qr
        soft = F.softmax(soft, dim=-1)

        # out: b, d, h, w, k^2
        # v  : b, d, h, w, k^2
        v = v.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        # k = batch, # groups, group_size, height, width, kernel_size^2
        v = v.contiguous().view(b, self.groups, self.outplanes // self.groups,
                                h, w, -1)
        out = torch.einsum('bgdhwk,bgdhwk->bgdhw', soft, v).view(b, -1, h, w)
        return out


def conv1x1(inplanes, outplanes, stride=1):
    return nn.Conv2d(inplanes,
                     outplanes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = AttentionCell(planes,
                                   planes,
                                   kernel_size=7,
                                   padding=3,
                                   groups=8)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        if self.stride >= 2:
            self.avg_pool = nn.AvgPool2d(self.stride, self.stride)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #if self.stride >= 2:
        #    out = self.avg_pool(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        """
        print('before', out.shape)
        if self.stride >= 2:
            out = F.avg_pool2d(out, (self.stride, self.stride))
        print('after', out.shape)
        """

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttnResnet(nn.Module):

    def __init__(self, param):
        """
        param.INSIZE = image size tuple
        param.OUTSIZE = # of class
        param.EXTRA:
            [
              {
                FIRST: (inchannel, outchannel),
                LAYER: [(inchannel, repeat, stride),
                         ...
                       ]
              }
            ]
        """
        super(AttnResnet, self).__init__()

        layers = param.LAYER
        self.zero_init_residual = param.zero_init_residual
        self.inplane = layers[0][0]
        w, h, first = param.IN_SIZE
        size = (w, h)
        outsize = param.OUT_SIZE
        block = param.BLOCK
        if isinstance(block, str):
            block = import_class(block)

        self.stem = self._make_stem_cell(size, first, layers[0][0])

        seq_layers = [
            self._make_layer(block, channel, layer, stride)
            for channel, layer, stride in layers
        ]
        self.layers = nn.Sequential(*seq_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(layers[-1][0] * block.expansion, outsize)

    def _make_stem_cell(self, size, first, second):
        size, conv = M.Conv2d(size,
                              in_channels=first,
                              out_channels=second,
                              kernel_size=7,
                              stride=2,
                              padding=3)
        size, bn = M.BatchNorm2d(size, second)
        size, relu = M.ReLU(size, inplace=True)
        size, maxpool2d = M.MaxPool2d(size, kernel_size=3, stride=2, padding=1)
        return nn.Sequential(conv, bn, relu, maxpool2d)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplane != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplane, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplane, planes, stride, downsample))
        self.inplane = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplane, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.layers(out)
        out = self.avgpool(out)
        out = out.view(x.size(0), -1)
        out = self.linear(out)
        return out


def AttnResnet18(param):
    layers = [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)]
    param.BLOCK = BottleNeck
    param.LAYER = layers
    param.zero_init_residual = True
    return AttnResnet(param)


def AttnResnet26(param):
    layers = [(64, 1, 1), (128, 2, 2), (256, 4, 2), (512, 1, 2)]
    param.BLOCK = BottleNeck
    param.LAYER = layers
    param.zero_init_residual = True
    return AttnResnet(param)


def AttnResnet34(param):
    layers = [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]
    param.BLOCK = BottleNeck
    param.LAYER = layers
    param.zero_init_residual = True
    return AttnResnet(param)


def AttnResnet50(param):
    layers = [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]
    param.BLOCK = BottleNeck
    param.LAYER = layers
    param.zero_init_residual = True
    return AttnResnet(param)


def AttnResnet101(param):
    layers = [(64, 3, 1), (128, 4, 2), (256, 23, 2), (512, 3, 2)]
    param.BLOCK = BottleNeck
    param.LAYER = layers
    param.zero_init_residual = True
    return AttnResnet(param)


def AttnResnet152(param):
    layers = [(64, 3, 1), (128, 4, 2), (256, 23, 2), (512, 3, 2)]
    param.BLOCK = BottleNeck
    param.LAYER = layers
    param.zero_init_residual = True
    return AttnResnet(param)
