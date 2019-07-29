from collections import namedtuple

from torch import nn

from ..utils import modules as M
from ..utils.importer import import_class


def conv3x3(inplanes, outplanes, stride=1):
    return nn.Conv2d(inplanes,
                     outplanes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1(inplanes, outplanes, stride=1):
    return nn.Conv2d(inplanes,
                     outplanes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Resnet(nn.Module):

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
        super(Resnet, self).__init__()

        layers = param.LAYER
        self.zero_init_residual = param.zero_init_residual
        self.inplane = layers[0][0]

        w, h, first = param.IN_SIZE
        size = (w, h)
        outsize = param.OUT_SIZE
        block = param.BLOCK
        if isinstance(block, str):
            block = import_class(block)

        size, self.conv1 = M.Conv2d(size,
                                    in_channels=first,
                                    out_channels=layers[0][0],
                                    kernel_size=7,
                                    stride=2,
                                    padding=3)
        size, self.bn1 = M.BatchNorm2d(size, layers[0][0])
        size, self.relu1 = M.ReLU(size, inplace=True)
        size, self.maxpool2d1 = M.MaxPool2d(size,
                                            kernel_size=3,
                                            stride=2,
                                            padding=1)

        seq_layers = []
        for channel, layer, stride in layers:
            lay = self._make_layer(block, channel, layer, stride)
            seq_layers.append(lay)
        self.layers = nn.Sequential(*seq_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(layers[-1][0] * block.expansion, outsize)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if hasattr(m.weight, 'bias') and m.weight.bias is not None:
                    nn.init.constant_(m.weight.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplane != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplane, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplane, planes, stride, downsample))
        self.inplane = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplane, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool2d1(x)

        x = self.layers(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def Resnet18(param):
    layers = [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)]
    param.BLOCK = BottleNeck
    param.LAYER = layers
    param.zero_init_residual = True
    return Resnet(param)


def Resnet26(param):
    layers = [(64, 1, 1), (128, 2, 2), (256, 4, 2), (512, 1, 2)]
    param.BLOCK = BottleNeck
    param.LAYER = layers
    param.zero_init_residual = True
    return Resnet(param)


def Resnet34(param):
    layers = [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]
    param.BLOCK = BasicBlock
    param.LAYER = layers
    param.zero_init_residual = True
    return Resnet(param)


def Resnet50(param):
    layers = [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]
    param.BLOCK = BottleNeck
    param.LAYER = layers
    param.zero_init_residual = True
    return Resnet(param)


def Resnet101(param):
    layers = [(64, 3, 1), (128, 4, 2), (256, 23, 2), (512, 3, 2)]
    param.BLOCK = BottleNeck
    param.LAYER = layers
    param.zero_init_residual = True
    return Resnet(param)


def Resnet152(param):
    layers = [(64, 3, 1), (128, 4, 2), (256, 23, 2), (512, 3, 2)]
    param.BLOCK = BottleNeck
    param.LAYER = layers
    param.zero_init_residual = True
    return Resnet(param)
