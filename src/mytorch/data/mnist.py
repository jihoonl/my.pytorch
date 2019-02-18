import os
from torchvision import datasets, transforms

from ..utils.logger import logger
from ..utils.config import cfg

dataset = {'train': None, 'test': None}

preprocess = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (1.0,))])


def mnist(path=None):
    global dataset

    if not path:
        datapath = cfg.DATASET.PATH
    else:
        datapath = path
    logger.debug('Loading MNIST into {}'.format(datapath))

    dataset['train'] = datasets.MNIST(
        datapath, train=True, download=True, transform=preprocess)

    dataset['test'] = datasets.MNIST(
        datapath, train=False, download=True, transform=preprocess)
    return dataset
