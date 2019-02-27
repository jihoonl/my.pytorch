import os
from torchvision import datasets

from ..utils.importer import import_method
from ..utils.logger import logger

dataset = {'train': None, 'test': None}


def mnist(cfg):
    global dataset

    datapath = cfg.PATH
    preprocess = import_method(cfg.PREPROCESS)
    logger.debug('Loading MNIST into {}'.format(datapath))

    dataset['train'] = datasets.MNIST(
        datapath, train=True, download=True, transform=preprocess)

    dataset['test'] = datasets.MNIST(
        datapath, train=False, download=True, transform=preprocess)
    return dataset
