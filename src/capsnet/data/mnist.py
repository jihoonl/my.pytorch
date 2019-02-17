import os
from torchvision import datasets, transforms

from ..utils.logger import logger

dataset = {'train': None, 'test': None}

DATA_ROOT = '/data/public/rw/datasets'
IN_PLACE = '../data/mnist'

preprocess = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (1.0,))])

def is_braincloud():
    return os.path.exists(DATA_ROOT)

def mnist():
    global dataset

    logger.debug('Loading MNIST...')

    is_cloud = is_braincloud()

    if is_cloud:
        logger.debug('- In brain cloud')

    datapath = os.path.join(DATA_ROOT,
                            'mnist') if is_cloud else IN_PLACE
    dataset['train'] = datasets.MNIST(
        datapath, train=True, download=True, transform=preprocess)

    dataset['test'] = datasets.MNIST(
        datapath, train=False, download=True, transform=preprocess)
    return dataset
