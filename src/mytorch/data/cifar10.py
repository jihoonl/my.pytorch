from torchvision import datasets, transforms

from ..utils.logger import logger

dataset = {'train': None, 'test': None}


def cifar10(cfg):
    global dataset

    datapath = cfg.PATH
    logger.debug('Loading cifar10 into {}'.format(datapath))

    train_transform = transforms.Compose([
        transforms.RandomCrop(cfg.IN_SIZE[:2], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616))
    ])
    dataset['train'] = datasets.CIFAR10(datapath,
                                        train=True,
                                        download=True,
                                        transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616))
    ])

    dataset['test'] = datasets.CIFAR10(datapath,
                                       train=False,
                                       download=True,
                                       transform=test_transform)
    return dataset
