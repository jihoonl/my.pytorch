#!/usr/bin/env python

import argparse

from capsnet.utils.logger import set_debug, logger
from capsnet.utils.config import cfg_from_file, cfg

from capsnet.data.mnist import mnist
from capsnet.data import loader
from capsnet.model.mnist_base import MnistBaseNet
from capsnet.optimizer import get_optimizer

import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument(
        '-v',
        '--verbose',
        help='Verbose output',
        default=True,
        action='store_true')
    parser.add_argument(
        '-c', '--config', help='Configuration yaml file', default='', type=str)
    return parser.parse_args()


def train(model, dataloader, device, optimizer, epoch):
    model.train()
    for idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if idx % cfg.TRAIN.LOG_INTERVAL == 0:
            print('Train Epoch: {:3} [{:6}/{:6} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(data), len(dataloader.dataset),
                100. * idx / len(dataloader), loss.item()))


def main():
    args = parse_args()
    if args.verbose:
        set_debug(True)

    if args.config:
        logger.info('Loading data from {}'.format(args.config))
        cfg_from_file(args.config)

    logger.info("Use GPU      : {}".format(cfg.GPU.USE))
    logger.info("Number of GPU: {}".format(cfg.GPU.NUM))
    logger.info("Device       : {}".format(cfg.DEVICE))

    device = cfg.DEVICE
    epoch = cfg.TRAIN.EPOCH
    data = mnist()
    train_loader = loader.train(data['train'])
    test_loader = loader.test(data['test'])
    model = MnistBaseNet((28, 28))
    model.to(device)
    opt = get_optimizer(model.parameters())

    for e in range(1, epoch + 1):
        train(model, train_loader, device, opt, e)


if __name__ == '__main__':
    main()
