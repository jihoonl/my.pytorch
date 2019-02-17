#!/usr/bin/env python

import argparse

from capsnet.utils.logger import set_debug, logger
from capsnet.utils.config import cfg_from_file, cfg

from capsnet.data.mnist import mnist
from capsnet.data import loader
from capsnet.model.mnist_base import MnistBaseNet
from capsnet.optimizer import get_optimizer

import torch
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


def train(model, dataloader, device, optimizer):
    model.train()
    final_loss = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        final_loss = loss.detach().item()
    return final_loss


def test(model, dataloader, device):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)

    return test_loss, accuracy


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
        train_loss = train(model, train_loader, device, opt)
        test_loss, accuracy = test(model, test_loader, device)
        logger.info('Epoch [{:3}/{:3}] - Train Loss: {:.6f}, '
                    'Test Loss: {:.6f}, '
                    'Accuracy: {:.3f}'.format(e, epoch, train_loss, test_loss,
                                              accuracy))


if __name__ == '__main__':
    main()
