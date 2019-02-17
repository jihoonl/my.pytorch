#!/usr/bin/env python

import argparse

from capsnet.utils.logger import set_debug, logger
from capsnet.utils.config import cfg_from_file, cfg
from capsnet.utils.timer import Timer

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
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


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
    opt = get_optimizer(model.parameters())

    if cfg.GPU.MULTI:
        model = torch.nn.DataParallel(model)
    model.to(device)
    logger.info(opt)

    t = Timer()
    for e in range(1, epoch + 1):
        t.tic()
        train(model, train_loader, device, opt)
        elapse = t.toc()
        train_loss, train_accuracy = test(model, train_loader, device)
        test_loss, test_accuracy = test(model, test_loader, device)
        lr = opt.param_groups[0]['lr']
        logger.info('[{:2}/{:2}][{:.3f}s][{:.6f}] - '
                    '(Train, Test) '
                    'Loss: {:.6f} - {:.6f}, '
                    'Acc: {:.4f} - {:.4f}'.format(e, epoch, elapse, lr,
                                                 train_loss, test_loss,
                                                 train_accuracy, test_accuracy))


if __name__ == '__main__':
    main()
