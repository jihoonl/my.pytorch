#!/usr/bin/env python

import argparse

from capsnet.utils.logger import set_debug, logger
from capsnet.utils.config import cfg_from_file, cfg

from capsnet.data.mnist import mnist
from capsnet.data import loader


def parse_args():

    parser = argparse.ArgumentParser(description='Train a capsule network')
    parser.add_argument(
        '-v',
        '--verbose',
        help='Verbose output',
        default=True,
        action='store_true')
    parser.add_argument(
        '-c', '--config', help='Configuration yaml file', default='', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.verbose:
        set_debug(True)

    if args.config:
        logger.info('Loading data from {}'.format(args.config))
        cfg_from_file(args.config)

    logger.info("Use GPU      : {}".format(cfg.GPU.USE))
    logger.info("Number of GPU: {}".format(cfg.GPU.NUM))

    data = mnist()

    train_loader = loader.train(data)
    test_loader = loader.test(data)
