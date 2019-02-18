#!/usr/bin/env python

import argparse

from mytorch.utils.logger import set_debug, logger
from mytorch.utils.config import cfg_from_file, cfg
from mytorch.data.mnist import mnist
from mytorch.train import train_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a network')
    parser.add_argument(
        '-v',
        '--verbose',
        help='Verbose output',
        default=True,
        action='store_true')
    parser.add_argument(
        '-c',
        '--cfg',
        help='Configuration yaml file',
        default='',
        type=str,
        dest='config')
    return parser.parse_args()


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

    data = mnist()

    train_model(data)


if __name__ == '__main__':
    main()
