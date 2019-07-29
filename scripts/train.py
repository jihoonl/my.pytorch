#!/usr/bin/env python

import argparse

from mytorch.run import train_model
from mytorch.utils.config import cfg, cfg_from_file
from mytorch.utils.logger import logger, set_debug


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
    parser.add_argument(
        '-lr',
        '--learning-rate',
        help='learning rate',
        default=0.0,
        type=float,
        dest='learning_rate')
    parser.add_argument(
        '--batch-size',
        help='batch size',
        default=0,
        type=int,
        dest='batch_size')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        set_debug(True)

    if args.config:
        logger.info('Loading config from {}'.format(args.config))
        cfg_from_file(args.config)

        if args.learning_rate > 0.0:
            cfg.TRAIN.OPTIMIZER.PARAM.lr = args.learning_rate
        if args.batch_size > 0:
            cfg.TRAIN.BATCH_SIZE = args.batch_size

    logger.info("Use GPU      : {}".format(cfg.GPU.USE))
    logger.info("Number of GPU: {}".format(cfg.GPU.NUM))
    logger.info("Device       : {}".format(cfg.DEVICE))

    train_model()


if __name__ == '__main__':
    main()
