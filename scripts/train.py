#!/usr/bin/env python

import argparse

from capsnet.data.mnist import mnist
from capsnet.utils.logger import set_debug, logger


def parse_args():

    parser = argparse.ArgumentParser(description='Train a capsule network')
    parser.add_argument(
        '-v',
        '--verbose',
        help='Verbose output',
        default=True,
        action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.verbose:
        set_debug(True)
    data = mnist()

    data_loader 
