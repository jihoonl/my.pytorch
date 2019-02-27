#!/usr/bin/env python

import argparse

from mytorch.run import inference_from_file
from mytorch.utils.logger import logger, set_debug
from mytorch.utils.config import cfg, cfg_from_file


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument(
        '-v',
        '--verbose',
        help='Verbose output',
        default=False,
        action='store_true')
    parser.add_argument(
        '-c',
        '--cfg',
        help='Configuration yaml file',
        default='/data/private/exp/mnist/leaderboard/config.yaml',
        type=str,
        dest='config')
    parser.add_argument(
        '-pt',
        '--checkpoint',
        help='Checkpoint file',
        default='/data/private/exp/mnist/leaderboard/top.pth',
        type=str,
        dest='checkpoint')
    parser.add_argument(
        '-i',
        '--input',
        help='Inputdata',
        type=str,
        dest='input',
        required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        set_debug(True)

    if args.config:
        logger.info('Loading config from {}'.format(args.config))
        cfg_from_file(args.config)

    logger.info("Use GPU      : {}".format(cfg.GPU.USE))
    logger.info("Number of GPU: {}".format(cfg.GPU.NUM))
    logger.info("Device       : {}".format(cfg.DEVICE))
    logger.info("Input        : {}".format(args.input))
    cfg.RESUME.CHECKPOINT = args.checkpoint

    output = inference_from_file(args.input)
    logger.info(output)


if __name__ == '__main__':
    main()
