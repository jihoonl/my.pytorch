import os

import torch
import yaml

from .config import cfg, get_logdir_name
from .leaderboard import Leaderboard
from .logger import logger

CHECKPOINT = 'checkpoint.txt'
CONFIG_NAME = 'config.yaml'


def export(model, log, multi_gpu, time, epoch=None, extra=None):
    """
    Exports the current model, checkpoint, and config
    """
    dirpath = get_logdir_name(time)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    filename = '{}.pth'.format('final' if not epoch else str(epoch))

    _export_model(model, multi_gpu, dirpath, filename, epoch)
    _export_checkpoint(dirpath, filename, log)
    _export_config(dirpath, cfg)
    logger.info('Exported to {}'.format(dirpath))

    lbdirpath = cfg.EXP.LEADERBOARD.PATH
    datapath = os.path.relpath(os.path.join(dirpath, filename), lbdirpath)
    if not os.path.exists(lbdirpath):
        os.makedirs(lbdirpath)

    board = Leaderboard(lbdirpath)
    board.add(datapath, log)
    board.save(CONFIG_NAME)


def _export_model(model, multi_gpu, dirpath, filename, epoch):
    filepath = os.path.join(dirpath, filename)

    # Assume it uses data parallel if it is multi gpu
    m = model.module if multi_gpu else model
    torch.save(m.state_dict(), filepath)


def _export_checkpoint(dirpath, filename, log):
    cppath = os.path.join(dirpath, CHECKPOINT)
    if not os.path.exists(cppath):
        with open(cppath, 'w') as f:
            print('#{:>9s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
                'File,', 'Test Acc,', 'Test Loss,', 'Train Acc,',
                'Train Loss,'),
                  file=f)
    with open(os.path.join(cppath), 'a') as f:
        print('{}, {}'.format(filename, log), file=f)


def _export_config(dirpath, config):
    cpath = os.path.join(dirpath, CONFIG_NAME)

    if not os.path.exists(cpath):
        with open(cpath, 'w') as f:
            config.dump(f)
