import time
import os
import numpy as np
from collections import OrderedDict

from .config import cfg


def force_symlink(src, dst):
    try:
        os.symlink(src, dst)
    except FileExistsError as e:
        os.unlink(dst)
        os.symlink(src, dst)


class Leaderboard(object):

    def __init__(self, path, filename='leaderboard.txt'):
        self._dirpath = path
        self._path = os.path.join(path, filename)
        if os.path.exists(self._path):
            self._board = self.load(self._path)
        else:
            self._board = OrderedDict()

    def load(self, path):
        """
        Rank, name, test_acc, test_loss, train_acc, train_loss
        """
        board = OrderedDict()
        with open(path, 'r') as f:
            lines = f.readlines()
            for l in lines[1:]:
                key, value = l.split(':')
                splits = value.split(',')
                board[splits[0].strip()] = [float(s) for s in splits[1:]]
        return board

    def add(self, datapath, log):
        board = self._board
        new_log = [float(l) for l in log.split(',')]
        board[datapath] = new_log

        k = list(board.keys())
        b = np.array(list((board.values())))
        acc_list = b[:, 0]
        sorted_idx = np.argsort(acc_list)[::-1]
        new_board = OrderedDict()
        for idx in sorted_idx:
            new_board[k[idx]] = b[idx]
        self._board = new_board

    def save(self, config_name):
        with open(self._path, 'w') as f:
            print(
                '{:6s}: {:30s}, {}'.format(
                    '#Rank', 'Data',
                    '  Test Acc   Test Loss   Train Acc  Train Loss'),
                file=f)
            for rank, (k, v) in enumerate(self._board.items(), 1):
                print(
                    '   {:3s}: {:30s}, {}'.format(
                        str(rank), k,
                        ', '.join(['{:>10.6f}'.format(float(vv)) for vv in v])),
                    file=f)
                if rank > cfg.EXP.LEADERBOARD.MAX:
                    break
        k, v = next(iter(self._board.items()))
        force_symlink(
            k, os.path.join(self._dirpath, cfg.EXP.LEADERBOARD.TOP1_CHECKPOINT))
        force_symlink(
            os.path.join(os.path.dirname(k), config_name),
            os.path.join(self._dirpath, cfg.EXP.LEADERBOARD.TOP1_CONFIG))
