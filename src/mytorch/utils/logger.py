import logging

from tqdm import tqdm

#formatter = logging.Formatter(
#    '[%(levelname)s] %(asctime)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
formatter = logging.Formatter(
    '[%(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
ch = logging.StreamHandler()
ch.setFormatter(formatter)

logger = logging.getLogger('capsnet')
logger.setLevel(logging.INFO)
logger.addHandler(ch)


def set_debug(debug=True):
    global logger

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


class Progressbar(object):

    def __init__(self, iterator, mode, epoch, max_epoch):
        self._mode = mode
        self._epoch_msg = '[{:3d}/{:3d}]'.format(epoch, max_epoch)
        self._tqdm = tqdm(
            iterator,
            ncols=79,
            bar_format='{desc}{percentage:3.0f}%|{bar}|[{elapsed}, {n_fmt}/{total_fmt}]{postfix}',
            desc='[INFO] {:8s} {:s}'.format(self._mode, self._epoch_msg),
            ascii=True)

    def __call__(self):
        return self._tqdm

    def desc(self, msg):
        self._tqdm.set_postfix(msg)
