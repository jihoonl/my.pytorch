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
        self._tqdm = tqdm(
            iterator,
            ncols=72,
            bar_format='{desc}{percentage:3.0f}%|{bar}|[{elapsed}]',
            ascii=True)
        self._mode = mode
        self._epoch_msg = '[{:3d}/{:3d}]'.format(epoch, max_epoch)

    def __call__(self):
        return self._tqdm

    def desc(self, msg):
        self._tqdm.set_description('[INFO] {:5s} {:s} {:s} '.format(
            self._mode, self._epoch_msg, msg))
