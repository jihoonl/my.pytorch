import os

from tensorboardX import SummaryWriter
from .config import get_logdir_name


def get_writer(time):
    dirpath = get_logdir_name(time)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    writer = SummaryWriter(dirpath)
    return writer
