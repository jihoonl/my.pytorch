
import os

from tensorboardX import SummaryWriter

def get_writer(time, cfg):
    dirpath = os.path.join(cfg.LOG_DIR, time)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    writer = SummaryWriter(dirpath)
    return writer
