

from tensorboardX import SummaryWriter

def get_writer(cfg):
    writer = SummaryWriter(cfg.LOG_DIR)
    return writer
