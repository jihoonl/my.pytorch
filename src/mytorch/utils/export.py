
import os
import time

from .logger import logger
from .config import cfg


def export(model, log, epoch=None):
    """
    Exports the current model
    """
    current_time = time.strftime("%b%d-%H%M-%S")
    final_path = os.path.join(cfg.EXP.PATH, current_time)
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    logger.info('Exporting to {}'.format(final_path))

    filename = '{}.pth'.format('final' if not epoch else str(epoch))
    filepath = os.path.join(final_path, filename)
