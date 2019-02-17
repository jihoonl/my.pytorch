from torch.utils.data import DataLoader
from ..utils.config import cfg
from ..utils.logger import logger


def train(dataset):
    logger.debug(
        'Train Loader - Batch: {}, pin_memory: {}, Num Worker: {}'.
        format(cfg.TRAIN.BATCH_SIZE, cfg.GPU.USE, cfg.DATASET.NUM_WORKERS))
    return DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        pin_memory=cfg.GPU.USE,
        shuffle=True,
        num_workers=cfg.DATASET.NUM_WORKERS)


def test(dataset):
    logger.debug(
        'Test Loader  - Batch: {}, pin_memory: {}, Num Worker: {}'.
        format(cfg.TRAIN.BATCH_SIZE, cfg.GPU.USE, cfg.DATASET.NUM_WORKERS))

    return DataLoader(
        dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        pin_memory=cfg.GPU.USE,
        shuffle=False,
        num_workers=cfg.DATASET.NUM_WORKERS)
