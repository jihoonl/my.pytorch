
from torch.nn import DataParallel

from ..utils.config import cfg
from ..utils.logger import logger
from ..utils.importer import import_class

from .mnist_base import MnistBaseNet


def create_model(model_config, multi_gpu=False):
    """
    Instantiate a model from configuration already loaded
    """

    model_class = import_class(model_config.NAME)
    model = model_class(model_config.PARAM)

    if multi_gpu:
        logger.info('Multi GPU Mode. Wrap the model with DataParallel')
        model = DataParallel(model)
    return model


def get_trainable_params(model, multi_gpu=False):
    m = model.module if multi_gpu else model
    return filter(lambda p: p.requires_grad, model.parameters())
