from importlib import import_module

from torch.nn import DataParallel

from ..utils.config import cfg
from ..utils.logger import logger

from .mnist_base import MnistBaseNet


def import_class(fullpath):
    """
    Import a class from a string based class path.
    1. Import the module in fullpath
    2. Load Class from the loaded module

    Args:
        fullpath: string based class path

    Returns:
        Class definition
    """
    module_name, class_name = fullpath.rsplit('.', 1)
    logger.debug('Importing {}.{}..'.format(module_name, class_name))
    module = import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class


def create_model(model_config, multi_gpu=False):
    """
    Instantiate a model from configuration already loaded
    """

    model_class = import_class(model_config.NAME)
    model = model_class(**model_config.PARAM)

    if multi_gpu:
        logger.info('Multi GPU Mode. Wrap the model with DataParallel')
        model = DataParallel(model)
    return model


def get_trainable_params(model, multi_gpu=False):
    m = model.module if multi_gpu else model
    return filter(lambda p: p.requires_grad, model.parameters())
