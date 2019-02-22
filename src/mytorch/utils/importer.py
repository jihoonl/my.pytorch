from importlib import import_module

from .logger import logger


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


def import_method(fullpath):
    """
    Import a method from a string based class path.
    1. Import the module in fullpath
    2. Load method from the loaded module

    Args:
        fullpath: string based class path

    Returns:
        Class definition
    """
    module_name, method_name = fullpath.rsplit('.', 1)
    logger.debug('Importing {}.{}..'.format(module_name, method_name))
    module = import_module(module_name)
    method = getattr(module, method_name)
    return method
