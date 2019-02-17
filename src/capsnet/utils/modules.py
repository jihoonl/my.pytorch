from .logger import logger
from torch import nn
from . import calc


def print_input_output(f):
    def wrapper(input_size, *args, **kwargs):
        output_size = calc.output2d(input_size, **kwargs)
        logger.debug("{:10} : ({}) -> ({})".format(f.__name__, input_size,
                                                   output_size))
        return output_size, f(input_size, *args, **kwargs)

    return wrapper


@print_input_output
def Conv2d(input_size, *args, **kwargs):
    return nn.Conv2d(*args, **kwargs)


@print_input_output
def MaxPool2d(input_size, *args, **kwargs):
    return nn.MaxPool2d(*args, **kwargs)


def ReLU(input_size, *args, **kwargs):
    return input_size, nn.ReLU(*args, **kwargs)


def Linear(input_size, *args, **kwargs):
    return input_size, nn.Linear(*args, **kwargs)
