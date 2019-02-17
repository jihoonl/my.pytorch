import numpy as np


def calc(v, k, s, p, d):
    return int(np.floor((v + 2 * p - d * (k - 1) - 1) / s + 1))


def output2d(input_size,
             kernel_size=1,
             stride=1,
             padding=0,
             dilation=1,
             *args,
             **kwargs):
    w, h = input_size
    new_w = calc(w, kernel_size, stride, padding, dilation)
    new_h = calc(h, kernel_size, stride, padding, dilation)
    return (new_w, new_h)
