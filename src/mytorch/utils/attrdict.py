import types
from ast import literal_eval

import numpy as np
import yaml


class AttrDict(dict):

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            if isinstance(self[name], type):
                return self[name]
            elif callable(self[name]):
                return self[name]()  # To support generatable configuration
            else:
                return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value

    def merge(self, other, stack=None):
        """Merge config dictionary other into config dictionary b, clobbering the
        options in b whenever they are also specified in a. update_cfg()
        """
        _merge_a_into_b(other, self, stack)

    def dump(self, fp):

        def representer(dumper, data):
            dicted_data = {}
            for k, v in data.items():
                if isinstance(v, type):
                    dicted_data[k] = '{}.{}'.format(v.__module__, v.__name__)
                elif callable(v):
                    dicted_data[k] = v()
                else:
                    dicted_data[k] = v
            node = dumper.represent_data(dicted_data)
            return node

        yaml.add_representer(AttrDict, representer)
        yaml.dump(self, fp)


def _merge_a_into_b(a, b, stack=None):
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        #TODO(jihoonl): This is good sanity check. Please re-enable back
        #if k not in b:
        #    raise KeyError('Non-existent config key: {}'.format(full_key))

        v = _decode_cfg_value(v_)
        #v = _check_and_coerce_cfg_value_type(v, b[k], k,
        #                                     full_key)  # TODO(jihoonl): WTF

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    if value_b is None:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key))
    return value_a
