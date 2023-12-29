import numpy as np
from rsdl.tensors import Tensor


# TODO: implement xavier_initializer, zero_initializer

def _calculate_fan_in_fan_out(tensor: Tensor):
    fan_in = np.size(tensor.data, 1)
    fan_out = np.size(tensor.data, 0)
    return fan_in, fan_out


# we can use this function for calculating gain to use with xavier_initializer or he_initializer
def calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return np.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return np.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def _calculate_correct_fan(tensor, mode_str):
    mode = mode_str.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(f"Mode {mode} not supported, please use one of {valid_modes}")

    fan_in, fan_out = _calculate_fan_in_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def xavier_initializer(shape, gain=1.0):  # xavier normal
    fan_in, fan_out = _calculate_fan_in_fan_out(Tensor(np.ones(shape)))
    std = gain * np.sqrt(2.0 / float(fan_in + fan_out))
    return np.random.normal(0, std, size=shape)


def he_initializer(shape, gain):  # kaiming he normal
    fan = _calculate_correct_fan(Tensor(np.ones(shape)))
    std = gain / np.sqrt(fan)
    return np.random.normal(0, std, size=shape)


def zero_initializer(shape):
    return np.zeros(shape, dtype=np.float64)


def one_initializer(shape):
    return np.ones(shape, dtype=np.float64)


def initializer(shape, mode="xavier"):
    if mode == "xavier":
        return xavier_initializer(shape)
    elif mode == "he":
        return he_initializer(shape)
    elif mode == "zero":
        return zero_initializer(shape)
    elif mode == "one":
        return one_initializer(shape)
    else:
        raise NotImplementedError("Not implemented initializer method")
