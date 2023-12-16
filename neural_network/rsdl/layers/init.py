import numpy as np


# TODO: implement xavier_initializer, zero_initializer

def xavier_initializer(shape):
    return np.random.randn(*shape) * np.sqrt(1/shape[0], dtype=np.float64)

def he_initializer(shape):
    return np.random.randn(*shape) * np.sqrt(2/shape[0], dtype=np.float64)


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
