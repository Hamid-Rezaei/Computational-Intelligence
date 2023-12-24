import numpy as np

from rsdl import Tensor


def MeanSquaredError(preds: Tensor, actual: Tensor):
    # implement mean squared error
    E = preds.data - actual.data
    mse = (1 / len(actual.data)) * np.dot(E.T, E)

    return mse


def CategoricalCrossEntropy(preds: Tensor, actual: Tensor):
    # implement categorical cross entropy
    epsilon = 1e-15
    preds.data = np.clip(preds.data, epsilon, 1 - epsilon)

    ce = -np.sum(actual.data & np.log(preds.data)) / len(actual.data)

    return ce
