from rsdl import Tensor
from rsdl.activations import Softmax


def MeanSquaredError(preds: Tensor, actual: Tensor):
    # implement mean squared error
    SErr = (preds - actual) ** 2
    mse = SErr.sum() * Tensor(1 / len(actual.data), requires_grad=actual.requires_grad, depends_on=actual.depends_on)

    return mse


def CategoricalCrossEntropy(preds: Tensor, actual: Tensor):
    # implement categorical cross entropy
    softmax_pred = Softmax(preds)
    log_softmax = softmax_pred.log()
    vector_cce = actual * log_softmax
    cce = - (vector_cce.sum())
    return cce
