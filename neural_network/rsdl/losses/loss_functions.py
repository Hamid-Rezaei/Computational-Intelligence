from rsdl import Tensor


def MeanSquaredError(preds: Tensor, actual: Tensor):
    # implement mean squared error
    SErr = (preds - actual) ** 2
    mse = SErr.sum() * Tensor(1 / len(actual.data), requires_grad=actual.requires_grad, depends_on=actual.depends_on)

    return mse


def CategoricalCrossEntropy(preds: Tensor, actual: Tensor):
    # implement categorical cross entropy
    sum_exp_preds = preds.exp().sum()
    softmax_sum = sum_exp_preds * Tensor(
        data=(1 / sum_exp_preds.data),
        requires_grad=sum_exp_preds.requires_grad,
        depends_on=sum_exp_preds.depends_on
    )
    log_softmax = softmax_sum.log()
    vector_cce = - (actual * log_softmax)
    cce = - (vector_cce.sum())
    return cce
