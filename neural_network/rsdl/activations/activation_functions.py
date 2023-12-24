from rsdl import Tensor, Dependency
import numpy as np


def Sigmoid(t: Tensor) -> Tensor:
    # implement sigmoid function
    t = -t
    return 1 / (1 + t.exp())


def Tanh(t: Tensor) -> Tensor:
    # implement tanh function
    x = t
    neg_x = -t
    tanh = (x.exp() - neg_x.exp()) / (x.exp() + neg_x.exp())

    return tanh


def Softmax(t: Tensor) -> Tensor:
    # TODO: implement softmax function
    # hint: you can do it using function you've implemented (not directly define grad func)
    # hint: you can't use sum because it has not axis argument so there are 2 ways:
    #        1. implement sum by axis
    #        2. using matrix mul to do it :) (recommended)
    # hint: a/b = a*(b^-1)
    return None


def Relu(t: Tensor) -> Tensor:
    # implement relu function

    data = np.maximum(0, t.data)
    req_grad = t.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            return grad * np.where(t.data < 0, 0, 1)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)


def LeakyRelu(t: Tensor, leak=0.05) -> Tensor:
    # implement leaky_relu function

    data = np.where(t.data < 0, leak * t.data, t.data)
    req_grad = t.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            return grad * np.where(t.data < 0, leak, 1)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
