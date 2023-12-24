import numpy as np
from typing import List, NamedTuple, Callable, Optional, Union


class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]


Arrayable = Union[float, list, np.ndarray]


def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


Tensorable = Union[float, 'Tensor', np.ndarray]


def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Tensor:

    def __init__(
            self,
            data: np.ndarray,
            requires_grad: bool = False,
            depends_on: List[Dependency] = None
    ) -> None:

        """
        Args:
            data: value of tensor (numpy.ndarray)
            requires_grad: if tensor needs grad (bool)
            depends_on: list of dependencies
        """
        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on

        if not depends_on:
            self.depends_on = []

        self.shape = self._data.shape
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        self.grad = None

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def sum(self) -> 'Tensor':
        return _tensor_sum(self)

    def log(self) -> 'Tensor':
        return _tensor_log(self)

    def exp(self) -> 'Tensor':
        return _tensor_exp(self)

    def __add__(self, other) -> 'Tensor':
        # self + other
        return _add(self, ensure_tensor(other))

    def __radd__(self, other) -> 'Tensor':
        # other + self
        return _add(ensure_tensor(other), self)

    def __iadd__(self, other) -> 'Tensor':
        # self += other
        res = _add(self, ensure_tensor(other))
        self.data = res.data
        self.grad = res.grad
        self.requires_grad = res.requires_grad

        return self

    def __sub__(self, other) -> 'Tensor':
        # self - other
        return _sub(self, ensure_tensor(other))

    def __rsub__(self, other) -> 'Tensor':
        # other - self
        return _sub(ensure_tensor(other), self)

    def __isub__(self, other) -> 'Tensor':
        # self -= other
        res = _sub(self, ensure_tensor(other))
        self.data = res.data
        self.grad = res.grad
        self.requires_grad = res.requires_grad

        return self

    def __mul__(self, other) -> 'Tensor':
        # self * other
        return _mul(self, ensure_tensor(other))

    def __rmul__(self, other) -> 'Tensor':
        # other * self
        return _mul(ensure_tensor(other), self)

    def __imul__(self, other) -> 'Tensor':
        # self *= other
        res = _mul(self, ensure_tensor(other))
        self.data = res.data
        self.grad = res.grad
        self.requires_grad = res.requires_grad

        return self

    def __matmul__(self, other) -> 'Tensor':
        # self @ other
        return _matmul(self, ensure_tensor(other))

    def __pow__(self, power: float):
        # self ** power
        return _tensor_pow(self, power)

    def __getitem__(self, idcs):
        return _tensor_slice(self, idcs)

    def __neg__(self, idcs):
        return _tensor_neg(self)

    def backward(self, grad: 'Tensor' = None) -> None:
        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")
        self.grad.data = self.grad.data + grad.data
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))


def _tensor_sum(t: Tensor) -> Tensor:
    data = t.data.sum()
    req_grad = t.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            return grad * np.ones_like(t.data)

        depends_on = [Dependency(t, grad_fn)]

    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)


def _tensor_log(t: Tensor) -> Tensor:
    data = np.log(t.data)
    req_grad = t.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            return grad / t.data

        depends_on = [Dependency(t, grad_fn)]

    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)


def _tensor_exp(t: Tensor) -> Tensor:
    data = np.exp(t.data)
    req_grad = t.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            return grad * np.exp(t.data)

        depends_on = [Dependency(t, grad_fn)]

    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)


def _tensor_pow(t: Tensor, power: float) -> Tensor:
    data = np.power(t.data, power)
    req_grad = t.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            return grad * power * np.power(t.data, power - 1)

        depends_on = [Dependency(t, grad_fn)]

    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)


def _tensor_slice(t: Tensor, idcs) -> Tensor:
    data = t.data[idcs]
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(data)
            bigger_grad[idcs] = grad
            return bigger_grad

        depends_on = Dependency(t, grad_fn)
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def _tensor_neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t, lambda x: -x)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)


def _add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    req_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(
        data=data,
        requires_grad=req_grad,
        depends_on=depends_on
    )


def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    # Hint: a-b = a+(-b)
    return _add(t1, _tensor_neg(t2))


def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    # Done ( Don't change )
    data = t1.data * t2.data
    req_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t1, grad_fn1))
    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(
        data=data,
        requires_grad=req_grad,
        depends_on=depends_on
    )


def _matmul(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return grad @ t2.data.T

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(
        data=data,
        requires_grad=requires_grad,
        depends_on=depends_on
    )
