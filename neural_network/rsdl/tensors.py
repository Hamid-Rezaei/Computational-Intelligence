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
            data : np.ndarray,
            requires_grad: bool = False,
            depends_on : List[Dependency] = None) -> None:
        
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
        self.grad : Optional['Tensor'] = None

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
        # TODO: implement sum over tensor elems
        # Hint use _tensor_sum function
        return None
    
    def log(self) -> 'Tensor':
        # TODO: implement log
        # Hint use _tensor_log function
        return None
    
    def exp(self) -> 'Tensor':
        # TODO: implement exp
        # Hint use _tensor_exp function
        return None

    def __add__(self, other) -> 'Tensor':
        # Done ( Don't change )
        # Hint use _add function
        # self + other
        return _add(self, ensure_tensor(other))
    
    def __radd__(self, other) -> 'Tensor':
        # TODO: implement radd
        # Hint use _add function
        # other + self
        return None
    
    def __iadd__(self, other) -> 'Tensor':
        # TODO: implement inc add
        # Hint use _add function
        # self += other
        return None
    
    def __sub__(self, other) -> 'Tensor':
        # TODO: implement sub
        # Hint use _sub function
        # self - other
        return None
    
    def __rsub__(self, other) -> 'Tensor':
        # TODO: implement rsub
        # Hint use _sub function
        # other - self
        return None
    
    def __isub__(self, other) -> 'Tensor':
        # TODO: implement inc sub
        # Hint use _sub function
        # self -= other
        return None
    
    def __mul__(self, other) -> 'Tensor':
        # TODO: implement elemnet-wise mul
        # Hint use _mul function
        # self * other
        return None
    
    def __rmul__(self, other) -> 'Tensor':
        # TODO: implement elemnet-wise rmul
        # Hint use _mul function
        # other * self
        return None
    
    def __imul__(self, other) -> 'Tensor':
        # TODO: implement elemnet-wise inc mul
        # Hint use _mul function
        # self *= other
        return None

    def __matmul__(self, other) -> 'Tensor':
        # TODO: implement matrix mul
        # Hint use _matmul function
        # self @ other
        return None
    
    def __pow__(self, power: float):
        # TODO: implement power
        # Hint use _tensor_pow function
        # self ** power
        return None
    
    def __getitem__(self, idcs):
        # TODO: implement getitem [:]
        # Hint use _tensor_slice function
        return None
    
    def __neg__(self, idcs):
        # TODO: implement neg (-)
        # Hint use -_tensor_neg function
        return None
        
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
    # TODO
    data = ...
    req_grad = ...
    
    if req_grad:
        def grad_fn(grad: np.ndarray):
            return None
        
        depends_on = [Dependency(t, grad_fn)]

    else:
        depends_on = []
    
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)

def _tensor_exp(t: Tensor) -> Tensor:
    # TODO
    data = ...
    req_grad = ...
    
    if req_grad:
        def grad_fn(grad: np.ndarray):
            return ...
        
        depends_on = [Dependency(t, grad_fn)]

    else:
        depends_on = []
    
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)

def _tensor_pow(t: Tensor, power:float) -> Tensor:
    # TODO
    data = ...
    req_grad = ...
    
    if req_grad:
        def grad_fn(grad: np.ndarray):
            return None
        
        depends_on = [Dependency(t, grad_fn)]

    else:
        depends_on = []
    
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)

def _tensor_slice(t: Tensor, idcs) -> Tensor:
    # TODO
    data = ...
    requires_grad = ...

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
    # TODO
    data = ...
    requires_grad = ...
    if requires_grad:
        depends_on = [Dependency(t, lambda x: -x)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def _add(t1: Tensor, t2: Tensor) -> Tensor:

    data = t1.data + t2.data
    req_grad = t1.requires_grad or t2.requires_grad
    depends_on : List[Dependency] = []

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
    # TODO: implement sub
    # Hint: a-b = a+(-b)
    return None

def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    # Done ( Don't change )
    data = t1.data * t2.data
    req_grad = t1.requires_grad or t2.requires_grad
    depends_on : List[Dependency] = []

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
    # TODO: implement matrix multiplication
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return ...
        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return ...
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)
