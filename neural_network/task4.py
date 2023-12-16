# Task 4
import numpy as np

from rsdl import Tensor
from rsdl.losses import loss_functions

X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-7, +3, -9]))
y = X @ coef + 6

# TODO: define w and b (y = w x + b) with random initialization ( you can use np.random.randn )
w = ...
b = ...

print(w)
print(b)

learning_rate = ...
batch_size = ...

for epoch in range(100):
    
    epoch_loss = 0.0
    
    for start in range(0, 100, batch_size):
        end = start + batch_size

        inputs = X[start:end]

        # TODO: predicted
        predicted = ...

        actual = y[start:end]
        # TODO: calcualte MSE loss
        
        # TODO: backward
        # hint you need to just do loss.backward()

        epoch_loss += ...


        # TODO: update w and b (Don't use 'w -= ' and use ' w = w - ...') (you don't need to use optim.SGD in this task)
        w = ...
        b = ...

print(w)
print(b)

