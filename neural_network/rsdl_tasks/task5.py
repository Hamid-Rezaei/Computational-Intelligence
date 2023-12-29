# Task 5
import sys

sys.setrecursionlimit(100000)

import numpy as np
from rsdl import Tensor
from rsdl.layers import Linear
from rsdl.optim import SGD
from rsdl.losses import loss_functions

X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-7, +3, -9]))
y = X @ coef + 5

# define a linear layer using Linear() class
l = Linear(in_channels=3, out_channels=1)

# define an optimizer using SGD() class
criterion = loss_functions.MeanSquaredError
optimizer = SGD(layers=[l], learning_rate=0.002)

# print weight and bias of linear layer
print("initial weight:", l.weight)
print("initial bias:", l.bias)

learning_rate = 0.002
batch_size = 10

for epoch in range(100):

    epoch_loss = 0.0
    print("Epoch:", epoch)
    for start in range(0, 100, batch_size):
        end = start + batch_size

        print(start, end)

        inputs = X[start:end]

        # predicted
        predicted = l(inputs)

        actual = y[start:end]
        actual.data = actual.data.reshape(-1, 1)

        loss = criterion(predicted, actual)

        # backward
        # hint you need to just do loss.backward()
        loss.backward()

        # add loss to epoch_loss
        epoch_loss += loss

        # update w and b using optimizer.step()
        optimizer.step()

# print weight and bias of linear layer
print("final weight:", l.weight)
print("final bias:", l.bias)
