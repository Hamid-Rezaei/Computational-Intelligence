# Task 4
import sys

sys.setrecursionlimit(100000)
import numpy as np

from rsdl import Tensor
from rsdl.losses import MeanSquaredError

X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-7, +3, -9]))
y = X @ coef + 6

# define w and b (y = w x + b) with random initialization
w = Tensor(data=np.random.randn(3, 1) * 0.1, requires_grad=True)
b = Tensor(data=np.random.randn(), requires_grad=True)

print("W = ", w)
print("b = ", b)

learning_rate = 0.002
batch_size = 10

criterion = MeanSquaredError

for epoch in range(200):

    epoch_loss = 0.0
    print("Epoch:", epoch)
    for start in range(0, 100, batch_size):
        end = start + batch_size

        inputs = X[start:end]

        inputs.zero_grad()
        w.zero_grad()
        b.zero_grad()

        # predicted
        predicted = inputs @ w + b

        actual = y[start:end]
        actual.data = actual.data.reshape(-1, 1)

        # calculate MSE loss
        loss = criterion(predicted, actual)

        # backward
        loss.backward()

        epoch_loss += loss

        # update w and b
        w = w - learning_rate * w.grad
        b = b - learning_rate * b.grad

print("final W:", w)
print("final b", b)
