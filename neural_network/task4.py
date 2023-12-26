# Task 4
import numpy as np

from rsdl import Tensor
from rsdl.losses import MeanSquaredError

X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-7, +3, -9]))
y = X @ coef + 6

# define w and b (y = w x + b) with random initialization ( you can use np.random.randn )
w = np.random.randn(3, 1) * 0.1
b = 0

print("W = ", w)
print("b = ", b)

learning_rate = 0.002
batch_size = 10

criterion = MeanSquaredError

for epoch in range(100):

    epoch_loss = 0.0

    for start in range(0, 100, batch_size):
        end = start + batch_size

        inputs = X[start:end]

        # predicted
        predicted = inputs @ w + b

        actual = y[start:end].data.reshape(batch_size, 1)

        # calculate MSE loss
        loss = criterion(predicted, actual)

        # backward
        inputs.data = inputs.data.T

        gradient_w = (2 / batch_size) * (inputs @ (predicted - actual)).data
        gradient_b = (2 / batch_size) * (predicted - actual).sum().data

        epoch_loss += loss

        # update w and b (Don't use 'w -= ' and use ' w = w - ...') (you don't need to use optim.SGD in this task)

        w = w - learning_rate * gradient_w
        b = b - learning_rate * gradient_b

print(w)
print(b)
