# Task 5
import numpy as np

from rsdl import Tensor
from rsdl.layers import Linear
from rsdl.optim import SGD
from rsdl.losses import loss_functions

X = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-7, +3, -9]))
y = X @ coef + 5

# TODO: define a linear layer using Linear() class  
l = ...

# TODO: define an optimizer using SGD() class 
optimizer = ....

# TODO: print weight and bias of linear layer


learning_rate = ...
batch_size = ...

for epoch in range(100):
    
    epoch_loss = 0.0
    
    for start in range(0, 100, batch_size):
        end = start + batch_size


        print(start, end)

        inputs = X[start:end]

        # TODO: predicted
        predicted = ...

        actual = y[start:end]
        actual.data = actual.data.reshape(batch_size, 1)
        # TODO: calcualte MSE loss
        
        # TODO: backward
        # hint you need to just do loss.backward()


        # TODO: add loss to epoch_loss
        epoch_loss += ...


        # TODO: update w and b using optimizer.step()
        

# TODO: print weight and bias of linear layer
 
