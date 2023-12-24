import numpy as np
from rsdl.optim import Optimizer
from rsdl import Tensor


class RMSprop(Optimizer):
    def __init__(self, layers, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon

        self.squared_gradients = [
            Tensor(data=np.zeros_like(param.data)) for layer in self.layers for param in layer.parameters()
        ]

    def step(self):
        for squared_gradient, layer in zip(self.squared_gradients, self.layers):
            for param in layer.parameters():
                gradient = param.grad()

                squared_gradient.data = self.decay_rate * squared_gradient.data + (1 - self.decay_rate) * gradient ** 2

                param.data = param.data - (
                            self.learning_rate / (np.sqrt(squared_gradient.data) + self.epsilon)) * gradient
