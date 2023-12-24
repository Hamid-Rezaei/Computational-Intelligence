import numpy as np

from rsdl.optim import Optimizer
from rsdl import Tensor


class Momentum(Optimizer):
    def __init__(self, layers, learning_rate=0.1, momentum_factor=0.9):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.momentum_factor = momentum_factor

        self.velocities = [
            Tensor(data=np.zeros_like(param.data)) for layer in self.layers for param in layer.parameters()
        ]

    def step(self):
        for velocity, layer in zip(self.velocities, self.layers):
            for param in layer.parameters():
                gradient = param.grad()

                velocity.data = self.momentum_factor * velocity.data - self.learning_rate * gradient

                param.data = param.data + velocity.data
