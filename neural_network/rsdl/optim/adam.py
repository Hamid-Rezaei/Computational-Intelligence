import numpy as np

from rsdl import Tensor
from rsdl.optim import Optimizer


# implement Adam optimizer like SGD
class Adam(Optimizer):
    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.momentums = [
            Tensor(data=np.zeros_like(param.data)) for layer in self.layers for param in layer.parameters()
        ]
        self.squared_momentums = [
            Tensor(data=np.zeros_like(param.data)) for layer in self.layers for param in layer.parameters()
        ]
        self.timestep = 0

    def step(self):
        self.timestep += 1

        for layer in self.layers:
            for param, momentum, squared_momentum in zip(layer.parameters(), self.momentums, self.squared_momentums):
                gradient = param.grad()

                momentum.data = self.beta1 * momentum.data + (1 - self.beta1) * gradient

                squared_momentum.data = self.beta2 * squared_momentum.data + (1 - self.beta2) * gradient ** 2

                corrected_momentum = momentum.data / (1 - self.beta1 ** self.timestep)
                corrected_squared_momentum = squared_momentum.data / (1 - self.beta2 ** self.timestep)

                param.data = param.data - self.learning_rate * corrected_momentum / (
                            np.sqrt(corrected_squared_momentum) + self.epsilon)
