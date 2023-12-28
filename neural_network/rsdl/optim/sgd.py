from rsdl.optim import Optimizer


# implement step function
class SGD(Optimizer):
    def __init__(self, layers, learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        # update weight and biases ( Don't use '-=' and use l.weight = l.weight - ... )

        for param in self.layers.parameters()[0]:
            gradient = param.grad

            self.layers.weight.data = self.layers.weight.data - self.learning_rate * gradient

        for param in self.layers.parameters()[1]:
            gradient = param.grad
            if self.layers.need_bias:
                self.layers.bias.data = self.layers.bias.data - self.learning_rate * gradient
