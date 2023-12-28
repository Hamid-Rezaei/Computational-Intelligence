from rsdl.optim import Optimizer


# implement step function
class SGD(Optimizer):
    def __init__(self, layers, learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        for l in self.layers:
            params = l.parameters()
            l.weight = l.weight - self.learning_rate * params[0].grad
            if l.need_bias:
                l.bias = l.bias - self.learning_rate * params[1].grad
