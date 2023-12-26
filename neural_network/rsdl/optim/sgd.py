from rsdl.optim import Optimizer


# implement step function
class SGD(Optimizer):
    def __init__(self, layers, learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        # update weight and biases ( Don't use '-=' and use l.weight = l.weight - ... )
        for layer in self.layers:
            for param in layer.parameters():
                gradient = param.grad()

                layer.weight.data = layer.weight.data - self.learning_rate * gradient
                if layer.need_bias:
                    layer.bias.data = layer.bias.data - self.learning_rate * gradient
