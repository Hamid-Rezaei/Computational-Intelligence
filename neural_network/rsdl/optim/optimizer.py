class Optimizer:
    def __init__(self, layers):
        self.layers = layers
    
    def zero_grad(self):
        for l in self.layers:
            l.zero_grad()