from rsdl.optim import Optimizer

# TODO: implement Momentum optimizer like SGD


class Momentum(Optimizer):
    def __init__(self, layers, learning_rate=0.1, momentum=0.9):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.momentum = momentum

    def step(self):
        for l in self.layers:
            parameters = l.parameters()
            if not hasattr(l, 'weight_momentum'):
                l.weight_momentum = 0.0
            if l.need_bias and not hasattr(l, 'bias_momentum'):
                l.bias_momentum = 0.0
            l.weight_momentum = self.momentum * l.weight_momentum + (1 - self.momentum) * parameters[0].grad
            l.weight = l.weight - self.learning_rate * l.weight_momentum
            if l.need_bias:
                l.bias_momentum = self.momentum * l.bias_momentum + (1 - self.momentum) * parameters[1].grad
                l.bias = l.bias - self.learning_rate * l.bias_momentum
