from rsdl.optim import Optimizer

# TODO: implement step function


class SGD(Optimizer):
    def __init__(self, layers, lr=0.1):
        super().__init__(layers)
        self.lr = lr

    def step(self):
        # TODO: update weight and biases ( Don't use '-=' and use l.weight = l.weight - ... )
        for l in self.layers:
            weights_and_bias = l.parameters()
            l.weight = l.weight - self.lr * weights_and_bias[0].grad
            if l.need_bias:
                l.bias = l.bias - self.lr * weights_and_bias[1].grad
