from rsdl.optim import Optimizer

# TODO: implement RMSprop optimizer like SGD
import numpy as np


class RMSprop(Optimizer):
    def __init__(self, layers, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon

    def step(self):
        for l in self.layers:
            parameters = l.parameters()

            if not hasattr(l, 'squared_grad_avg'):
                l.squared_grad_avg = 0.0
            l.squared_grad_avg = self.beta * l.squared_grad_avg + (1 - self.beta) * (parameters[0].grad ** 2)
            l.weight = l.weight - self.learning_rate * parameters[0].grad.data / (
                    np.sqrt(l.squared_grad_avg.data) + self.epsilon)
            if l.need_bias:
                if not hasattr(l, 'squared_bias_grad_avg'):
                    l.squared_bias_grad_avg = 0.0
                l.squared_bias_grad_avg = self.beta * l.squared_bias_grad_avg + (1 - self.beta) * (
                        parameters[1].grad ** 2)
                l.bias = l.bias - self.learning_rate * parameters[1].grad.data / (
                        np.sqrt(l.squared_bias_grad_avg.data) + self.epsilon)
