from rsdl.optim import Optimizer
# TODO: implement Adam optimizer like SGD
import numpy as np
from rsdl import Tensor


class Adam(Optimizer):
    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        for l in self.layers:
            l.moment1 = Tensor(np.zeros_like(l.weight.data), requires_grad=False)
            l.moment2 = Tensor(np.zeros_like(l.weight.data), requires_grad=False)
            if l.need_bias:
                l.bias_moment1 = Tensor(np.zeros_like(l.bias.data), requires_grad=False)
                l.bias_moment2 = Tensor(np.zeros_like(l.bias.data), requires_grad=False)
        self.iteration = 0

    def step(self):
        self.iteration += 1
        for l in self.layers:
            l.moment1.data = self.beta1 * l.moment1.data + (1 - self.beta1) * l.weight.grad.data
            l.moment2.data = self.beta2 * l.moment2.data + (1 - self.beta2) * l.weight.grad.data ** 2
            moment1_hat = l.moment1.data / (1 - self.beta1 ** self.iteration)
            moment2_hat = l.moment2.data / (1 - self.beta2 ** self.iteration)
            l.weight.data = l.weight.data - self.learning_rate * moment1_hat / (np.sqrt(moment2_hat) + self.epsilon)
            if l.need_bias:
                l.bias_moment1.data = self.beta1 * l.bias_moment1.data + (1 - self.beta1) * l.bias.grad.data
                l.bias_moment2.data = self.beta2 * l.bias_moment2.data + (1 - self.beta2) * l.bias.grad.data ** 2
                bias_moment1_hat = l.bias_moment1.data / (1 - self.beta1 ** self.iteration)
                bias_moment2_hat = l.bias_moment2.data / (1 - self.beta2 ** self.iteration)
                l.bias.data = l.bias.data - self.learning_rate * bias_moment1_hat / (
                        np.sqrt(bias_moment2_hat) + self.epsilon)
