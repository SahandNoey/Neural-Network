from rsdl import Tensor, Dependency
import numpy as np
import rsdl.tensors as rs


def Sigmoid(t: Tensor) -> Tensor:
    # TODO: implement sigmoid function
    # hint: you can do it using function you've implemented (not directly define grad func)
    neg_exp = rs._tensor_neg(t).exp()
    denominator = neg_exp + 1.0
    return Tensor(data=(1 / denominator), requires_grad=denominator.requires_grad, depends_on=denominator.depends_on)


def Tanh(t: Tensor) -> Tensor:
    # TODO: implement tanh function
    # hint: you can do it using function you've implemented (not directly define grad func)
    exp_matrix = t.exp()
    neg_exp_matrix = rs._tensor_neg(t).exp()
    numerator = exp_matrix - neg_exp_matrix
    denominator = exp_matrix + neg_exp_matrix
    return numerator * Tensor(data=(1 / denominator.data), requires_grad=denominator.requires_grad, depends_on=denominator.depends_on)


def Softmax(t: Tensor) -> Tensor:
    # TODO: implement softmax function
    # hint: you can do it using function you've implemented (not directly define grad func)
    # hint: you can't use sum because it has not axis argument so there are 2 ways:
    #        1. implement sum by axis
    #        2. using matrix mul to do it :) (recommended)
    # hint: a/b = a*(b^-1)

    exp_matrix = t.exp()
    sum = exp_matrix @ Tensor(data=np.ones((exp_matrix.data.shape[1], 1)), requires_grad=True)
    sum_invert = Tensor(data=(1 / sum.data), requires_grad=sum.requires_grad, depends_on=sum.depends_on)
    return exp_matrix * sum_invert


def Relu(t: Tensor) -> Tensor:
    # TODO: implement relu function

    # use np.maximum
    data = np.maximum(0, t.data)
    req_grad = t.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            # use np.where
            return np.where(t.data > 0, grad, 0)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []
    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)


def LeakyRelu(t: Tensor, leak=0.05) -> Tensor:
    """
    fill 'data' and 'req_grad' and implement LeakyRelu grad_fn
    hint: use np.where like Relu method but for LeakyRelu
    """
    # TODO: implement leaky_relu function

    data = np.where(t.data > 0, t.data, leak * t.data)
    req_grad = t.requires_grad

    if req_grad:
        def grad_fn(grad: np.ndarray):
            return np.where(t.data > 0, grad, leak * grad)

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
