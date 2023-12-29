from rsdl import Tensor


def MeanSquaredError(predicts: Tensor, actual: Tensor):
    # TODO : implement mean squared error
    error = actual - predicts
    return (error ** 2).sum() * (Tensor(1 / len(actual.data), requires_grad=actual.requires_grad,
                                        depends_on=actual.depends_on))


def CategoricalCrossEntropy(predicts: Tensor, actual: Tensor):
    # TODO : implement categorical cross entropy
    sum_exps = predicts.exp().sum()

    softmax = sum_exps * Tensor(data=(1 / sum_exps.data), requires_grad=sum_exps.requires_grad,
                                depends_on=sum_exps.depends_on)
    log_softmax = softmax.log()

    l_vector = -actual * log_softmax
    return -l_vector.sum()



