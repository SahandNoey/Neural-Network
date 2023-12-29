from rsdl import Tensor
from rsdl.activations import Softmax
from rsdl.tensors import _tensor_neg

def MeanSquaredError(preds: Tensor, actual: Tensor) -> Tensor:
    # TODO : implement mean squared error
    err = actual - preds
    return (err ** 2).sum() * (Tensor(1 / len(actual.data), requires_grad=actual.requires_grad, depends_on=actual.depends_on))

def CategoricalCrossEntropy(preds: Tensor, actual: Tensor) -> Tensor:
    # TODO : imlement categorical cross entropy
    preds = Softmax(preds)
    log_preds = preds.log()
    return _tensor_neg((actual * log_preds).sum())

