# TODO: in this task you have to
# 1. load mnist dataset for our framework
# 2. define your model
# 3. start training and have fun!
import pandas as pd
import numpy as np
from rsdl.layers import Linear
from rsdl.activations import Relu
from rsdl.optim import SGD
from rsdl.tensors import Tensor
from rsdl.losses import CategoricalCrossEntropy
import sys

sys.setrecursionlimit(10**8)

train_data = pd.read_csv(
    "/mnt/d/Projects/PythonProjects/computationalIntelligence/neuralNetworks/MNIST_CSV/mnist_train.csv",
    sep=',', header=None)
train_label = train_data.iloc[:, 0]
train_data = train_data.iloc[:, 1:]

train_data = train_data.to_numpy()
train_label = train_label.to_numpy()

test_data = pd.read_csv(
    "/mnt/d/Projects/PythonProjects/computationalIntelligence/neuralNetworks/MNIST_CSV/mnist_test.csv",
    sep=',', header=None)

test_label = test_data.iloc[:, 0]
test_data = test_data.iloc[:, 1:]
test_data = test_data.to_numpy()
test_label = test_label.to_numpy()

train_data = Tensor(data=train_data)
train_label = Tensor(data=train_label)
test_data = Tensor(data=test_data)
test_label = Tensor(data=test_label)

train_batch_size = 32
test_batch_size = 64
hidden_neuron_num = 50

model = {"layers": [
    Linear(28*28, hidden_neuron_num),
    Linear(hidden_neuron_num, 10)], "activations": [Relu]}

optimizer = SGD(layers=model["layers"], lr=1)

for epoch in range(20):
    epoch_loss = 0.0
    print(f"epoch is: {epoch}")
    print(f"epoch loss = {epoch_loss}")
    for start in range(0, train_data.shape[0], train_batch_size):
        end = start + train_batch_size

        inputs = train_data[start:end]

        inputs.zero_grad()
        for layer in model["layers"]:
            layer.zero_grad()

        hidden_input_data = model["layers"][0](inputs)
        hidden_output_data = model["activations"][0](hidden_input_data)
        output_data = model["layers"][1](hidden_output_data)

        actual = train_label[start:end]
        actual.data = np.eye(10)[actual.data.tolist()]

        loss = CategoricalCrossEntropy(output_data, actual)
        loss.backward()

        epoch_loss += loss

        optimizer.step()

    for start in range(0, test_data.shape[0], test_batch_size):
        end = start + test_batch_size
        inputs = test_data[start:end]

        hidden_input_data = model["layers"][0](inputs)
        hidden_output_data = model["activations"][0](hidden_input_data)
        output_data = model["layers"][1](hidden_output_data)

        output_data = np.argmax(output_data, axis=1)
        output_data = output_data.reshape(-1, 1)
