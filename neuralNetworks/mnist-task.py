# TODO: in this task you have to
# 1. load mnist dataset for our framework
# 2. define your model
# 3. start training and have fun!
import pandas as pd
import numpy as np
from rsdl.layers import Linear
from rsdl.activations import Relu
from rsdl.optim import SGD
from rsdl.optim import Adam
from rsdl.tensors import Tensor
from rsdl.losses import CategoricalCrossEntropy
from rsdl.losses import MeanSquaredError
import time
import sys
import pickle

sys.setrecursionlimit(1000000)

# loading the data
train_data = pd.read_csv("./MNIST_CSV/mnist_train.csv", sep=',', header=None)
train_label = train_data.iloc[:, 0]
train_data = train_data.iloc[:, 1:]

train_data = train_data.to_numpy()
train_label = train_label.to_numpy()

test_data = pd.read_csv("./MNIST_CSV/mnist_test.csv", sep=',', header=None)
test_label = test_data.iloc[:, 0]
test_data = test_data.iloc[:, 1:]
test_data = test_data.to_numpy()
test_label = test_label.to_numpy()

# now converting to out Tensor class
train_data = Tensor(data=train_data)
train_label = Tensor(data=train_label)
test_data = Tensor(data=test_data)
test_label = Tensor(data=test_label)

# now normalizing the data
# normalize(train_data)
# normalize(test_data)

# now building the model
hidden_neuron_num = 100
model = {"layers": [
    Linear(784, hidden_neuron_num),
    Linear(hidden_neuron_num, 10)
],
    "activations": [
        Relu,
    ]
}

optimizer = SGD(layers=model["layers"], learning_rate=0.1)
batch_size = 32

for epoch in range(20):
    epoch_loss = 0.0
    print(f"epoch is: {epoch}")

    for start in range(0, train_data.shape[0], batch_size):
        end = start + batch_size

        inputs = train_data[start:end]

        inputs.zero_grad()
        for layer in model["layers"]:
            layer.zero_grad()

        # forward phase
        a1_raw = model["layers"][0](inputs)
        a1 = model["activations"][0](a1_raw)
        predicted = model["layers"][1](a1)

        actual = train_label[start:end]
        actual.data = np.eye(10)[actual.data.tolist()]

        loss = MeanSquaredError(predicted, actual)
        loss.backward()

        epoch_loss += loss

        optimizer.step()
        print("loss:", loss)

    print("epoch ended")
    # now evaluation
    print("now, evaluating...")
    for start in range(0, test_data.shape[0], batch_size):
        end = start + batch_size
        inputs = test_data[start:end]

        a1_raw = model["layers"][0](inputs)
        a1 = model["activations"][0](a1_raw)
        predicted = model["layers"][1](a1)

        predicted = np.argmax(predicted, axis=1)
        predicted = predicted.reshape(-1, 1)
        print("predicted is:", predicted)

print("now here")