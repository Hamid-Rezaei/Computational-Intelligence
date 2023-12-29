import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rsdl.activations import Relu, Softmax
from rsdl.layers import Linear
from rsdl.losses import CategoricalCrossEntropy
from rsdl.optim import SGD
from rsdl.tensors import Tensor

# 1. load mnist dataset for our framework

# load data
train_set = pd.read_csv("../Datasets/MNIST/csv/train.csv", sep=',', header=None)
train_label = train_set.iloc[:, 0]
train_set = train_set.iloc[:, 1:]

test_set = pd.read_csv("../Datasets/MNIST/csv/test.csv", sep=',', header=None)
test_label = test_set.iloc[:, 0]
test_set = test_set.iloc[:, 1:]

# Convert to numpy array
train_set = train_set.to_numpy()
train_label = train_label.to_numpy()

test_set = test_set.to_numpy()
test_label = test_label.to_numpy()

# Convert to tensor
train_set = Tensor(data=train_set)
train_label = Tensor(data=train_label)
test_set = Tensor(data=test_set)
test_label = Tensor(data=test_label)


# 2. define your model

class Model:
    def __init__(self, in_features, hidden_size, out_classes):
        # Input Layer -> Hidden Layer -> Softmax Layer
        self.fully_connected1 = Linear(in_features, hidden_size)
        self.fully_connected2 = Linear(hidden_size, out_classes)
        self.relu = Relu
        self.softmax = Softmax

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # The Forward Pass
        linear1 = self.fully_connected1(x)
        relu1 = self.relu(linear1)
        linear2 = self.fully_connected2(relu1)
        softmax = self.softmax(linear2)

        return softmax


# 3. start training and have fun!

def train_one_epoch(model, data, optimizer, criterion, length):
    epoch_loss = 0.0
    accuracy = 0.0
    for start in range(0, length, batch_size):
        end = start + batch_size

        inputs = data[start:end]
        inputs.zero_grad()
        model.fully_connected1.zero_grad()
        model.fully_connected2.zero_grad()

        # predicted
        predicted = model(inputs)

        actual = train_label[start:end]
        actual_list = actual.data.tolist()
        actual.data = np.eye(10)[actual_list]

        loss = criterion(predicted, actual)

        # backward
        loss.backward()

        # add loss to epoch_loss
        epoch_loss = epoch_loss + loss

        # update w and b using optimizer.step()
        optimizer.step()

        yp = np.argmax(predicted.data, axis=1)
        accuracy += np.sum(yp == np.array(actual_list))

    print()
    print("Accuracy:", accuracy / length)
    return epoch_loss, accuracy / length


def caluculate_acc(data, model, length):
    accuracy = 0.0
    for start in range(0, length, batch_size):
        end = start + batch_size

        inputs = data[start:end]

        # predicted
        predicted = model(inputs)

        actual = train_label[start:end]
        yp = np.argmax(predicted.data, axis=1)
        accuracy += np.sum(yp == np.array(actual))

    return accuracy / length


# Defining model
model = Model(in_features=28 * 28, hidden_size=256, out_classes=10)

# Defining optimizer
optimizer = SGD(layers=[model.fully_connected1, model.fully_connected2], learning_rate=2e-3)

# Defining loss
criterion = CategoricalCrossEntropy


batch_size = 256

# Train
val_accs = []
accs = []
for epoch in tqdm(range(20)):
    accs.append(train_one_epoch(model, train_set, optimizer, criterion, len(train_set.data)))
    val_accs.append(caluculate_acc(test_set, model, len(train_set.data)))


print()
print(
    f"test accuracy: {caluculate_acc(test_set, model, len(test_set.data))}, "
    f"train accuracy: {caluculate_acc(train_set, model, len(train_set.data))}"
)
