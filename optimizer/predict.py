from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
import torch
import os

from no_optimizer import NoOptimizer
from sgd import MySGD
from momentum import MyMomentum
from adam import MyAdam


config = {
    "data_size": 150,
    "train_size": 120,
    "test_size": 30,
    "feature_number": 4,
    "num_class": 3,
    "batch_size": 30,
    "save_checkpoint_steps": 5,
    "keep_checkpoint_max": 1,
    "out_dir_no_opt": os.path.join(".", "model_iris", "no_opt"),
    "out_dir_sgd": os.path.join(".", "model_iris", "sgd"),
    "out_dir_momentum": os.path.join(".", "model_iris", "momentum"),
    "out_dir_adam": os.path.join(".", "model_iris", "adam"),
    "out_dir_prefix": "checkpoint_fashion_forward",
}


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train(data, net, loss_fn, optimizer):
    X_train, y_train = data
    for batch in range(y_train.shape[0]):
        pred = net(X_train[batch])
        loss = loss_fn(pred, y_train[batch])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(data, net, loss_fn):
    X_train, y_train = data
    loss = 0
    currect = 0
    with torch.no_grad():
        for batch in range(y_train.shape[0]):
            pred = net(X_train[batch])
            loss += loss_fn(pred, y_train[batch])
            if torch.argmax(pred) == y_train[batch]:
                currect += 1
    loss /= y_train.shape[0]
    currect /= y_train.shape[0]
    return loss, currect


iris_X, iris_y = datasets.load_iris(return_X_y=True)
X_training, X_test, y_training, y_test = train_test_split(
    iris_X, iris_y, test_size=config["test_size"], train_size=config["train_size"]
)
X_training = torch.tensor(X_training, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
y_training = torch.tensor(y_training, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


net0 = Net()
net1 = Net()
net2 = Net()
net3 = Net()
net4 = Net()
net5 = Net()
net6 = Net()
loss_fn = nn.CrossEntropyLoss()

no_optimizer = NoOptimizer(
    params=net0.parameters(),
)

my_sgd = MySGD(
    params=net1.parameters(),
    lr=0.05,
)

sgd = optim.SGD(
    params=net2.parameters(),
    lr=0.05,
    momentum=0.0,
)

my_momentum = MyMomentum(
    params=net3.parameters(),
    lr=0.01,
    momentum=0.9,
)

momentum = optim.SGD(
    params=net4.parameters(),
    lr=0.01,
    momentum=0.9,
)

my_adam = MyAdam(
    params=net5.parameters(),
    lr=0.001,
    beta1=0.9,
    beta2=0.99,
    epsilon=1e-8,
)

adam = optim.Adam(
    params=net6.parameters(),
    lr=0.001,
    betas=[
        0.9,
        0.99,
    ],
    eps=1e-8,
)

epoch = 10
print("=========== No optimizer ===========")
for _ in range(epoch):
    train(
        data=(X_training, y_training),
        net=net0,
        loss_fn=loss_fn,
        optimizer=no_optimizer,
    )

    test_loss, test_accuracy = test(
        data=(X_test, y_test),
        net=net0,
        loss_fn=loss_fn,
    )

    print("Epoch {}:\tLoss is {},\taccuracy is {}".format(_, test_loss, test_accuracy))
print("=========== My SGD ===========")
for _ in range(epoch):
    train(
        data=(X_training, y_training),
        net=net1,
        loss_fn=loss_fn,
        optimizer=my_sgd,
    )

    test_loss, test_accuracy = test(
        data=(X_test, y_test),
        net=net1,
        loss_fn=loss_fn,
    )

    print("Epoch {}:\tLoss is {},\taccuracy is {}".format(_, test_loss, test_accuracy))
print("=========== PyTorch SGD ===========")
for _ in range(epoch):
    train(
        data=(X_training, y_training),
        net=net2,
        loss_fn=loss_fn,
        optimizer=sgd,
    )

    test_loss, test_accuracy = test(
        data=(X_test, y_test),
        net=net2,
        loss_fn=loss_fn,
    )

    print("Epoch {}:\tLoss is {},\taccuracy is {}".format(_, test_loss, test_accuracy))
print("=========== My Momentum ===========")
for _ in range(epoch):
    train(
        data=(X_training, y_training),
        net=net3,
        loss_fn=loss_fn,
        optimizer=my_momentum,
    )

    test_loss, test_accuracy = test(
        data=(X_test, y_test),
        net=net3,
        loss_fn=loss_fn,
    )

    print("Epoch {}:\tLoss is {},\taccuracy is {}".format(_, test_loss, test_accuracy))
print("=========== PyTorch Momentum ===========")
for _ in range(epoch):
    train(
        data=(X_training, y_training),
        net=net4,
        loss_fn=loss_fn,
        optimizer=momentum,
    )

    test_loss, test_accuracy = test(
        data=(X_test, y_test),
        net=net4,
        loss_fn=loss_fn,
    )

    print("Epoch {}:\tLoss is {},\taccuracy is {}".format(_, test_loss, test_accuracy))
print("=========== My Adam ===========")
for _ in range(epoch):
    train(
        data=(X_training, y_training),
        net=net5,
        loss_fn=loss_fn,
        optimizer=my_adam,
    )

    test_loss, test_accuracy = test(
        data=(X_test, y_test),
        net=net5,
        loss_fn=loss_fn,
    )

    print("Epoch {}:\tLoss is {},\taccuracy is {}".format(_, test_loss, test_accuracy))
print("=========== PyTorch Adam ===========")
for _ in range(epoch):
    train(
        data=(X_training, y_training),
        net=net6,
        loss_fn=loss_fn,
        optimizer=adam,
    )

    test_loss, test_accuracy = test(
        data=(X_test, y_test),
        net=net6,
        loss_fn=loss_fn,
    )

    print("Epoch {}:\tLoss is {},\taccuracy is {}".format(_, test_loss, test_accuracy))
print("=========== End ===========")
