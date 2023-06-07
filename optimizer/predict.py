from sklearn import datasets
from torch import nn
import torch
import numpy as np
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


iris_X, iris_y = datasets.load_iris(return_X_y=True)
iris_X = torch.tensor(iris_X, dtype=torch.float)
iris_y = torch.tensor(iris_y, dtype=torch.long)


train_idx = np.random.choice(
    a=config["data_size"],
    size=config["train_size"],
    replace=False,
)
test_idx = np.array(list(set(range(config["data_size"])) - set(train_idx)))
data_train = iris_X[train_idx], iris_y[train_idx]
data_test = iris_X[test_idx], iris_y[test_idx]


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train(data_train, net, loss_fn, optimizer):
    X_train, y_train = data_train
    for batch in range(len(data_train)):
        pred = net(X_train[batch])
        loss = loss_fn(pred, y_train[batch])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            print(
                "loss: {}, pred: {}, actual: {}".format(
                    loss.data, torch.argmax(pred.data), y_train[batch]
                )
            )


net = Net()
loss_fn = nn.CrossEntropyLoss()

no_optimizer = NoOptimizer(
    params=net.parameters(),
)

my_sgd = MySGD(
    params=net.parameters(),
    lr=0.01,
)

my_momentum = MyMomentum(
    params=net.parameters(),
    lr=0.01,
    momentum=0.9,
)

my_adam = MyAdam(
    params=net.parameters(),
    lr=0.01,
    beta1=0.9,
    beta2=0.99,
    epsilon=1e-8,
)

epoch = 15
for _ in range(epoch):
    train(
        data_train=data_train,
        net=net,
        loss_fn=loss_fn,
        optimizer=my_adam,
    )
