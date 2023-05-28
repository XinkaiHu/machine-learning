import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models

import matplotlib.pyplot as plt

device = (
    torch.device(device="cuda")
    if torch.cuda.is_available()
    else torch.device(device="cpu")
)
print("Device: {}".format(device))

config = {
    "data_path": os.path.join(__file__, "..", "dataset", "training"),
    "test_path": os.path.join(__file__, "..", "dataset", "test"),
    "weights_path": os.path.join(__file__, "..", "model_weights.pth"),
    "data_size": 3616,
    "HEIGHT": 224,
    "WIDTH": 224,
    "_R_MEAN": 123.68,
    "_G_MEAN": 116.78,
    "_B_MEAN": 103.94,
    "_R_STD": 1,
    "_G_STD": 1,
    "_B_STD": 1,
    "_RESIZE_SIDE_MIN": 256,
    "_RESIZE_SIDE_MAX": 512,
    "batch_size": 32,
    "num_class": 5,
    "epoch_size": 150,
    "loss_scale_num": 1024,
    "prefix": "resent-ai",
    "directory": "model_resnet",
    "save_checkpoint_steps": 10,
}

lr = 5e-4


training_data = datasets.ImageFolder(
    root=config["data_path"],
    transform=transforms.Compose(
        transforms=[
            transforms.RandomResizedCrop(
                size=(config["HEIGHT"], config["WIDTH"]),
                scale=(0.5, 1.0),
                ratio=(1.0, 1.0),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ),
)


test_data = datasets.ImageFolder(
    root=config["test_path"],
    transform=transforms.Compose(
        transforms=[
            transforms.Resize(size=(config["_RESIZE_SIDE_MIN"])),
            transforms.CenterCrop(size=(config["HEIGHT"], config["WIDTH"])),
            transforms.ToTensor(),
        ]
    ),
)


training_loader = DataLoader(
    dataset=training_data,
    batch_size=config["batch_size"],
    shuffle=True,
    drop_last=True,
)

test_loader = DataLoader(
    dataset=test_data,
    batch_size=1,
)

print("Training set size: {}".format(len(training_data)))
print("Test set size: {}".format(len(test_data)))

training_features, training_labels = next(iter(training_loader))
print("Channel size, height, width: {}".format(training_features[0].shape))
print("Label of the first photo: {}".format(training_labels[0]))

plt.figure()
plt.imshow(training_features[0, 0, ...])
plt.colorbar()
plt.grid(visible=False)
plt.show()

model = models.resnet50()
model.fc = nn.Linear(in_features=2048, out_features=5)
model.to(device=device)
model.load_state_dict(torch.load(f=config["weights_path"]))
model.eval()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    params=model.parameters(),
    lr=lr,
    momentum=0.9,
    weight_decay=1e-4,
)


def train_loop(dataloader, model, loss_fn, optimizer):
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print("Accuracy: {}, Avg loss: {}\n".format(correct, test_loss))


epoch = 1
for _ in range(epoch):
    print("Epoch {}".format(_))
    train_loop(
        dataloader=training_loader, model=model, loss_fn=loss_fn, optimizer=optimizer
    )
    test_loop(dataloader=test_loader, model=model, loss_fn=loss_fn)
    torch.save(obj=model.state_dict(), f=config["weights_path"])
