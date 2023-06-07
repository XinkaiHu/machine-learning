import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models

import matplotlib.pyplot as plt
from PIL import Image


device = (
    torch.device(device="cuda")
    if torch.cuda.is_available()
    else torch.device(device="cpu")
)
print("Device: {}".format(device))

config = {
    "training_path": os.path.join(__file__, "..", "dataset", "training"),
    "test_path": os.path.join(__file__, "..", "dataset", "test"),
    "weights_path": os.path.join(__file__, "..", "model_weights.pth"),
    "optimizer_path": os.path.join(__file__, "..", "optimizer_state.pth"),
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


training_transform = transforms.Compose(
    transforms=[
        transforms.RandomResizedCrop(
            size=(config["HEIGHT"], config["WIDTH"]),
            scale=(0.5, 1.0),
            ratio=(1.0, 1.0),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


test_transform = transforms.Compose(
    transforms=[
        transforms.Resize(size=(config["_RESIZE_SIDE_MIN"])),
        transforms.CenterCrop(size=(config["HEIGHT"], config["WIDTH"])),
        transforms.ToTensor(),
    ]
)


training_data = datasets.ImageFolder(
    root=config["training_path"],
    transform=training_transform,
)


test_data = datasets.ImageFolder(
    root=config["test_path"],
    transform=test_transform,
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
print("\n")

plt.figure()
plt.imshow(training_features[0, 0, ...])
plt.colorbar()
plt.grid(visible=False)
plt.show()


training_loss = []
test_loss = []
accuracy_list = []


def train_loop(dataloader, model, loss_fn, optimizer):
    avg_loss = 0
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        avg_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss /= len(dataloader)
    training_loss.append(avg_loss)


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    avg_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    avg_loss /= num_batches
    test_loss.append(avg_loss)
    correct /= size
    accuracy_list.append(correct)
    print("Accuracy: {}, Avg loss: {}\n".format(correct, avg_loss))


model = models.resnet50()
model.fc = nn.Linear(in_features=2048, out_features=5)
model.to(device=device)
model.load_state_dict(state_dict=torch.load(f=config["weights_path"]))
model.eval()

lr = 5e-4
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    params=model.parameters(),
    lr=lr,
    momentum=0.9,
    weight_decay=1e-4,
)
optimizer.load_state_dict(state_dict=torch.load(f=config["optimizer_path"]))

epoch = 0

for _ in range(epoch):
    print("Epoch {}".format(_))
    train_loop(
        dataloader=training_loader, model=model, loss_fn=loss_fn, optimizer=optimizer
    )
    test_loop(dataloader=test_loader, model=model, loss_fn=loss_fn)
torch.save(obj=model.state_dict(), f=config["weights_path"])
torch.save(obj=optimizer.state_dict(), f=config["optimizer_path"])

with open(os.path.join(__file__, "..", "training_loss.txt"), mode="a") as f:
    for loss in training_loss:
        f.write("{}\n".format(loss))
with open(os.path.join(__file__, "..", "test_loss.txt"), mode="a") as f:
    for loss in test_loss:
        f.write("{}\n".format(loss))
with open(os.path.join(__file__, "..", "accuracy.txt"), mode="a") as f:
    for correct in accuracy_list:
        f.write("{}\n".format(correct))
with open(file=os.path.join(__file__, "..", "training_loss.txt"), mode="r") as f:
    training_loss = [float(loss) for loss in f.readlines()]
with open(file=os.path.join(__file__, "..", "test_loss.txt"), mode="r") as f:
    test_loss = [float(loss) for loss in f.readlines()]
with open(file=os.path.join(__file__, "..", "accuracy.txt"), mode="r") as f:
    accuracy = [float(acc) for acc in f.readlines()]
plt.figure()
plt.plot(training_loss, label="training loss")
plt.plot(test_loss, label="test loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(accuracy, label="Accuracy")
plt.legend()
plt.show()


class_map = training_data.classes

rose = Image.open(
    fp=os.path.join(__file__, "..", "dataset", "test", "roses", "rose.jpg")
)
sunflower = Image.open(
    fp=os.path.join(__file__, "..", "dataset", "test", "sunflowers", "sunflower.jpg")
)
tulips = Image.open(
    fp=os.path.join(__file__, "..", "dataset", "test", "tulips", "tulips.jpg")
)

rose = test_transform(rose)
sunflower = test_transform(sunflower)
tulips = test_transform(tulips)

rose = torch.unsqueeze(input=rose, dim=0)
sunflower = torch.unsqueeze(input=sunflower, dim=0)
tulips = torch.unsqueeze(input=tulips, dim=0)

rose = rose.to(device=device)
sunflower = sunflower.to(device=device)
tulips = tulips.to(device=device)

pred = model(rose)
print("Prediction of rose.jpg:")
print(
    "Scores:\n{},\nprediction: {},\npredicted label: {}\n".format(
        pred.data, torch.argmax(pred.data), class_map[torch.argmax(pred.data)]
    )
)

pred = model(sunflower)
print("Prediction of sunflower.jpg:")
print(
    "Scores:\n{},\nprediction: {},\npredicted label: {}\n".format(
        pred.data, torch.argmax(pred.data), class_map[torch.argmax(pred.data)]
    )
)

pred = model(tulips)
print("Prediction of tulips.jpg:")
print(
    "Scores:\n{},\nprediction: {},\npredicted label: {}\n".format(
        pred.data, torch.argmax(pred.data), class_map[torch.argmax(pred.data)]
    )
)
