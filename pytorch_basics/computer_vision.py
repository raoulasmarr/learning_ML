import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

print(torch.__version__)
print(torchvision.__version__)

def main():
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    image, label = train_data[0]
    print(f"Train data shape: {train_data}")
    print(f"Image shape: {image.shape}, Label: {label}")
    print(f"Train targets: {train_data.targets}")
    print(f"Train data: {train_data.data}")
    plt.imshow(image.squeeze(), cmap="gray")
    plt.show()
main()
 