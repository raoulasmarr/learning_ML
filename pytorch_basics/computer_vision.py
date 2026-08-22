import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
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


    #Displaying an image and cheching image shape
    image, label = train_data[0]
    print(f"Train data shape: {train_data}")
    print(f"Image shape: {image.shape}, Label: {label}")
    print(f"Train targets: {train_data.targets}")
    print(f"Train data: {train_data.data}")
    plt.imshow(image.squeeze(), cmap="gray")
    plt.show()

    # loading data into 32 batch size

    BATCH_SIZE = 32

    train_dataloader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False)

    print(train_dataloader, test_dataloader)
    print(len(train_dataloader))
    print(  BATCH_SIZE )

    #CHECK WHATS inside the data loaders
    train_features_batch, train_labels_batch= next(iter(train_dataloader[1]))
    print(train_features_batch, train_labels_batch)
    print(train_features_batch.shape, train_labels_batch.shape)


main()
 