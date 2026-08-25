import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from timeit import default_timer as timer #To time the training
from tqdm.auto import tqdm #progress bar


if Path("helper_functions.py").is_file():
    print("helper_funcation.py already exists, skipping download")
else:
    print("Downloading helper_function.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)


from helper_functions import accuracy_fn

print("torchvision:", torchvision.__version__)
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"



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
    #plt.show()

    # loading data into 32 batch size

    BATCH_SIZE = 32

    train_dataloader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False)

    print(train_dataloader, test_dataloader)
    print(len(train_dataloader))
    print(  BATCH_SIZE )

    class_names = train_data.classes[:]
    #CHECK WHATS inside the data loaders
    #print(class_names)
    train_features_batch, train_labels_batch= next(iter(train_dataloader))
    #print(train_features_batch, train_labels_batch)
    #print(train_features_batch.shape, train _labels_batch.shape)

    torch.manual_seed(42)

    model_0 = FashionMNISTModelV0(input_shape=784, hidden_units= 10, output_shape= len(class_names))
    model_1 = FashionMNISTModelV1(input_shape=784, # number of input features
    hidden_units=10,
    output_shape=len(class_names)).to(device)
    model_0.to(device)
    print(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer =  torch.optim.SGD(params=model_1.parameters(), lr=0.1)

    train_time_on_cpu = timer()

    epochs = 3

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        train_step(data_loader=train_dataloader,
            model=model_1,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn
        )
        eval_model(data_loader=test_dataloader,
            model=model_1,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn
        )

    
    train_time_end_on_cpu = timer()
    total_train_time_model_0 = print_train_time(start=train_time_on_cpu,
                                           end=train_time_end_on_cpu,
                                           device=str(next(model_1.parameters()).device))

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = input_shape, out_features  = hidden_units),
            nn.Linear(in_features= hidden_units, out_features = output_shape)
        )
    def forward(self, x):
        return self.layer_stack(x)

class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = input_shape, out_features  = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features= hidden_units, out_features = output_shape),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layer_stack(x)

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start 
    print(f"Train time one {device} : {total_time:.3f} seconds")
    return total_time


     

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y =X.to(device), y.to(device)# Make predictions with the model
            y_pred = model(X)

            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)

        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

        print(f"Test loss: {loss:.5f} | Test accuracy: {acc:.2f}%\n")

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")



main()
 