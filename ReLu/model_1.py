import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from pathlib import Path
from sklearn.datasets import make_moons

from helper_functions import accuracy_fn
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary


def main():
    samples = 1000
    random_state = 42
    
    X, y = make_moons(n_samples= samples, noise= 0.03, random_state=random_state)
    print(X)
    print(y)
    print(X.shape, y.shape)
    moons = pd.DataFrame({"X": X[:,0], "y": X[:,1], "label": y})
    print(moons.head(10))
    plt.scatter(x=X[:, 0],
                y=X[:, 1],
                c=y,
                cmap=plt.cm.RdYlBu)
    plt.show()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    model_gpu = model1().to(device)
    train_model(model=model_gpu, X_train=X_train, y_train=y_train, y_test=y_test, X_test=X_test, epochs=1000, lr=0.01)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model_gpu, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model_gpu, X_test, y_test)
    plt.show()

    print("Training complete")

    


class model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=2, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

def train_model(model, X_train, y_train, y_test, X_test, epochs=1000, lr= 0.01):
    
    torch.manual_seed(42)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        y_logits = model(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
        loss = loss_fn(y_logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
        if epoch % 10 == 0:
                    print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%")





if __name__ == "__main__":
    main()