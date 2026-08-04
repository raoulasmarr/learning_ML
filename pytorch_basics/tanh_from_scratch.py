import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from torch import nn
import math
import requests
from pathlib import Path

def main():
    x = torch.arange(-10,10,1)
    print(x)
    
    plt.plot(x, hyperbolic_tangent(x)) 
    plt.axhline(0, color='black', linewidth=1)  # Adds a dashed line at y=0
    plt.axvline(0, color='black', linewidth=1)  # Adds a dashed line with slope -1
    plt.xlabel("X Axis Label")  # Labels the horizontal axis
    plt.ylabel("Y Axis Label")
    plt.show()
    

def hyperbolic_tangent(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
main()