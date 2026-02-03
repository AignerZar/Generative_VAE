"""
File to plot the loss function
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn.functional as F
import re
from sklearn.model_selection import train_test_split

"""
Plotting the training and validation loss course------------------------------------------------------------------------------------------------------------------------------------
"""
def plot_loss(loss_history, outfile="loss.pdf"):
    loss_history = np.array(loss_history)
    plt.figure(figsize=(8,4))
    plt.plot(loss_history[:,0], label=r"Train Loss")
    plt.plot(loss_history[:,1], label=r"Val Loss")
    plt.legend()
    plt.xlabel(r"Epoch")
    plt.ylabel(r"Loss")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()
