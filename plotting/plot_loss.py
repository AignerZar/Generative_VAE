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
def plot_loss(loss_history, outfile="loss_5000MC.pdf"):
    loss_history = np.array(loss_history)
    plt.figure(figsize=(10,5))
    plt.plot(loss_history[:,1], label=r"Validation Loss", linewidth=0.8)
    plt.plot(loss_history[:,0], label=r"Training Loss", linewidth=0.8)
    #plt.plot(loss_history[:,1], label=r"Val Loss")
    plt.xlabel(r"Epoch", fontsize=14)
    plt.ylabel(r"Loss", fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=13, loc="best", frameon=True)
    #plt.xlabel(r"Epoch", fontsize=14)
    #plt.ylabel(r"Loss", fontsize=12)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()
