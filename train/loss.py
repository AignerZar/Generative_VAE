"""
File to compute the loss
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

# definition of the loss function    
def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    x_hat = x_hat.view(x.size(0), 15, 9)
    x = x.view(x.size(0), 15, 9)
    recon_loss = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0) #divinding for loss per batch
    return recon_loss + beta * kl_div, recon_loss, kl_div