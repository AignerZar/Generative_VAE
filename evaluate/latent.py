"""
File for latent space computation
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

def compute_latent(model, data_tensor, device):
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encoder(data_tensor.to(device))
        z = model.reparameterize(mu, logvar)

    return mu.cpu().numpy(), logvar.cpu().numpy(), z.cpu().numpy()