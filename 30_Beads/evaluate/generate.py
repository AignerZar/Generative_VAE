"""
File to generate new data
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

def sample_from_aggregated_posterior(mu_all, n_samples, device):
    # Zufällige mu aus dem gesamten Datensatz auswählen
    idx = torch.randint(0, mu_all.size(0), (n_samples,))
    z = mu_all[idx].to(device)
    return z

def generate_from_latent(model, z, mean, std, device):
    """
    Generate real-space positions from latent vectors z.
    """
    model.eval()
    with torch.no_grad():
        x_gen = model.decoder(z.to(device)).cpu().numpy()

    # denormalize
    x_denorm = x_gen * std + mean
    return x_gen, x_denorm