"""
File to compiute the geoemtetry of the molecule -> bond lengths and angles
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
from scipy.stats import gaussian_kde

def compute_bond_lengths(x):
    """
    x shape: (P, 3, 3)
    returns two arrays:
        r1: O-H1 distances
        r2: O-H2 distances
    """
    O  = x[:,1,:]
    H1 = x[:,0,:]
    H2 = x[:,2,:]

    r1 = np.linalg.norm(O - H1, axis=1)
    r2 = np.linalg.norm(O - H2, axis=1)
    return r1, r2


def compute_angle(a, b, c):
    """
    Computes angle ABC, in radians
    a,b,c: (3,)
    """
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
    return np.arccos(np.clip(cosang, -1.0, 1.0))


def compute_angle_distribution(x):
    """
    x shape: (P, 3, 3)
    Returns: list of angles (in degrees)
    """
    O  = x[:,1,:]
    H1 = x[:,0,:]
    H2 = x[:,2,:]

    angles = []
    for i in range(len(O)):
        ang = compute_angle(H1[i], O[i], H2[i])
        angles.append(np.degrees(ang))
    return np.array(angles)

def get_distributions(data_tensor, P, num_atoms, model=None, mode="original", n_samples=500, device="cpu", denorm_func=None):
    bond1 = []
    bond2 = []
    angles = []

    if mode == "original":
        iterator = range(len(data_tensor))
        def get_x(i):
            x = denorm_func(data_tensor[i].cpu().numpy())
            return x.reshape(P, num_atoms, 3)

    elif mode == "reconstructed":
        model.eval()
        iterator = range(len(data_tensor))
        def get_x(i):
            with torch.no_grad():
                x_flat = data_tensor[i].unsqueeze(0).to(device)
                x_hat, _, _ = model(x_flat)
                x = denorm_func(x_hat.squeeze(0).cpu().numpy())
                return x.reshape(P, num_atoms, 3)

    elif mode == "generated":
        model.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, model.encoder.fc_mu.out_features).to(device)
            x_gen = model.decoder(z).cpu().numpy()

        iterator = range(n_samples)
        def get_x(i):
            x = denorm_func(x_gen[i])
            return x.reshape(P, num_atoms, 3)

    for i in iterator:
        x = get_x(i)
        r1, r2 = compute_bond_lengths(x)
        bond1.extend(r1)
        bond2.extend(r2)
        angles.extend(compute_angle_distribution(x))

    return np.array(bond1), np.array(bond2), np.array(angles)

