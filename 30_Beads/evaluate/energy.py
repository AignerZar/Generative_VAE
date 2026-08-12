"""
File to evalutate the ernergy
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

def morse_potential(r, D=0.20, alpha=2.0, r0=1.0):
    return D * (1 - np.exp(-alpha * (r - r0)))**2


def compute_energy_sample(x):
    # x: (P, num_atoms=3, 3)
    O  = x[:, 1, :]
    H1 = x[:, 0, :]
    H2 = x[:, 2, :]

    r1 = np.linalg.norm(O - H1, axis=1)
    r2 = np.linalg.norm(O - H2, axis=1)

    E1 = morse_potential(r1)
    E2 = morse_potential(r2)

    return np.mean(E1 + E2)


def compute_energy_distribution_original(data_tensor, P, num_atoms, denorm_func):
    energies = []
    for i in range(len(data_tensor)):
        x = denorm_func(data_tensor[i].cpu().numpy())
        x = x.reshape(P, num_atoms, 3)
        energies.append(compute_energy_sample(x))
    return np.array(energies)


def compute_energy_distribution_reconstructed(model, data_tensor, P, num_atoms, device, denorm_func):
    energies = []
    model.eval()
    with torch.no_grad():
        for i in range(len(data_tensor)):
            x = data_tensor[i].unsqueeze(0).to(device)
            x_hat, _, _ = model(x)
            x_hat = x_hat.squeeze(0).cpu().numpy()
            x_hat = denorm_func(x_hat)
            x_hat = x_hat.reshape(P, num_atoms, 3)
            energies.append(compute_energy_sample(x_hat))
    return np.array(energies)


def compute_energy_distribution_generated(model, n_samples, latent_dim, P, num_atoms, device, denorm_func):
    z = torch.randn(n_samples, latent_dim).to(device)
    model.eval()
    with torch.no_grad():
        x_gen = model.decoder(z).cpu().numpy()

    energies = []
    for i in range(n_samples):
        x = denorm_func(x_gen[i]).reshape(P, num_atoms, 3)
        energies.append(compute_energy_sample(x))
    return np.array(energies)
