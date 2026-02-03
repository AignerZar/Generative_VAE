"""
Main file to evalute the VAE, before executing that code the VAE must be trained, therefore run main_train.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn.functional as F
import config

# importing all classes and functions from the files
from model.VAE import VAE
from graph.H2O_graph import build_edge_index
from evaluate.latent import compute_latent
from evaluate.generate import sample_from_aggregated_posterior, generate_from_latent
from evaluate.energy import (
    compute_energy_distribution_original,
    compute_energy_distribution_generated,
    compute_energy_distribution_reconstructed
)
from evaluate.geometry import get_distributions
from plotting.plot_energy import plot_energy_distributions
from plotting.plot_geometry import (
    plot_bond_angle_distributions, 
    plot_bond_angle_distributions_with_kde, 
    print_distribution_means, 
    plot_bond_angle_distributions_mix, 
    print_distribution_stats,
    plot_bond_angle_kde_only
)
# loading the data
data_flat = np.loadtxt(config.input_file, delimiter=",", dtype=np.float32)

# normalization
mean = data_flat.mean(axis=0)
std = data_flat.std(axis=0)
data_norm = (data_flat - mean) / std

# converting to üpytorch tensor
data_tensor = torch.tensor(data_norm, dtype=torch.float32)#.to(config.device)

P = config.P
num_atoms = config.num_atoms

def denorm_flat(x_flat):
    return x_flat * std + mean

# loadint the model
edge_index = build_edge_index(P, num_atoms).to(config.device)

model = VAE(config.input_dim, config.latent_dimension, P, num_atoms, edge_index).to(config.device)
model.load_state_dict(torch.load("vae_h2o.pt", map_location=config.device))
model.eval()

# latent space
mu, logvar, z = compute_latent(model, data_tensor, config.device)
mu_tensor = torch.tensor(mu, dtype=torch.float32)

np.savetxt("latent_mu.csv", mu, delimiter=",")

# generate the structure
z_samples = sample_from_aggregated_posterior(mu_tensor, config.num_samples, config.device)
x_gen_norm, x_gen_denorm = generate_from_latent(model, z_samples, mean, std, config.device)

np.savetxt("generated_norm.csv", x_gen_norm, delimiter=",")
np.savetxt("generated_denorm.csv", x_gen_denorm, delimiter=",")

# energy distribution
E_true = compute_energy_distribution_original(data_tensor, P, num_atoms, denorm_flat)
E_rec  = compute_energy_distribution_reconstructed(model, data_tensor, P, num_atoms, config.device, denorm_flat)
E_gen  = compute_energy_distribution_generated(model, 500, config.latent_dimension, P, num_atoms, config.device, denorm_flat)

plot_energy_distributions(E_true, E_rec, E_gen)

# geometries
dist_original = get_distributions(data_tensor, P, num_atoms, mode="original", denorm_func=denorm_flat)
dist_recon    = get_distributions(data_tensor, P, num_atoms, model=model, mode="reconstructed", device=config.device, denorm_func=denorm_flat)
dist_generated = get_distributions(data_tensor, P, num_atoms, model=model, mode="generated", device=config.device, n_samples=500, denorm_func=denorm_flat)

plot_bond_angle_distributions(dist_original, dist_recon, dist_generated)
plot_bond_angle_distributions_with_kde(dist_original, dist_recon, dist_generated)
plot_bond_angle_distributions_mix(dist_original, dist_recon, dist_generated)
print_distribution_means(dist_original, dist_recon, dist_generated)

print_distribution_stats(dist_original, dist_recon, dist_generated, ddof=1, robust=True)

plot_bond_angle_kde_only(dist_original, dist_recon, dist_generated, bw_method="scott")

def width_table(dist_original, dist_recon, dist_generated, ddof=1):
    names = ["O-H(1)", "O-H(2)", "H-O-H"]
    sets = [("Original", dist_original), ("Reconstructed", dist_recon), ("Generated", dist_generated)]

    print("\n=== Width comparison (std and IQR) ===\n")
    print(f"{'Quantity':10s} | {'Set':15s} | {'std':>10s} | {'IQR':>10s}")
    print("-"*55)

    for i, q in enumerate(names):
        for label, dist in sets:
            x = np.asarray(dist[i])
            std = np.std(x, ddof=ddof)
            q25, q75 = np.percentile(x, [25, 75])
            iqr = q75 - q25
            print(f"{q:10s} | {label:15s} | {std:10.5f} | {iqr:10.5f}")
        print("-"*55)
