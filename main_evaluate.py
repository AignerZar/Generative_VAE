import torch
import numpy as np
import config
from graph_h2o import build_edge_index
from model import VAE
from sampling import compute_latent, generate_from_latent, sample_from_aggregated_posterior
from evaluate_results import get_distributions, plot_bond_angle_distributions, summary_table

# loading the data
data_flat = np.loadtxt(config.input_file, delimiter=",", dtype=np.float32)

# normalization -> still under work -> changing this normalization
mean = data_flat.mean(axis=0)
std = data_flat.std(axis=0)
data_norm = (data_flat - mean) / std

# converting to pytorch tensor
data_tensor = torch.tensor(data_norm, dtype=torch.float32)

P = config.P
num_atoms = config.num_atoms

def denorm_flat(x_flat):
    return x_flat * std + mean

# loadint the model
edge_index = build_edge_index(P, num_atoms).to(config.device)

model = VAE(latent_dim=config.latent_dimension, P=P, num_atoms=num_atoms, edge_index=edge_index).to(config.device)
model.load_state_dict(torch.load("vae_h2o_30Beads.pt", map_location=config.device))
model.eval()

# latent space
mu, logvar, z = compute_latent(model=model, data_tensor=data_tensor, device=config.device)
mu_tensor = torch.tensor(mu, dtype=torch.float32)
logvar = torch.tensor(logvar, dtype=torch.float32)

np.savetxt("latent_mu_30Beads.csv", mu, delimiter=",")

# generate the structure
z_samples = sample_from_aggregated_posterior(mu_all=mu_tensor, logvar_all=logvar, n_samples=config.num_samples, device=config.device)
x_gen_norm, x_gen_denorm = generate_from_latent(model=model, z=z_samples, mean=mean, std=std, device=config.device)

np.savetxt("generated_norm_30Beads.csv", x_gen_norm, delimiter=",")
np.savetxt("generated_denorm_30Beads.csv", x_gen_denorm, delimiter=",")

dist_original = get_distributions(data_tensor=data_tensor, P=P, num_atoms=num_atoms, mode="original", n_samples=config.num_samples, denorm_func=denorm_flat)
dist_recon    = get_distributions(data_tensor=data_tensor, P=P, num_atoms=num_atoms, model=model, mode="reconstructed", n_samples=config.num_samples, device=config.device, denorm_func=denorm_flat)
dist_generated = get_distributions(data_tensor=data_tensor, P=P, num_atoms=num_atoms, model=model, mode="generated", n_samples=config.num_samples, device=config.device, denorm_func=denorm_flat)

plot_bond_angle_distributions(dist_original, dist_recon, dist_generated)

summary_table(dist_original=dist_original, dist_recon=dist_recon, dist_generated=dist_generated)