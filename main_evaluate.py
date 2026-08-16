import torch
import numpy as np
import config
from graph_h2o import build_edge_index
from model import VAE
from sampling import compute_latent, generate_from_latent, sample_from_aggregated_posterior
from evaluate_results import get_distributions

# loading the data
data_flat = np.loadtxt(config.input_file, delimiter=",", dtype=np.float32)

# normalization
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

model = VAE(config.input_dim, config.latent_dimension, P, num_atoms, edge_index).to(config.device)
model.load_state_dict(torch.load("vae_h2o_30Beads_new.pt", map_location=config.device))
model.eval()

# latent space
mu, logvar, z = compute_latent(model, data_tensor, config.device)
mu_tensor = torch.tensor(mu, dtype=torch.float32)

np.savetxt("latent_mu_30Beads.csv", mu, delimiter=",")

# generate the structure
z_samples = sample_from_aggregated_posterior(mu_tensor, config.num_samples, config.device)
x_gen_norm, x_gen_denorm = generate_from_latent(model, z_samples, mean, std, config.device)

np.savetxt("generated_norm_30Beads.csv", x_gen_norm, delimiter=",")
np.savetxt("generated_denorm_30Beads.csv", x_gen_denorm, delimiter=",")

dist_original = get_distributions(data_tensor, P, num_atoms, mode="original", denorm_func=denorm_flat)
dist_recon    = get_distributions(data_tensor, P, num_atoms, model=model, mode="reconstructed", device=config.device, denorm_func=denorm_flat)
dist_generated = get_distributions(data_tensor, P, num_atoms, model=model, mode="generated", device=config.device, n_samples=2000, denorm_func=denorm_flat)


############################################ Function to print the results ###################################################################
def summary_table(dist_original, dist_recon, dist_generated, ddof=1):
    quantities = ["O-H(1)", "O-H(2)", "H-O-H"]
    datasets = [
        ("Original", dist_original),
        ("Reconstructed", dist_recon),
        ("Generated", dist_generated),
    ]

    print(f"{'Quantity':10s} {'Dataset':15s} {'Mean':>10s} {'Std':>10s}")
    print("-" * 49)

    for i, quantity in enumerate(quantities):
        for label, distributions in datasets:
            values = np.asarray(distributions[i])

            mean = np.mean(values)
            std = np.std(values, ddof=ddof)

            print(
                f"{quantity:10s} "
                f"{label:15s} "
                f"{mean:10.5f} "
                f"{std:10.5f}"
            )