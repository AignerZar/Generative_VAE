"""
Main file to run the code -> imoritng parts from different files
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import config
from graph.H2O_graph import build_edge_index
from model.VAE import VAE
from train.train import train
from plotting.plot_loss import plot_loss

# reproducibility
torch.manual_seed(config.seed)
np.random.seed(config.seed)

P = config.P
num_atoms = config.num_atoms

data_flat = np.loadtxt(config.input_file, delimiter=",", dtype=np.float32)
mean = data_flat.mean(axis=0)
std = data_flat.std(axis=0)
data_norm = (data_flat - mean) / std
data_tensor = torch.tensor(data_norm, dtype=torch.float32)

config.input_dim = data_tensor.shape[1]

train_data, val_data = train_test_split(data_tensor, test_size=config.validation_split, random_state=config.seed)
train_loader = DataLoader(TensorDataset(train_data), batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data), batch_size=config.batch_size)

edge_index = build_edge_index(P, num_atoms).to(config.device)

model = VAE(config.input_dim, config.latent_dimension, P, num_atoms, edge_index).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

loss_history = train(model, train_loader, val_loader, optimizer, config)
#train(model, train_loader, val_loader, optimizer, config)
torch.save(model.state_dict(), "vae_h2o.pt")

plot_loss(loss_history, outfile="loss_plot.pdf")