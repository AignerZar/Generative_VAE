import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import config # hier weg mit dem 
from graph_h2o import build_edge_index
from model import VAE, train, plot_loss

# reproducibility
torch.manual_seed(config.seed)
np.random.seed(config.seed)
"""
aufpassen wenn ich was lade usw. hardcoding usw aufpassen, diensionen aufpassen
"""
P = config.P
num_atoms = config.num_atoms

data = np.loadtxt(config.input_file, delimiter=",", dtype=np.float32)

if data.shape[1] != config.input_dim:
    raise ValueError("Wrong sizes of input dimension and input file.")

train_raw, val_raw = train_test_split(data, test_size=config.validation_split, random_state=config.seed, shuffle=True)

# achtung beim spit immer eine ganze configuration nicht einfach random splitten
# splitten immer erst nach einer vollen configuration
mean = train_raw.mean(axis=0)
std = train_raw.std(axis=0)
std = np.where(std == 0, 1.0, std)

train_norm = (train_raw - mean) / std
val_norm = (val_raw - mean) / std

train_data = torch.tensor(train_norm, dtype=torch.float32)
val_data = torch.tensor(val_norm, dtype=torch.float32)

config.input_dim = data.shape[1]

train_loader = DataLoader(TensorDataset(train_data), batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data), batch_size=config.batch_size, shuffle=False)

edge_index = build_edge_index(P, num_atoms).to(config.device)

model = VAE(latent_dim=config.input_dim, P=P, num_atoms=num_atoms, edge_index=edge_index).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

loss_history = train(model, train_loader, val_loader, optimizer, config)
torch.save(model.state_dict(), "vae_h2o_30Beads.pt")

plot_loss(loss_history, outfile="loss_plot_30Beads.pdf")    # adjusting all the names and storing everything separatedly 


"""
Achtung bei normalisierung aufpassen dass ich physikalisch bleibe 
"""