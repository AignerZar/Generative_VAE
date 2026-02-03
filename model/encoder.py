"""
File to define the encoder of the VAE
here the EGCL is used in the encoder, meaning the properties are remained, same later used for the decoder
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

from model.EGCL import EGCL, EGCL_encoder
from graph.H2O_graph import build_node_features

# definition of the encoder class
class Encoder(nn.Module):
    def __init__(self, latent_dim, P, num_atoms, edge_index, node_feat_dim=3, hidden_dim=64, num_layers=4): 
        
        super(Encoder, self).__init__()

        self.P = P
        self.num_atoms = num_atoms
        self.edge_index = edge_index

        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        self.layers = nn.ModuleList([
            EGCL(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x_flat):
        B = x_flat.size(0)
        pos = x_flat.view(B, self.P * self.num_atoms, 3)

        h = build_node_features(B, self.P, self.num_atoms, x_flat.device)
        h = self.input_proj(h)                     

        for layer in self.layers:
            h, pos = layer(h, pos, self.edge_index)

        h_mol = h.mean(dim=1)

        mu = self.fc_mu(h_mol)
        logvar = self.fc_logvar(h_mol)
        return mu, logvar
