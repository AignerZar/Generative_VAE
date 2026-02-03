"""
File for the completer VAE consisting of Decoder and Encoder, setting everything together
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

from model.encoder import Encoder
from model.decoder import Decoder_new, Decoder

# definition VAE
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, P, num_atoms, edge_index):
        super().__init__()
        self.encoder = Encoder(latent_dim, P, num_atoms, edge_index)
        self.decoder = Decoder_new(latent_dim=latent_dim, P=15, num_atoms=3, edge_index=edge_index, hidden_dim=64, num_layers=4)

    # reparametrize function to make it trainable with the backpropagation  
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # random value
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar