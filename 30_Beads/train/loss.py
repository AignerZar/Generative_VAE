"""
File to compute the loss
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
#import config

# definition of the loss function    
#def vae_loss(x, x_hat, mu, logvar, beta=0.0):   # maybe adjusting beta further
#    x_hat = x_hat.view(x.size(0), 1, 36)     # 9 should be 6 for 2 atoms
#    x = x.view(x.size(0), 1, 36)             # 9 should be 6 for 2 atoms

#    recon_loss = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
#    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0) #divinding for loss per batch
#    return recon_loss + beta * kl_div, recon_loss, kl_div

# here I also want to include a loss dependent on the geometry of the data -> bond lengths and bond angles
# computing the geometry
def h2o_geometry(x, P, num_atoms):
    B = x.size(0)

    if x.dim() == 2:
        x = x.view(B, P, num_atoms, 3)
    elif x.dim() != 4:
        raise ValueError("Wrong x shape must be different!")

    H1 = x[:, :, 0, :]
    O = x[:, :, 1, :]
    H2 = x[:, :, 2, :]

    v1 = H1 - O
    v2 = H2 - O

    bond1 = torch.norm(v1, dim=-1)
    bond2 = torch.norm(v2, dim=-1)

    dot = torch.sum(v1 * v2, dim=-1)
    norm1 = torch.norm(v1, dim=-1)
    norm2 = torch.norm(v2, dim=-1)

    cos_angle = dot / (norm1 * norm2 + 1e-8)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    angle = torch.acos(cos_angle)

    return bond1, bond2, angle

# now computing the loss of the bond and the angle
def geometry_loss(x, x_hat, P, num_atoms):
    bond1_true, bond2_true, angle_true = h2o_geometry(x, P, num_atoms)
    bond1_hat, bond2_hat, angle_hat = h2o_geometry(x_hat, P, num_atoms)

    bond_loss = (
        F.mse_loss(bond1_hat, bond1_true) +
        F.mse_loss(bond2_hat, bond2_true)
    )

    angle_loss = F.mse_loss(angle_hat, angle_true)

    return bond_loss, angle_loss

def vae_loss(
    x, x_hat, mu, logvar,
    P, num_atoms,
    beta=0.5,
    lambda_bond=0.0,
    lambda_angle=0.0
):
    B = x.size(0)

    recon_loss = F.mse_loss(x_hat, x, reduction='sum') / B
    total_loss = recon_loss

    device = x.device

    kl_div = torch.tensor(0.0, device=device)
    bond_loss = torch.tensor(0.0, device=device)
    angle_loss = torch.tensor(0.0, device=device)

    if beta > 0.0:
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B 
        total_loss = total_loss + beta * kl_div

    if lambda_bond > 0.0 or lambda_angle > 0.0:
        bond_loss, angle_loss = geometry_loss(x, x_hat, P, num_atoms)

        if lambda_bond > 0.0:
            total_loss = total_loss + lambda_bond * bond_loss

        if lambda_angle > 0.0:
            total_loss = total_loss * lambda_angle * angle_loss
    
    return total_loss, recon_loss, kl_div, bond_loss, angle_loss