"""
File tp define the training process
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

from train.loss import vae_loss

def train(model, train_loader, val_loader, optimizer, config):
    loss_history = []
    for epoch in range(config.n_epochs):
        model.train()
        total_loss = total_recon = total_kl = 0

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(config.device)
            optimizer.zero_grad()

            x_hat, mu, logvar = model(x_batch)
            loss, recon, kl = vae_loss(x_batch, x_hat, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (x_batch,) in val_loader:
                x_batch = x_batch.to(config.device)
                x_hat, mu, logvar = model(x_batch)
                loss, _, _ = vae_loss(x_batch, x_hat, mu, logvar)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1:03d}: TrainLoss={avg_loss:.3f}, ValLoss={val_loss:.3f}")
        loss_history.append((avg_loss, val_loss))

    return loss_history