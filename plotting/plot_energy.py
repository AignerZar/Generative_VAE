"""
File for plotting the energy distribution of the data, once for the original data, once for the reconstruted data and once for the generated data
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn.functional as F

def plot_energy_distributions(E_true, E_rec, E_gen, outfile="energy_distributions.pdf"):
    plt.figure(figsize=(10,6))

    plt.hist(E_true, bins=100, alpha=0.6, label=r"Original", density=True)
    plt.hist(E_rec, bins=100, alpha=0.6, label=r"Reconstructed", density=True)
    plt.hist(E_gen, bins=100, alpha=0.6, label=r"Generated", density=True)

    plt.xlabel(r"Energy (a.u.)")
    plt.ylabel(r"Probability Density")
    plt.xlim(0.0, 0.1)
    plt.title(r"Energy Distribution Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()
