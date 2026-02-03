"""
File to define the graph structure of the H2O
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


# die folgende FUnktion wird verwendet um die Graphen Struktur von H2O zu bauen -> je nach Input muss das angepasst werden 
def build_edge_index(P, num_atoms):
    # Bauen der Kanten, wobei es die Kanten (O-H1,H1-O) und die Kanten zwischen den Timesamples gibt
    edges = []

    # Kanten zwischen den Atome des H2O molecules
    for b in range(P):
        H1 = num_atoms * b + 0
        O = num_atoms * b + 1
        H2 = num_atoms * b + 2
        # immer O mit einem der H verbinden aber nie H mit H
        for i, j in [(O, H1), (H1, O), (O, H2), (H2, O)]:
            edges.append((i, j))
    
    # Kanten zwischen den Zeitsamples
    for b in range(P):
        nb = (b + 1) % P # next bead
        for a in range(num_atoms):
            i = num_atoms * b + a
            j = num_atoms * nb + a
            for u, v in [(i, j), (j, i)]:
                edges.append((u, v))
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    return edge_index

def build_node_features(batch_size, P, num_atoms, device):
    # in der funktion werden jedem node eine position (3 koordinaten) zugeordnet und auch die features (vom modell gelernte eigenschaften -> h)
    atom_types = []
    for b in range(P):
        atom_types.extend([0, 1, 1]) #0=O, 1= H
    
    atom_types = torch.tensor(atom_types, dtype=torch.long, device=device)
    atom_onehot = F.one_hot(atom_types, num_classes=2).float()

    bead_indices = []
    for b in range(P):
        bead_indices.extend([b] * num_atoms)
    bead_indices = torch.tensor(bead_indices, dtype=torch.float32, device=device)
    bead_norm = (bead_indices / (P - 1)).unsqueeze(-1)

    base_feat = torch.cat([atom_onehot, bead_norm], dim=-1)

    base_feat = base_feat.unsqueeze(0).expand(batch_size, P * num_atoms, base_feat.size(-1))
    return base_feat
