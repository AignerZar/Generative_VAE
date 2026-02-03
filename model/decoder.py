"""
File to define the Decoder of the VAE
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

from model.EGCL import EGCL, EGCL_decoder #_decoder
from graph.H2O_graph import build_node_features


class Decoder(nn.Module):
    """
    EGNN-basierter Decoder, stilistisch angelehnt an Vagrant:
    - deconv(z) erzeugt ein globales Conditioning
    - node_emb bettet Node-Features ein
    - mehrere EGCL_decoder-Schichten updaten Features & Koordinaten
    """
    def __init__(
        self,
        latent_dim,
        P,
        num_atoms,
        edge_index,
        node_feat_dim=3,   # z.B. 2 (one-hot) + 1 (bead index)
        hidden_dim=64,
        num_layers=16       # lieber klein und stabil starten
    ):
        super().__init__()
        self.P = P
        self.num_atoms = num_atoms
        self.edge_index = edge_index   # (2, E) LongTensor

        self.N = P * num_atoms         # Anzahl Nodes = 15 * 3 = 45

        # z → globales Conditioning (analog zu Vagrant.deconv)
        self.deconv = nn.Linear(latent_dim, hidden_dim)

        # Node-Feature-Embedding (analog zu seq_emb ohne PosEnc)
        self.node_emb = nn.Linear(node_feat_dim, hidden_dim)

        # EGCL-Decoder-Schichten
        self.layers = nn.ModuleList([
            EGCL(hidden_dim, hidden_dim) for _ in range(num_layers)
            #EGCL_decoder(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Learnable initial positions (aber stabil!)
        self.initial_pos = nn.Parameter(torch.zeros(1, self.N, 3))

    def forward(self, z):
        """
        z: (B, latent_dim)
        Rückgabe: pos: (B, N, 3)
        """
        B = z.size(0)
        device = z.device

        # ---- (1) globale Info aus z (analog zu deconv + Broadcast) ----
        z_cond = self.deconv(z)              # (B, hidden_dim)
        z_cond = z_cond.unsqueeze(1).repeat(1, self.N, 1)  # (B, N, hidden_dim)

        # ---- (2) Node-Features aufbauen (wie bei dir) ----
        node_features = build_node_features(
            batch_size=B,
            P=self.P,
            num_atoms=self.num_atoms,
            device=device
        )    # (B, N, node_feat_dim)

        # Optional: Node-Features etwas skalieren, damit sie z nicht dominieren
        # node_features = node_features * 0.1

        # ---- (3) Node-Embedding + Conditioning (analog zu y + z in Vagrant.decode) ----
        h0 = self.node_emb(node_features)    # (B, N, hidden_dim)
        h = h0 + z_cond                      # (B, N, hidden_dim)

        # ---- (4) Initial-Koordinaten ----
        pos = self.initial_pos.expand(B, -1, -1).clone()  # (B, N, 3)

        # ---- (5) EGCL-Decoder-Stack (analog Trans-Schleife) ----
        for layer in self.layers:
            #h = layer(h)
            h, pos = layer(h, pos, self.edge_index)

        # ---- (6) Rückgabe der finalen Koordinaten ----
        return pos.view(pos.size(0), -1)  # ggf. in Training: pos.view(B, -1) mit target matchen

        
# definition of the decoder class    
class Decoder_new(nn.Module):
    def __init__(self, latent_dim, P, num_atoms, edge_index, node_feat_dim=3,
                 hidden_dim=64, num_layers=4):
        super().__init__()
        self.P = P
        self.num_atoms = num_atoms
        self.N = P * num_atoms
        self.edge_index = edge_index

        self.fc_global = nn.Linear(latent_dim, hidden_dim)
        self.node_emb  = nn.Linear(node_feat_dim, hidden_dim)

        self.layers = nn.ModuleList([EGCL(hidden_dim, hidden_dim) for _ in range(num_layers)])

        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, z):
        B = z.size(0)
        device = z.device

        node_features = build_node_features(B, self.P, self.num_atoms, device)  # (B,N,3)

        g = self.fc_global(z).unsqueeze(1).repeat(1, self.N, 1)                # (B,N,H)
        h = self.node_emb(node_features) + g                                   # (B,N,H)

        # Dummy pos nur für dist2, aber fix (kleines noise ist ok)
        pos = torch.zeros(B, self.N, 3, device=device)

        for egcl, ln in zip(self.layers, self.norms):
            h_new, pos = egcl(h, pos, self.edge_index)  # pos bleibt gleich in deinem EGCL
            h = ln(h_new)

        pos_hat = self.fc_out(h)               # (B,N,3)
        return pos_hat.view(B, -1)
