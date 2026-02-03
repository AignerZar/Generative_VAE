"""
File for the EGCL class 
EGCL is an equivariant graph convolution layer with is äquivariant again transformation of the euclidean group
-> translation, rotation and symmetry
therefore often used in physics as the physical properties remain correct during training
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

class EGCL(nn.Module):
     # E(n) invariant CNN 
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.phi_e = nn.Sequential(
            nn.Linear(2 * in_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.phi_h = nn.Sequential(
            nn.Linear(hidden_dim, in_dim),
            nn.ReLU()
        )
    
    def forward(self, h, pos, edge_index):
        src, dst = edge_index

        h_src = h[:, src, :] # start node
        h_dst = h[:, dst, :] # goal nide

        diff = pos[:, src, :] - pos[:, dst, :] # distance between two nodes
        dist2 = (diff ** 2).sum(-1, keepdim=True)

        m_ij = self.phi_e(torch.cat([h_src, h_dst, dist2], dim=-1)) 

        B, E, H = m_ij.shape
        N = h.size(1)

        # graph convolutional layer
        m_agg = torch.zeros(B, N, H, device=h.device)
        index = dst.view(1, E, 1).expand(B, E, H)
        m_agg.scatter_add_(1, index, m_ij)

        # update the node features
        h_out = h + self.phi_h(m_agg)
        #pos = pos + 0.01 * m_agg[:, :, :3]
        return h_out, pos
    

# EGCL for decoder
class EGCL_decoder(nn.Module):
    """
    EGCL for the decoder -> needs to be used for generating 
    New coordinates need to be updated with each step -> no static coordinates
    Inspired by Github: https://github.com/oriondollar/vagrant_en/blob/main/vagrant/gcl.py
    Seperate definition of Nodes and Edges
    """
    # E(n) invariant CNN 
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * feat_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        # Node MLP 
        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feat_dim)
        )

        # Coord update MLP
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()     # trying different fucntions maybe sigmoid ?
        )
    
    def forward(self, h, pos, edge_index):
        src, dst = edge_index

        h_src = h[:, src, :] # features of the nodes
        h_dst = h[:, dst, :] # resulting node

        diff = pos[:, src, :] - pos[:, dst, :]
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-8
        dist2 = dist**2

        m_ij = self.edge_mlp(torch.cat([h_src, h_dst, dist2], dim=-1))


        B, N, F = h.shape
        _, E, H = m_ij.shape
        #N = h.size(1)

        # graph convolutional layer
        m_agg = torch.zeros(B, N, H, device=h.device, dtype=h.dtype)
        index = dst.view(1, E, 1).expand(B, E, H)
        m_agg.scatter_add_(1, index, m_ij)

        h_new = h + self.node_mlp(torch.cat([h, m_agg], dim=-1))

        w = self.coord_mlp(m_ij)

        direction = diff / dist

        coord_update = w * direction * 0.1     
        #coord_update = torch.zeros_like(pos)  # (B, N, 3)
        #update_values = torch.tanh(w * diff * 0.1)            # (B, E, 3)

        update = torch.zeros_like(pos)
        index_coord = dst.view(1, E, 1).expand_as(coord_update)
        update.scatter_add_(1, index_coord, coord_update)

        #index_coord = dst.view(1, E, 1).expand_as(update_values)
        #coord_update.scatter_add_(1, index_coord, update_values)

        x_new = pos + update

        return h_new, x_new
    


# EGCL for encoder
class EGCL_encoder(nn.Module):
   # E(n) invariant CNN 
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.phi_e = nn.Sequential(
            nn.Linear(2 * in_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.phi_h = nn.Sequential(
            nn.Linear(hidden_dim, in_dim),
            nn.ReLU()
        )
    
    def forward(self, h, pos, edge_index):
        src, dst = edge_index

        h_src = h[:, src, :] # features of the nodes
        h_dst = h[:, dst, :] # resulting node

        diff = pos[:, src, :] - pos[:, dst, :] # distance between two nodes
        dist2 = (diff ** 2).sum(-1, keepdim=True)

        m_ij = self.phi_e(torch.cat([h_src, h_dst, dist2], dim=-1))

        B, E, H = m_ij.shape
        N = h.size(1)

        # graph convolutional layer
        m_agg = torch.zeros(B, N, H, device=h.device)
        index = dst.view(1, E, 1).expand(B, E, H)
        m_agg.scatter_add_(1, index, m_ij)

        # update the node features
        h_out = h + self.phi_h(m_agg)
        return h_out
