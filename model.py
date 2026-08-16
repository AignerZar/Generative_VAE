"""
The following code contains the architecture of the VAE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from graph_h2o import build_edge_index, build_node_features

############################## E(n)-equivariant Graph Convolution Layers ########################################
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
        h_dst = h[:, dst, :] # goal node

        diff = pos[:, src, :] - pos[:, dst, :] # distance between two nodes
        dist2 = (diff ** 2).sum(-1, keepdim=True)

        m_ij = self.phi_e(torch.cat([h_src, h_dst, dist2], dim=-1)) 

        B, E, H = m_ij.shape
        N = h.size(1)

        # graph convolutional layer
        m_agg = torch.zeros(B, N, H, device=h.device) #initialization with zeros
        index = dst.view(1, E, 1).expand(B, E, H)   
        m_agg.scatter_add_(1, index, m_ij)

        # update the node features
        h_out = h + self.phi_h(m_agg)
        #pos = pos + 0.01 * m_agg[:, :, :3]
        return h_out, pos
    


############################## Encoder Class ########################################
class Encoder(nn.Module):
    def __init__(
            self, 
            latent_dim, 
            P, 
            num_atoms, 
            edge_index, 
            node_feat_dim=3, 
            hidden_dim=64, 
            num_layers=6
        ): 
            super().__init__()
            self.P = P
            self.num_atoms = num_atoms
            self.node_feat_dim = hidden_dim
            self.register_buffer("edge_index", edge_index)

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

        for i, layer in enumerate(self.layers):
            h, pos = layer(h, pos, self.edge_index)

           

        h_mol = h.mean(dim=1)

        mu = self.fc_mu(h_mol)
        logvar = self.fc_logvar(h_mol)

        return mu, logvar


################################# Decoder Class ########################################
class Decoder(nn.Module):
    def __init__(
            self, 
            latent_dim, 
            P, 
            num_atoms, 
            edge_index, 
            node_feat_dim=3,
            hidden_dim=64, 
            num_layers=6
        ):
            super().__init__()

            self.P = P
            self.num_atoms = num_atoms
            self.node_feat_dim = node_feat_dim
            self.hidden_dim = hidden_dim

            self.register_buffer("edge_index", edge_index)

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
        h = self.node_emb(node_features) + g                                  # (B,N,H)

        pos = torch.zeros(B, self.N, 1, device=device)

        for egcl, ln in zip(self.layers, self.norms):
            h_new, pos = egcl(h, pos, self.edge_index) 
            h = ln(h_new)

        pos_hat = self.fc_out(h)               # (B,N,3)
        return pos_hat.view(B, -1)


################################## VAE Class ########################################
class VAE(nn.Module):
    def __init__(
            self, 
            input_dim, 
            latent_dim, 
            P, 
            num_atoms, 
            edge_index,
            node_feat_dim=3,
            hidden_dim=64,
            num_layers=6,
        ):
            super().__init__()

            self.encoder = Encoder(
                latent_dim=latent_dim,
                P=P,
                num_atoms=num_atoms,
                edge_index=edge_index,
                node_feat_dim=node_feat_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )

            self.decoder = Decoder(
                latent_dim=latent_dim,
                P=P,
                num_atoms=num_atoms,
                edge_index=edge_index,
                node_feat_dim=node_feat_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            )
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


################################## Loss Function ########################################
def vae_loss(
        x, x_hat, mu, logvar, beta
):
    B = x.size(0)
    device = x.device

    # reconstruction loss
    recon_loss = F.mse_loss(x_hat, x, reduction='sum') / B 

    # reparametrization loss
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    total_loss = recon_loss + beta * kl_div

    return total_loss, recon_loss, kl_div


################################## Train Function ##########################################
def train(model, train_loader, val_loader, optimizer, config):
    loss_history = []
    for epoch in range(config.n_epochs):
        model.train()
        total_loss = total_recon = total_kl = 0

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(config.device)
            optimizer.zero_grad()

            x_hat, mu, logvar = model(x_batch)
            loss, recon_loss, kl_div = vae_loss(x_batch, x_hat, mu, logvar, beta=0.5)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_div.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (x_batch,) in val_loader:
                x_batch = x_batch.to(config.device)
                x_hat, mu, logvar = model(x_batch)
                loss, _, _ = vae_loss(x_batch, x_hat, mu, logvar, beta=0.5)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1:03d}: TrainLoss={avg_loss:.3f}, ValLoss={val_loss:.3f}")
        loss_history.append((avg_loss, val_loss))

    return loss_history

##################################### Function to plot the loss function ########################################################
def plot_loss(loss_history, outfile="loss_30Beads.pdf"):
    loss_history = np.array(loss_history)
    plt.figure(figsize=(10,5))
    plt.plot(loss_history[:,1], label=r"Validation Loss", linewidth=0.8)
    plt.plot(loss_history[:,0], label=r"Training Loss", linewidth=0.8)
    plt.xlabel(r"Epoch", fontsize=14)
    plt.ylabel(r"Loss", fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=13, loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()
