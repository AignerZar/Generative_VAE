"""
The following code contains the architecture of the VAE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from graph_h2o import build_node_features

############################## E(n)-equivariant Graph Convolution Layers ########################################
class EGCL(nn.Module):
    """E(n) Equivariant Graph COnvolutional Layer (EGCL)

    This class and the layers inside process the node features and coordinates while the goal is to preserve the symmetry
    and geometric equivariance based on inter nodal distances

    Args:
        in_dim (int): Input dimension of the node features
        hidden_dim (int): Dimension of the hidden layers used inside the MLP
    """

    def __init__(self, in_dim: int, hidden_dim: int):
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
    
    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        """Massage passing and node-update step

        Args:
            h: Node feature tensor, shape (B, N, in_dim)
            pos: 3D coordinates tensor, shape (B, N, 3)
            edge_index: Tensor with all the edges, shape (2, Edges)

        Returns:
            h_out: Updates node feature tensor, shape (B, N, in_dim)
            pos: Unchanges or updated 3D coordinates
        """
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
    """Encoder class for the Variational Autoencoder (VAE)

    The encoder uses EGCL layers to project PIMC configurations into a latent space. The goal is to reconstruct 3D positions and generate different node features.

    Args:
        latent_dim (int): Dimension of the latent space
        P (int): Number of beads per configuration
        num_atoms (int): Number of atoms per molecule
        edge_index (torch.Tensor): Tensor with all the edges, shape (2, Edges)
        node_feat_dim (int): Number of coordinates per atoms, xyz coordinates
        hidden_dim (int): Dimension of the hidden space
        num_layers (int): Number of EGCL layers inside of the encoder
    """
    def __init__(
            self, 
            latent_dim: int, 
            P: int, 
            num_atoms: int, 
            edge_index: torch.Tensor, 
            node_feat_dim: int = 3, 
            hidden_dim: int = 64, 
            num_layers: int = 6
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

    def forward(self, x_flat: torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        """Mapping the input data to the latent space distribution

        Args:
            x_flat (torch.Tensor): Flattend input data, shape: (B, P * num_atoms * 3)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                mu: Mean vector of the latent distribution, shape (B, latent_dim)
                logvar: Log-variance vector of the latent distribution, shape (B, latent_dim)
        """
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
    """Decoder class for the Variational Autoencoder (VAE).

    Samples from the latent space to produce new PIMC configurations

    Args:
        latent_dim (int): Dimension of the latent space
        P (int): Number of beads per configuration
        num_atoms (int): Number of atoms per molecule
        edge_index (torch.Tensor): Tensor with all the edges, shape (2, Edges)
        node_feat_dim (int): Number of coordinates per atoms, xyz coordinates
        hidden_dim (int): Dimension of the hidden space
        num_layers (int): Number of EGCL layers inside of the encoder 
    """
    def __init__(
            self, 
            latent_dim: int, 
            P: int, 
            num_atoms: int, 
            edge_index: torch.Tensor, 
            node_feat_dim: int = 3,
            hidden_dim: int = 64, 
            num_layers: int = 6
        ):
            super().__init__()

            self.P = P
            self.num_atoms = num_atoms
            self.N = P * num_atoms
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

    def forward(self, z: torch.Tensor)-> torch.Tensor:
        """Decodes the latent space back into a flattend PIMC configuration

        Args:
            z (torch.Tensor): Latent vectors sampled from VAE space

        Returns:
            torch.Tensor: Reconstructed flattened PIMC configuration, shape (B, P * num_atoms * 3)
        """
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
    """VAE class 

    Architecture of the Variational Autoencoder (VAE) for sampling and producing new PIMC configurations

    Args:
        latent_dim (int): Dimension of the latent space
        P (int): NUmber of beads per configuration
        num_atoms (int): Number of atoms per molecule
        edge_index (torch.Tensor): Tensor with all the edges, shape (2, Edges)
        ode_feat_dim (int): Number of coordinates per atoms, xyz coordinates
        hidden_dim (int): Dimension of the hidden space
        num_layers (int): Number of EGCL layers inside of the encoder
    """
    def __init__(
            self, 
            latent_dim: int, 
            P: int, 
            num_atoms: int, 
            edge_index: torch.Tensor,
            node_feat_dim: int = 3,
            hidden_dim: int = 64,
            num_layers: int = 6,
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
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor)-> torch.Tensor:
        """Applying the reparametrization trick to perform latent sampling

        Args:
            mu (torch.Tensor): Mean vector of latent space
            logvar (torch.Tensor): Log-variance vector of latent space

        Returns:
            z (torch.Tensor): Sampled latent vector z, shape (B, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        # random value
        eps = torch.randn_like(std)
        z = mu + eps *std
        return z
    
    def forward(self, x: torch.Tensor)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE: Input -> Encoder -> Latent Space -> Decoder -> Output

        Args:
            x (torch.Tensor): Flattened input PIMC configurations, shape (B, num_atoms * P * 3)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                x_hat: Reconstructed PIMC configuration, shape (B, P * num_atoms * 3)
                mu: Latent mean vector, shape (B, latent_dim)
                logvar: Log-variance vector, shape (B, latent_dim)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar


################################## Loss Function ########################################
def vae_loss(
        x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar:torch.Tensor, beta:float
)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function to compute the loss of the VAE

    Args:
        x (torch.Tensor): Input PIMC configuration
        x_hat (torch.Tensor): Reconstructed output PIMC configuration
        mu (torch.Tensor): Mean vector of latent space
        logvar (torch.Tensor): Log-variance vector of latent space
        beta (float): Parameter defining the influence of the KL divergence

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            total_loss: Total loss 
            recon_loss: Reconstruction loss defined by the MSE loss
            kl_div: Regularization loss defined by the KL divergence
    """
    B = x.size(0)
    device = x.device

    # reconstruction loss
    recon_loss = F.mse_loss(x_hat, x, reduction='sum') / B 

    # regularization loss -> regularization oder representation loss? Im Stats VU wars representation loss ?
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    total_loss = recon_loss + beta * kl_div

    return total_loss, recon_loss, kl_div


################################## Train Function ##########################################
def train(model: nn.Module, train_loader, val_loader, optimizer:  torch.optim.Optimizer, config): #-> list[tuple[float, float]]:
    """Function to train and validate the VAE over a defined number of epochs

    For each epoch the function trains the VAE on all batches, all the training batches (train loader) 
    and then validates over all the validation batches (val_loader)

    Args:
        model (nn.Module): Model which sould be trained here the VAE
        train_loader (Dataloader): Training data
        val_loader (Dataloader): Validation data
        optimizer (torch.optim.Optimizer): Optimizer used to update the parameters of the model
        config (_type_): Configuration file containing different informations like the epoch size or the number of beads per configuration

    Returns:
        _type_: _description_
    """
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
def plot_loss(loss_history: list, outfile: str = "loss_30Beads.pdf"):
    """Function to plot the validation and training loss
    Both loss functions should converge to the same value 

    Args:
        loss_history (list): Array of the training and validation loss
        outfile (str, optional): Name of the plot. Defaults to "loss_30Beads.pdf".
    """
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
