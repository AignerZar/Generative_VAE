""" 
Defining functions used for the traiing process 
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

font1 = {'family':'sans-serif','color':'black','size':12}
font2 = {'family':'sans-serif','color':'black','size':20}

plt.rcParams['text.usetex'] = True #LaTeX

LossHistory = Dict[str, list]


def make_data_loaders(
    train_data,
    val_data,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Create loaders from normalized NumPy arrays or tensors."""
    train_tensor = torch.as_tensor(train_data, dtype=torch.float32)
    val_tensor = torch.as_tensor(val_data, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def _flat_output(output: torch.Tensor) -> torch.Tensor:
    return output.reshape(output.size(0), -1)


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Function to compute the mean squared error loss -> recostruction loss, use to produce physically plausible results

    Args:
        prediction (torch.Tensor): Produced output, serves as the prediction
        target (torch.Tensor): Original input, used as the target

    Returns:
        torch.Tensor: MSE loss
    """
    prediction = _flat_output(prediction)
    target = _flat_output(target)
    return F.mse_loss(prediction, target, reduction="sum") / target.size(0)


def kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """Function to compute the KL divergence, serves as the representation loss (kann i da wirklich represnetationbloss sagen -> steht so im VU skript)

    Args:
        mu (torch.Tensor): Mean value
        logvar (torch.Tensor): Logarithm of the variance

    Returns:
        torch.Tensor: KL divergence
    """
    return -0.5 * torch.sum(
        1.0 + logvar - mu.square() - logvar.exp()
    ) / mu.size(0)


def geometry_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    P: int,
    num_atoms: int,
    global_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function to compute the geometry loss

    Args:
        prediction (torch.Tensor): Predicted Output of the VAE
        target (torch.Tensor): Input of the VAE, used as the Target
        P (int): Number of beads per configuration
        num_atoms (int): Number of atoms per molecule
        global_scale (float): Scaling factor of the data 

    Returns:
        torch.Tensor: Geometry Loss
    """
    if num_atoms != 3:
        raise ValueError("H2O geometry does not have right dimensions.")
    if not float(global_scale) > 0.0:
        raise ValueError("Global scale has an invalid value!")

    prediction_coords = prediction.reshape(-1, P, num_atoms, 3)
    target_coords = target.reshape(-1, P, num_atoms, 3)

    prediction_coords = prediction_coords * float(global_scale)
    target_coords = target_coords * float(global_scale)

    hat_h1 = prediction_coords[:, :, 0, :]
    hat_o = prediction_coords[:, :, 1, :]
    hat_h2 = prediction_coords[:, :, 2, :]

    h1 = target_coords[:, :, 0, :]
    o = target_coords[:, :, 1, :]
    h2 = target_coords[:, :, 2, :]

    hat_o_h1 = hat_h1 - hat_o
    hat_o_h2 = hat_h2 - hat_o
    o_h1 = h1 - o
    o_h2 = h2 - o

    hat_r1 = torch.linalg.vector_norm(hat_o_h1, dim=-1)
    hat_r2 = torch.linalg.vector_norm(hat_o_h2, dim=-1)
    r1 = torch.linalg.vector_norm(o_h1, dim=-1)
    r2 = torch.linalg.vector_norm(o_h2, dim=-1)

    number_of_configurations = target_coords.size(0)

    bond_loss = (F.mse_loss(hat_r1, r1, reduction="sum") + F.mse_loss(hat_r2, r2, reduction="sum")) / number_of_configurations

    hat_cosine = F.cosine_similarity(hat_o_h1, hat_o_h2, dim=-1, eps=1e-8)
    cosine = F.cosine_similarity(o_h1, o_h2, dim=-1, eps=1e-8)
    angle_cosine_loss = F.mse_loss(hat_cosine, cosine, reduction="sum") / number_of_configurations

    total_geometry = bond_loss + angle_cosine_loss

    return total_geometry, bond_loss, angle_cosine_loss


def vae_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    gamma: float,
    P: int,
    num_atoms: int,
    global_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function to compute the overall loss consisting of representation and reconstruction loss

    Args:
        target (torch.Tensor): Input, serves as the target
        prediction (torch.Tensor): Predicted output
        mu (torch.Tensor): Mean value
        logvar (torch.Tensor): Logarithm of the variance
        beta (float): Factor which determines how much influence the KL divergence has
        gamma (float): Factor which determines how much influence the geometry loss should have
        P (int): Number of beads per configurations
        num_atoms (int): Number of atoms per molecule
        gloabl_scale (float): Factor for the scaling version

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Overall loss, reconstruction, representation loss and geometry loss
    """
    reconstruction = reconstruction_loss(prediction, target)
    kl = kl_divergence(mu, logvar)
    geometry, bonds, angle = geometry_loss(prediction, target, P,  num_atoms, global_scale)
    total = reconstruction + beta * kl + gamma * geometry
    return total, reconstruction, kl, geometry, bonds, angle


def beta_for_epoch(
    epoch_index: int,
    beta_max: float,
    warmup_epochs: int,
) -> float:
    """Function to compute beta, starts raising gradually till it reaches a constant value

    Args:
        epoch_index (int): Index determining the epoch
        beta_max (float): Maximum beta value (constant value)
        warmup_epochs (int): How many epochs it takes till beta reaches the constant value
        beta (float): Beginning value of beta

    Returns:
        float: Gradually raising beta, till it reaches the constant value
    """
    if warmup_epochs <= 1:
        return float(beta_max)
    
    progress = min(epoch_index / float(warmup_epochs - 1), 1.0)

    return float(beta_max) * progress


def _run_autoencoder_epoch(
    model,
    loader: DataLoader,
    device: str,
    optimizer: Optional[torch.optim.Optimizer],
) -> float:
    """Function which performs one epoch of the AE training

    Args:
        model (_type_): Model defined in model.py
        loader (DataLoader): 
        device (str): 
        optimizer (Optional[torch.optim.Optimizer]): Optimizer used -> e.g. Adam or Lion

    Returns:
        float: Returns the reconstruction error of one configuration, only reconstruction -> AE training
    """
    is_training = optimizer is not None
    model.train(is_training)
    weighted_loss = 0.0
    number_of_configurations = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for (batch,) in loader:
            batch = batch.to(device)
            if is_training:
                optimizer.zero_grad()

            mu, _ = model.encoder(batch)
            prediction = model.decoder(mu)  # deterministic AE: z = mu
            loss = reconstruction_loss(prediction, batch)

            if is_training:
                loss.backward()
                optimizer.step()

            weighted_loss += loss.item() * batch.size(0)
            number_of_configurations += batch.size(0)

    return weighted_loss / number_of_configurations


def train_autoencoder(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int,
    checkpoint_path: str,
) -> LossHistory:
    """Function which performs an epoch run two times one time for evaluation one time for training

    Args:
        model (_type_): Model defined in model.py
        train_loader (DataLoader): 
        val_loader (DataLoader): 
        optimizer (torch.optim.Optimizer): Used optimizer, e.g. Lion or Adam
        device (str): 
        epochs (int): Number of epochs 
        checkpoint_path (str): Path where best val set should be stored 

    Returns:
        LossHistory: History of the loss stored in a dictionary
    """
    output_path = Path(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history: LossHistory = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        train_loss = _run_autoencoder_epoch(
            model, train_loader, device, optimizer
        )
        val_loss = _run_autoencoder_epoch(
            model, val_loader, device, optimizer=None
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)

        print(
            f"Epoch {epoch + 1:03d}: "
            f"TrainLoss={train_loss:.6f}, "
            f"ValLoss={val_loss:.6f}, "
            f"BestVal={best_val_loss:.6f}"
        )

    print(f"Saved best AE checkpoint: {output_path}")
    return history


def _run_vae_epoch(
    model,
    loader: DataLoader,
    device: str,
    beta: float,
    gamma: float,
    P: int,
    num_atoms: int,
    global_scale: float,
    optimizer: Optional[torch.optim.Optimizer],
) -> Tuple[float, float, float, float, float, float]:
    """Function to run the VAE for one epoch

    Args:
        model (_type_): VAE model
        loader (DataLoader): 
        device (str): 
        beta (float): Factor determining the influence of KL divergence
        optimizer (Optional[torch.optim.Optimizer]): Optimizer, e.g. Adam or Lion
        gamma (float): Factor determining the influence of the geometry loss

    Returns:
        Tuple[float, float, float]: Total loss, reconstruction loss, representation loss
    """
    is_training = optimizer is not None
    model.train(is_training)
    weighted_loss = 0.0
    weighted_reconstruction = 0.0
    weighted_kl = 0.0
    weighted_geometry = 0.0
    weighted_bonds = 0.0
    weighted_angle = 0.0
    number_of_configurations = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for (batch,) in loader:
            batch = batch.to(device)
            if is_training:
                optimizer.zero_grad()

            mu, logvar = model.encoder(batch)
            latent = model.reparameterize(mu, logvar) #vae training -> using reparametrization trick
            prediction = model.decoder(latent)
            loss, reconstruction, kl, geometry, bonds, angle = vae_loss(
                batch, prediction, mu, logvar, beta, gamma, P, num_atoms, global_scale,
            )

            if is_training:
                loss.backward()
                optimizer.step()

            count = batch.size(0)
            weighted_loss += loss.item() * count
            weighted_reconstruction += reconstruction.item() * count
            weighted_kl += kl.item() * count
            weighted_geometry += geometry.item() * count
            weighted_bonds += bonds.item() * count
            weighted_angle += angle.item() * count
            number_of_configurations += count

    return (
        weighted_loss / number_of_configurations,
        weighted_reconstruction / number_of_configurations,
        weighted_kl / number_of_configurations,
        weighted_geometry / number_of_configurations,
        weighted_bonds / number_of_configurations,
        weighted_angle / number_of_configurations,
    )


def train_vae(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int,
    beta_max: float,
    warmup_epochs: int,
    P: int,
    num_atoms: int,
    global_scale: float,
    gamma: float = 1.0,
) -> LossHistory:
    """Function for training the VAE

    Args:
        model (_type_): VAE model
        train_loader (DataLoader): 
        val_loader (DataLoader): 
        optimizer (torch.optim.Optimizer): Optimizer, e.g. Lion or Adam
        device (str): 
        epochs (int): Number of epochs
        beta_max (float): Maximum value of the constant factor, determining the influence of KL divergencce
        warmup_epochs (int): Number of epochs it takes till beta reaches its maximum and constant value
        gamma (float): Factor determining the influence of the geometry loss

    Returns:
        LossHistory: Loss history 
    """
    history: LossHistory = {
        "train_loss": [],
        "val_loss": [],
        "reconstruction": [],
        "kl": [],
        "geometry": [],
        "bond geometry": [],
        "angle geometry": [],
        "beta": [],
        "gamma": [],
    }

    for epoch in range(epochs):
        beta = beta_for_epoch(epoch, beta_max, warmup_epochs)
        train_loss, reconstruction, kl, geometry, bonds, angle = _run_vae_epoch(
            model, train_loader, device, beta, gamma, P, num_atoms, global_scale, optimizer
        )
        val_loss, _, _, _, _, _ = _run_vae_epoch(
            model, val_loader, device, beta, gamma, P, num_atoms, global_scale,  optimizer=None
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["reconstruction"].append(reconstruction)
        history["kl"].append(kl)
        history["geometry"].append(geometry)
        history["bond geometry"].append(bonds)
        history["angle geometry"].append(angle)
        history["beta"].append(beta)
        history["gamma"].append(gamma)

        print(
            f"Epoch {epoch + 1:03d}: "
            f"Beta={beta:.5f}, "
            f"TrainLoss={train_loss:.6f}, "
            f"Recon={reconstruction:.6f}, "
            f"KL={kl:.6f}, "
            f"Geom={geometry:.6f}, "
            f"WeightedGeom={gamma * geometry:.6f}, "
            f"ValLoss={val_loss:.6f}"
        )

    return history


def plot_loss(history: LossHistory, output_file: str) -> None:
    """Function to plot the loss

    Args:
        history (LossHistory): History of the loss function (training and validation loss)
        output_file (str): Output file, lies in folder where also Plot should be stored
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(history["train_loss"], label=r"Training loss")
    axis.plot(history["val_loss"], label=r"Validation loss")
    axis.set_xlabel(r"Epoch")
    axis.set_ylabel(r"Loss")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
