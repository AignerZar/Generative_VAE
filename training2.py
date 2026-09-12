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
GeometryMetricDict = Dict[str, float]


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


# Loss functions -------------------------------------------------------------------------------------------------------------
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

def _reshape_physical_coordinates(
        tensor: torch.Tensor,
        P: int,
        num_atoms: int,
        global_scale: float,
) -> torch.Tensor:
    """Function for reshaping the physical coordinates

    Args:
        tensor (torch.Tensor): Tensor containing the data which should be reshaped
        P (int): Number of beads per PIMC configuration
        num_atoms (int): Number of atoms per molecule
        global_scale (float): Scaling factor used for the normalization

    Returns:
        torch.Tensor: Reshpaed input data
    """
    expected_features = P * num_atoms * 3
    flat = _flat_output(tensor)
    if flat.size(1) != expected_features:
        raise ValueError("Wrong coordinates!")

    return flat.reshape(-1, P, num_atoms, 3) * float(global_scale)

def _h2o_geometry_quantities(
    tensor: torch.Tensor,
    P: int,
    num_atoms: int,
    global_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function to compute the bond length and the bond angles of the input and output dataset

    Args:
        tensor (torch.Tensor): Dataset which shall be investigated
        P (int): Number of beads per configuration
        num_atoms (int): Number of atoms per molecule
        global_scale (float): Global scale used for the normalization

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
            - bond length between H1 and O
            - bond length between H2 and O
            - Norm of oh1
            - Norm of oh2
            - cosine alpha of the bond angle
    """
    coords = _reshape_physical_coordinates(tensor, P, num_atoms, global_scale)

    h1 = coords[:, :, 0, :]
    oxygen = coords[:, :, 1, :]
    h2 = coords[:, :, 2, :]

    oh1 = h1 - oxygen
    oh2 = h2 - oxygen

    r1 = torch.linalg.vector_norm(oh1, dim=-1)
    r2 = torch.linalg.vector_norm(oh2, dim=-1)
    cosine = F.cosine_similarity(oh1, oh2, dim=-1, eps=1e-8)
    cosine = torch.clamp(cosine, -1.0, 1.0)

    return oh1, oh2, r1, r2, cosine

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
    _, _, pred_r1, pred_r2, pred_cosine = _h2o_geometry_quantities(
        prediction, P, num_atoms, global_scale
    )
    _, _, true_r1, true_r2, true_cosine = _h2o_geometry_quantities(
        target, P, num_atoms, global_scale
    )

    number_of_configurations = prediction.size(0)

    bond_loss = (
        F.mse_loss(pred_r1, true_r1, reduction="sum")
        + F.mse_loss(pred_r2, true_r2, reduction="sum")
    ) / number_of_configurations

    angle_cosine_loss = F.mse_loss(
        pred_cosine, true_cosine, reduction="sum"
    ) / number_of_configurations

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
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Overall loss, reconstruction, representation loss, geometry, bond and angle loss 
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
                optimizer.zero_grad(set_to_none=True)

            mu, _ = model.encoder(batch)
            prediction = model.decoder(mu)
            loss = reconstruction_loss(prediction, batch)

            if is_training:
                loss.backward()
                optimizer.step()

            count = batch.size(0)
            weighted_loss += loss.item() * count
            number_of_configurations += count

    if number_of_configurations == 0:
        raise ValueError("DataLoader is empty.")

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

# physical geometry metrics
def geometry_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    P: int,
    num_atoms: int,
    global_scale: float,
) -> GeometryMetricDict:
    """Function to compute the metrics during the training process, only used for evaluation, used on the validation data set 

    Args:
        prediction (torch.Tensor): Output data, compute and generated
        target (torch.Tensor): Input data
        P (int): Number of beads per configuration
        num_atoms (int): Number of atoms per molecule
        global_scale (float): Global scaling factor, used for the normalization

    Returns:
        GeometryMetricDict: Dictionary contianing the geometry metrics like the MAE and RMSE of the bond length and the MAE and RMSE of the bond angle
    """
    _, _, pred_r1, pred_r2, pred_cosine = _h2o_geometry_quantities(
        prediction, P, num_atoms, global_scale
    )
    _, _, true_r1, true_r2, true_cosine = _h2o_geometry_quantities(
        target, P, num_atoms, global_scale
    )

    bond_errors = torch.cat(
        [
            (pred_r1 - true_r1).reshape(-1),
            (pred_r2 - true_r2).reshape(-1),
        ]
    )

    bond_mae = torch.mean(torch.abs(bond_errors))
    bond_rmse = torch.sqrt(torch.mean(bond_errors.square()))

    pred_angle = torch.rad2deg(torch.acos(pred_cosine))
    true_angle = torch.rad2deg(torch.acos(true_cosine))
    angle_errors = pred_angle - true_angle

    angle_mae = torch.mean(torch.abs(angle_errors))
    angle_rmse = torch.sqrt(torch.mean(angle_errors.square()))

    return {
        "bond_mae_angstrom": float(bond_mae.item()),
        "bond_rmse_angstrom": float(bond_rmse.item()),
        "angle_mae_degree": float(angle_mae.item()),
        "angle_rmse_degree": float(angle_rmse.item()),
    }




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
    compute_physical_metrics: bool = False,
) -> Tuple[float, float, float, float, float, float, Optional[GeometryMetricDict]]:
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

    metric_sums = {
        "bond_mae_angstrom": 0.0,
        "bond_rmse_angstrom": 0.0,
        "angle_mae_degree": 0.0,
        "angle_rmse_degree": 0.0,
    }

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for (batch,) in loader:
            batch = batch.to(device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            mu, logvar = model.encoder(batch)
            latent = model.reparameterize(mu, logvar)
            prediction = model.decoder(latent)

            loss, reconstruction, kl, geometry, bonds, angle = vae_loss(
                batch,
                prediction,
                mu,
                logvar,
                beta,
                gamma,
                P,
                num_atoms,
                global_scale,
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

            if compute_physical_metrics:
                metrics = geometry_metrics(
                    prediction.detach(), batch, P, num_atoms, global_scale
                )
                for key in metric_sums:
                    metric_sums[key] += metrics[key] * count

    if number_of_configurations == 0:
        raise ValueError("DataLoader is empty.")

    physical_metrics = None
    if compute_physical_metrics:
        physical_metrics = {
            key: value / number_of_configurations
            for key, value in metric_sums.items()
        }

    return (
        weighted_loss / number_of_configurations,
        weighted_reconstruction / number_of_configurations,
        weighted_kl / number_of_configurations,
        weighted_geometry / number_of_configurations,
        weighted_bonds / number_of_configurations,
        weighted_angle / number_of_configurations,
        physical_metrics,
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
        "bond_geometry": [],
        "angle_geometry": [],
        "beta": [],
        "gamma": [],
        "val_bond_mae_angstrom": [],
        "val_bond_rmse_angstrom": [],
        "val_angle_mae_degree": [],
        "val_angle_rmse_degree": [],
    }

    for epoch in range(epochs):
        beta = beta_for_epoch(epoch, beta_max, warmup_epochs)

        (
            train_loss,
            reconstruction,
            kl,
            geometry,
            bonds,
            angle,
            _,
        ) = _run_vae_epoch(
            model,
            train_loader,
            device,
            beta,
            gamma,
            P,
            num_atoms,
            global_scale,
            optimizer,
            compute_physical_metrics=False,
        )

        (
            val_loss,
            _,
            _,
            _,
            _,
            _,
            val_metrics,
        ) = _run_vae_epoch(
            model,
            val_loader,
            device,
            beta,
            gamma,
            P,
            num_atoms,
            global_scale,
            optimizer=None,
            compute_physical_metrics=True,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["reconstruction"].append(reconstruction)
        history["kl"].append(kl)
        history["geometry"].append(geometry)
        history["bond_geometry"].append(bonds)
        history["angle_geometry"].append(angle)
        history["beta"].append(beta)
        history["gamma"].append(gamma)

        if val_metrics is None:
            raise RuntimeError("Validation geometry metrics were not computed.")

        history["val_bond_mae_angstrom"].append(val_metrics["bond_mae_angstrom"])
        history["val_bond_rmse_angstrom"].append(val_metrics["bond_rmse_angstrom"])
        history["val_angle_mae_degree"].append(val_metrics["angle_mae_degree"])
        history["val_angle_rmse_degree"].append(val_metrics["angle_rmse_degree"])

        print(
            f"Epoch {epoch + 1:03d}: "
            f"Beta={beta:.5f}, "
            f"TrainLoss={train_loss:.6f}, "
            f"Recon={reconstruction:.6f}, "
            f"KL={kl:.6f}, "
            f"BetaKL={beta * kl:.6f}, "
            f"Geom={geometry:.6f}, "
            f"WeightedGeom={gamma * geometry:.6f}, "
            f"Bond={bonds:.6f}, "
            f"Angle={angle:.6f}, "
            f"ValLoss={val_loss:.6f}, "
            f"ValBondMAE={val_metrics['bond_mae_angstrom']:.6f}, "
            f"ValAngleMAE={val_metrics['angle_mae_degree']:.4f} deg"
        )

    return history


def _prepare_plot_range(history_length: int, start_epoch: int) -> np.ndarray:
    """Return one-based epoch numbers after omitting the first start_epoch epochs."""
    if start_epoch < 0:
        raise ValueError("start_epoch must be non-negative.")
    if start_epoch >= history_length:
        raise ValueError(
            f"start_epoch={start_epoch}, but history contains only "
            f"{history_length} epochs."
        )
    return np.arange(start_epoch + 1, history_length + 1)


def plot_loss(
    history: LossHistory,
    output_file: str,
    start_epoch: int = 50,
) -> None:
    """Plot total loss and weighted VAE loss components.

    First few epochs are not display, could lead to confusion because we load pretrained weights, but weights are only trained on reconstruction loss, therefore
    the loss is really small at first then peaks before the network starts learning properly and the loss sinks again.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_epochs = len(history["train_loss"])
    epochs = _prepare_plot_range(n_epochs, start_epoch)

    train_loss = np.asarray(history["train_loss"], dtype=float)[start_epoch:]
    val_loss = np.asarray(history["val_loss"], dtype=float)[start_epoch:]
    reconstruction = np.asarray(history["reconstruction"], dtype=float)[start_epoch:]
    kl = np.asarray(history["kl"], dtype=float)[start_epoch:]
    geometry = np.asarray(history["geometry"], dtype=float)[start_epoch:]
    bonds = np.asarray(history["bond_geometry"], dtype=float)[start_epoch:]
    angles = np.asarray(history["angle_geometry"], dtype=float)[start_epoch:]
    beta = np.asarray(history["beta"], dtype=float)[start_epoch:]
    gamma = np.asarray(history["gamma"], dtype=float)[start_epoch:]

    weighted_kl = beta * kl
    weighted_geometry = gamma * geometry
    weighted_bonds = gamma * bonds
    weighted_angles = gamma * angles

    # Total loss plot
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(epochs, train_loss, label=r"Training loss", linewidth=2.0)
    axis.plot(epochs, val_loss, label=r"Validation loss", linewidth=2.0)
    axis.set_xlabel(r"Epoch", fontsize=18)
    axis.set_ylabel(r"Total loss", fontsize=18)
    axis.tick_params(axis="both", labelsize=15)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=18)
    figure.tight_layout()

    total_path = output_path.with_name(
        f"{output_path.stem}_total_loss{output_path.suffix}"
    )
    figure.savefig(total_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    # Weighted component plot
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.plot(
        epochs,
        reconstruction,
        label=r"$\mathcal{L}_{\mathrm{rec}}$",
        linewidth=2.0,
    )
    axis.plot(
        epochs,
        weighted_kl,
        label=r"$\beta\,\mathcal{L}_{\mathrm{KL}}$",
        linewidth=2.0,
    )
    axis.plot(
        epochs,
        weighted_geometry,
        label=r"$\gamma\,\mathcal{L}_{\mathrm{geom}}$",
        linewidth=2.0,
    )
    axis.plot(
        epochs,
        weighted_bonds,
        label=r"$\gamma\,\mathcal{L}_{\mathrm{bond}}$",
        linewidth=2.0,
    )
    axis.plot(
        epochs,
        weighted_angles,
        label=r"$\gamma\,\mathcal{L}_{\mathrm{angle}}$",
        linewidth=2.0,
    )
    axis.set_xlabel(r"Epoch", fontsize=18)
    axis.set_ylabel(r"Weighted loss contribution", fontsize=18)
    axis.tick_params(axis="both", labelsize=15)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=18)
    figure.tight_layout()

    components_path = output_path.with_name(
        f"{output_path.stem}_components{output_path.suffix}"
    )
    figure.savefig(components_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_geometry_metrics(
    history: LossHistory,
    output_file: str,
    start_epoch: int = 50,
) -> None:
    """File to plot the geometrics metrics of the training process

    Args:
        history (LossHistory): Training history also containing the metrics
        output_file (str): Output file, where results should be written
        start_epoch (int, optional): First few epochs are not displayed because we have a pretrained network at first
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bond_mae = history["val_bond_mae_angstrom"]
    n_epochs = len(bond_mae)
    epochs = _prepare_plot_range(n_epochs, start_epoch)

    bond_mae = np.asarray(bond_mae, dtype=float)[start_epoch:]
    bond_rmse = np.asarray(
        history["val_bond_rmse_angstrom"], dtype=float
    )[start_epoch:]
    angle_mae = np.asarray(
        history["val_angle_mae_degree"], dtype=float
    )[start_epoch:]
    angle_rmse = np.asarray(
        history["val_angle_rmse_degree"], dtype=float
    )[start_epoch:]

    # Bond-length plot
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(epochs, bond_mae, label=r"O--H bond-length MAE", linewidth=2.0)
    axis.plot(epochs, bond_rmse, label=r"O--H bond-length RMSE", linewidth=2.0)
    axis.set_xlabel(r"Epoch", fontsize=18)
    axis.set_ylabel(r"Bond-length error / \AA", fontsize=18)
    axis.tick_params(axis="both", labelsize=15)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=18)
    figure.tight_layout()

    bond_path = output_path.with_name(
        f"{output_path.stem}_bond_metrics{output_path.suffix}"
    )
    figure.savefig(bond_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    # Bond-angle plot
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(epochs, angle_mae, label=r"H--O--H angle MAE", linewidth=2.0)
    axis.plot(epochs, angle_rmse, label=r"H--O--H angle RMSE", linewidth=2.0)
    axis.set_xlabel(r"Epoch", fontsize=18)
    axis.set_ylabel(r"Bond-angle error / $^\circ$", fontsize=18)
    axis.tick_params(axis="both", labelsize=15)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=18)
    figure.tight_layout()

    angle_path = output_path.with_name(
        f"{output_path.stem}_angle_metrics{output_path.suffix}"
    )
    figure.savefig(angle_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
