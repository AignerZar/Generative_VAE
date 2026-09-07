"""
Function to train the AE, before training the actual VAE -> no usage of reparametrization trick just using mu
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import config
from graph_h2o import build_edge_index
from model import VAE, plot_loss

import numpy as np
import torch

import config
from graph_h2o import build_edge_index
from model import VAE
from preprocessing import create_preprocessed_splits, load_flat_data
from training import make_data_loaders, plot_loss, train_autoencoder


def main() -> None:
    """Main function, containing all function and so on for the AE traiing
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    data = load_flat_data(config.input_file, config.P, config.num_atoms)
    preprocessing_file = getattr(
        config,
        "preprocessing_file",
        "preprocessing_h2o_30Beads.npz",
    )
    train_data, val_data = create_preprocessed_splits(
        data_flat=data,
        P=config.P,
        num_atoms=config.num_atoms,
        validation_split=config.validation_split,
        seed=config.seed,
        preprocessing_file=preprocessing_file,
    )
    train_loader, val_loader = make_data_loaders(
        train_data,
        val_data,
        config.batch_size,
    )

    edge_index = build_edge_index(config.P, config.num_atoms).to(config.device)
    model = VAE(
        latent_dim=config.latent_dimension,
        P=config.P,
        num_atoms=config.num_atoms,
        edge_index=edge_index,
        node_feat_dim=4,
    ).to(config.device)

    for parameter in model.encoder.fc_logvar.parameters():
        parameter.requires_grad_(False)

    epochs = getattr(config, "ae_epochs", 100)
    learning_rate = getattr(
        config,
        "ae_learning_rate",
        config.learning_rate,
    )
    checkpoint = getattr(
        config,
        "ae_checkpoint",
        "checkpoints/ae_h2o_nodespecific_best.pt",
    )
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )

    print(f"Using device: {config.device}")
    print(f"AE training: epochs={epochs}, learning rate={learning_rate}")
    print("Latent mode: z = mu (no random noise)")
    print("KL weight: beta = 0")

    history = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=config.device,
        epochs=epochs,
        checkpoint_path=checkpoint,
    )

    if getattr(config, "save_loss_plot", True):
        plot_loss(
            history,
            getattr(config, "ae_loss_plot", "outputs/plots/loss_ae.pdf"),
        )


if __name__ == "__main__":
    main()
