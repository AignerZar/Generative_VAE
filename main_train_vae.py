"""Stage 2: initialize from the AE checkpoint and train the VAE."""

from pathlib import Path

import numpy as np
import torch

import config
from graph_h2o import build_edge_index
from model import VAE
from preprocessing import load_flat_data, load_preprocessed_splits
from training2 import make_data_loaders, plot_loss, train_vae, plot_geometry_metrics


def main() -> None:
    """Function containing all files, codes and so on used in the VAE training
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    data = load_flat_data(config.input_file, config.P, config.num_atoms)
    preprocessing_file = getattr(
        config,
        "preprocessing_file",
        "preprocessing_h2o_30Beads.npz",
    )
    train_data, val_data, preprocessing = load_preprocessed_splits(
        data_flat=data,
        P=config.P,
        num_atoms=config.num_atoms,
        preprocessing_file=preprocessing_file,
    )
    global_scale = float(preprocessing["global_scale"])
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

    ae_checkpoint = Path(
        getattr(
            config,
            "ae_checkpoint",
            "checkpoints/ae_h2o_best.pt",
        )
    )
    if not ae_checkpoint.is_file():
        raise FileNotFoundError(f"AE checkpoint not found: {ae_checkpoint}")

    model.load_state_dict(
        torch.load(
            ae_checkpoint,
            map_location=config.device,
            weights_only=True,
        )
    )

    torch.nn.init.zeros_(model.encoder.fc_logvar.weight)
    torch.nn.init.constant_(model.encoder.fc_logvar.bias, -4.0)

    epochs = getattr(config, "vae_epochs", config.n_epochs)
    learning_rate = getattr(
        config,
        "vae_learning_rate",
        config.learning_rate * 0.1,
    )
    gamma = float(getattr(config, "gamma", 1.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Using device: {config.device}")
    print(f"Loaded AE checkpoint: {ae_checkpoint}")
    print(f"VAE learning rate: {learning_rate}")
    print(
        f"VAE training: epochs={epochs}, "
        f"beta_max={config.beta_max}, "
        f"KL warm-up={config.kl_warmup_epochs} epochs, "
        f"Gamma={gamma}"
    )

    history = train_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=config.device,
        epochs=epochs,
        beta_max=config.beta_max,
        warmup_epochs=config.kl_warmup_epochs,
        P=config.P,
        num_atoms=config.num_atoms,
        global_scale=global_scale,
        gamma=config.gamma,
    )

    vae_checkpoint = Path(
        getattr(
            config,
            "vae_checkpoint",
            "checkpoints/vae_h2o_beta001.pt",
        )
    )
    vae_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), vae_checkpoint)
    print(f"Saved VAE checkpoint: {vae_checkpoint}")

    if getattr(config, "save_loss_plot", True):
        plot_loss(
            history,
            getattr(config, "vae_loss_plot", "outputs/plots/loss_vae.pdf"), start_epoch=50
        )
        plot_geometry_metrics(
            history, 
            getattr(config, "vae_geometry_plot", "outputs/plots/geometry_vae.pdf"),
            start_epoch=50,
        )


if __name__ == "__main__":
    main()
