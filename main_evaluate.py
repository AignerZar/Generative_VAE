"""
Code to evaluate the VAE, just evaluation no training
"""
from pathlib import Path
import numpy as np
import torch
import config
from evaluate_results import (
    #configuration_sse,
    geometry_distributions,
    physical_coordinates,
    plot_geometry_distributions,
    print_geometry_summary,
    #print_reconstruction_diagnostics,
)
from graph_h2o import build_edge_index
from model import VAE
from preprocessing import load_flat_data, load_preprocessed_splits
from sampling import (
    decode_in_batches,
    encode_in_batches,
    sample_aggregated_posterior,
    sample_posterior,
    sample_standard_normal,
)


def main() -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    data = load_flat_data(config.input_file, config.P, config.num_atoms)
    preprocessing_file = getattr(
        config,
        "preprocessing_file",
        "preprocessing_h2o_30Beads.npz",
    )
    train_data_np, val_data_np, preprocessing = load_preprocessed_splits(
        data_flat=data,
        P=config.P,
        num_atoms=config.num_atoms,
        preprocessing_file=preprocessing_file,
    )
    train_data = torch.tensor(train_data_np, dtype=torch.float32)
    val_data = torch.tensor(val_data_np, dtype=torch.float32)
    global_scale = float(preprocessing["global_scale"])

    edge_index = build_edge_index(config.P, config.num_atoms).to(config.device)
    model = VAE(
        latent_dim=config.latent_dimension,
        P=config.P,
        num_atoms=config.num_atoms,
        edge_index=edge_index,
        node_feat_dim=4,
    ).to(config.device)

    checkpoint = Path(
        getattr(
            config,
            "vae_checkpoint",
            "checkpoints/vae_h2o_nodespecific_beta001_final.pt",
        )
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint}")
    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=config.device,
            weights_only=True,
        )
    )
    model.eval()

    batch_size = getattr(config, "evaluation_batch_size", config.batch_size)
    number_of_samples = getattr(config, "num_samples", 3500)

    # Validation posteriors are used for deterministic and stochastic reconstruction.
    val_mu, val_logvar = encode_in_batches(
        model, val_data, batch_size, config.device
    )
    deterministic = decode_in_batches(
        model, val_mu, batch_size, config.device
    )
    stochastic_latent = sample_posterior(val_mu, val_logvar)
    stochastic = decode_in_batches(
        model, stochastic_latent, batch_size, config.device
    )

    # Aggregated-posterior generation uses training posteriors only.
    train_mu, train_logvar = encode_in_batches(
        model, train_data, batch_size, config.device
    )
    aggregated_latent = sample_aggregated_posterior(
        train_mu,
        train_logvar,
        number_of_samples,
    )
    generated_aggregated = decode_in_batches(
        model,
        aggregated_latent,
        batch_size,
        config.device,
    )

    # Independent VAE generation from the prescribed N(0, I) prior.
    prior_latent = sample_standard_normal(
        number_of_samples,
        config.latent_dimension,
    )
    generated_prior = decode_in_batches(
        model,
        prior_latent,
        batch_size,
        config.device,
    )

    validation_physical = physical_coordinates(
        val_data, global_scale, config.P, config.num_atoms
    )
    deterministic_physical = physical_coordinates(
        deterministic, global_scale, config.P, config.num_atoms
    )
    stochastic_physical = physical_coordinates(
        stochastic, global_scale, config.P, config.num_atoms
    )
    aggregated_physical = physical_coordinates(
        generated_aggregated, global_scale, config.P, config.num_atoms
    )
    prior_physical = physical_coordinates(
        generated_prior, global_scale, config.P, config.num_atoms
    )

    distributions = {
        "Validation targets": geometry_distributions(validation_physical),
        "Deterministic reconstruction": geometry_distributions(
            deterministic_physical
        ),
        "Stochastic reconstruction": geometry_distributions(stochastic_physical),
        "Aggregated posterior": geometry_distributions(aggregated_physical),
        "Standard-normal prior": geometry_distributions(prior_physical),
    }

    print(f"Using device: {config.device}")
    print(f"Using checkpoint: {checkpoint}")
    print(f"Validation configurations: {len(val_data)}")
    print(f"Generated configurations per method: {number_of_samples}")
    print_geometry_summary(distributions)

    plot_file = getattr(
        config,
        "evaluation_plot",
        "outputs/plots/vae_geometry_evaluation.pdf",
    )
    plot_geometry_distributions(distributions, plot_file)

    # storing and saving the different csv files
    project_directory = Path(__file__).resolve().parent
    sample_output_directory = project_directory / "outputs" / "samples"
    sample_output_directory.mkdir(parents=True, exist_ok=True)

    validation_file = (sample_output_directory / "validation_targets_aligned_angstrom.csv")
    stochastic_file = (sample_output_directory / "stochastic_aggregated_aligned_angstrom.csv")
    aggregated_file = (sample_output_directory /"generated_aggregated_aligned_angstrom.csv")

    np.savetxt(
        validation_file,
        validation_physical.reshape(len(validation_physical), -1),
        delimiter=",",
    )
    np.savetxt(
        stochastic_file,
        stochastic_physical.reshape(len(stochastic_physical), -1),
        delimiter=",",
    )
    np.savetxt(
        aggregated_file,
        aggregated_physical.reshape(len(aggregated_physical), -1),
        delimiter=",",
    )

    print(f"\nSaved plot: {plot_file}")
    print(f"Saved plot preview: {Path(plot_file).with_suffix('.png')}")
    print(f"Saved validation targets: {validation_file}")
    print(f"Saved stochastic reconstructions: {stochastic_file}")
    print(f"Saved aggregated-posterior samples: {aggregated_file}")


if __name__ == "__main__":
    main()
