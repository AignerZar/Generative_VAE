""" 
In this code all functions and so on are defined used for the evaluation of the VAE
"""
from pathlib import Path
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

font1 = {'family':'sans-serif','color':'black','size':12}
font2 = {'family':'sans-serif','color':'black','size':20}

plt.rcParams['text.usetex'] = True #LaTeX

Geometry = Dict[str, np.ndarray]


def physical_coordinates(
    normalized_flat: torch.Tensor,
    global_scale: float,
    P: int,
    num_atoms: int,
) -> np.ndarray:
    """Function to convert the coordinates back to obtain correct physical results

    Args:
        normalized_flat (torch.Tensor): normalized data which should be back transformed
        global_scale (float): Global scale factor used for normalization
        P (int): Number of beads per configuration
        num_atoms (int): Number of atoms per molecule

    Returns:
        np.ndarray: denormalized coordinates
    """
    return (
        normalized_flat.numpy().reshape(-1, P, num_atoms, 3)
        * global_scale
    )


def geometry_distributions(coordinates: np.ndarray) -> Geometry:
    """Function to compute the geometry of the results and the original input

    Args:
        coordinates (np.ndarray): Coordinates of the atoms 

    Returns:
        Geometry: Returning bond lengths and bond angle
    """
    H1 = coordinates[:, :, 0, :]
    O = coordinates[:, :, 1, :]
    H2 = coordinates[:, :, 2, :]

    vector_1 = H1 - O
    vector_2 = H2 - O
    bond_1 = np.linalg.norm(vector_1, axis=-1)
    bond_2 = np.linalg.norm(vector_2, axis=-1)
    cosine = np.sum(vector_1 * vector_2, axis=-1) / np.maximum(
        bond_1 * bond_2, 1e-12
    )
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    hydrogen_distance = np.linalg.norm(H1 - H2, axis=-1)

    #bond_1_mean = bond_1.mean(axis=1)
    #bond_2_mean = bond_2.mean(axis=1)
    #angle_mean = angle.mean(axis=1)
    #hydrogen_distance_mean = hydrogen_distance.mean(axis=-1)

    return {
        "O-H1": bond_1.reshape(-1),#bond_1_mean,#
        "O-H2": bond_2.reshape(-1),
        "H-O-H": angle.reshape(-1),
        "H-H": hydrogen_distance.reshape(-1),
    }


def print_geometry_summary(distributions: Dict[str, Geometry]) -> None:
    """Function to print out the computed statistics -> bond lengths and bond angles

    Args:
        distributions (Dict[str, Geometry]): Obtained and computed geometries
    """
    print("\nGeometry summary")
    print(f"{'Quantity':10s} {'Dataset':27s} {'Mean':>11s} {'Std':>11s}")
    print("-" * 63)

    for quantity in ("O-H1", "O-H2", "H-O-H", "H-H"):
        for label, geometry in distributions.items():
            values = geometry[quantity]
            print(
                f"{quantity:10s} {label:27s} "
                f"{values.mean():11.5f} {values.std(ddof=1):11.5f}"
            )


def plot_geometry_distributions(
    distributions: Dict[str, Geometry],
    output_file: str,
) -> None:
    """Function to plot the different geometry distributoions 

    Args:
        distributions (Dict[str, Geometry]): Various distributions inside of a dictionary, computed above
        output_file (str): Name of the file which should contain the results
    """
    plotted_labels = (
        r"Validation targets",
        #"Deterministic reconstruction",
        r"Stochastic reconstruction",
        r"Aggregated posterior",
        #"Standard-normal prior",
    )
    colors = ("black", "tab:blue", "tab:orange")#, "tab:green")
    quantities = (
        ("O-H1", r"$\mathrm{O-H_1}$ bond length [Angstrom]"),
        ("O-H2", r"$\mathrm{O-H_2}$ bond length [Angstrom]"),
        ("H-O-H", r"$\mathrm{H-O-H}$ angle [degree]"),
        #("H-H", "H-H distance [Angstrom]"),
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, (quantity, x_label) in zip(axes, quantities):
        plotted_values = [
            distributions[label][quantity] for label in plotted_labels
        ]
        lower = min(np.quantile(values, 0.001) for values in plotted_values)
        upper = max(np.quantile(values, 0.999) for values in plotted_values)
        padding = 0.15 * max(upper - lower, 1e-6)
        bins = np.linspace(lower - padding, upper + padding, 55)

        for label, color in zip(plotted_labels, colors):
            values = distributions[label][quantity]
            if label == r"Validation targets":
                axis.hist(
                    values,
                    bins=bins,
                    density=True,
                    alpha=0.25,
                    color=color,
                    label=label,
                )
            else:
                axis.hist(
                    values,
                    bins=bins,
                    density=True,
                    histtype="step",
                    linewidth=1.6,
                    color=color,
                    label=label,
                )

        axis.set_xlabel(x_label, fontsize=16, fontweight="bold")
        axis.set_ylabel(r"Probability density", fontsize=16, fontweight="bold")
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontweight("bold")
        axis.grid(alpha=0.2)

    #axes[0].legend(fontsize=14, prop={"size": 14, "weight": "bold"})
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, prop={"size": 14, "weight": "bold"}, frameon=False)

    figure.tight_layout(rect=[0, 0.12, 1, 1])
    #figure.tight_layout()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    figure.savefig(output_path.with_suffix(".png"), dpi=180)
    plt.close(figure)