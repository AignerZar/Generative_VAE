"""
File to plot the geometry of the result -> the bond lengths and the angle between them 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn.functional as F
from scipy.stats import gaussian_kde


# for latex -> for descriptions and names
font1 = {'family':'sans-serif','color':'black','size':12}
font2 = {'family':'sans-serif','color':'black','size':20}

plt.rcParams['text.usetex'] = True #LaTeX

def plot_bond_angle_distributions(dist_original, dist_rec, dist_gen, outfile="bond_angle_distributions.pdf"):
    (r1_o, r2_o, ang_o) = dist_original
    (r1_r, r2_r, ang_r) = dist_rec
    (r1_g, r2_g, ang_g) = dist_gen

    plt.figure(figsize=(14,5))

    # ---- Bond Lengths: O-H1 ----
    plt.subplot(1,3,1)
    plt.hist(r1_o, bins=40, density=True, alpha=0.6, label=r"Original data of $\mathrm{H_2O}$")
    plt.hist(r1_r, bins=40, density=True, alpha=0.6, label=r"Reconstructed data of $\mathrm{H_2O}$")
    plt.hist(r1_g, bins=40, density=True, alpha=0.6, label=r"Generated data of $\mathrm{H_2O}$")
    plt.title(r"$\mathrm{O-H(1)}$  Bond Length Distribution")
    plt.xlabel(r"Bond length [Å]")
    plt.legend()

    # ---- Bond Lengths: O-H2 ----
    plt.subplot(1,3,2)
    plt.hist(r2_o, bins=40, density=True, alpha=0.6, label=r"Original data of $\mathrm{H_2O}$")
    plt.hist(r2_r, bins=40, density=True, alpha=0.6, label=r"Reconstructed data of $\mathrm{H_2O}$")
    plt.hist(r2_g, bins=40, density=True, alpha=0.6, label=r"Generated data of $\mathrm{H_2O}$")
    plt.title(r"$\mathrm{O-H(2)}$ Bond Length Distribution")
    plt.xlabel(r"Bond length [Å]")

    # ---- Angles ----
    plt.subplot(1,3,3)
    plt.hist(ang_o, bins=40, density=True, alpha=0.6, label=r"Original data of $\mathrm{H_2O}$")
    plt.hist(ang_r, bins=40, density=True, alpha=0.6, label=r"Reconstruced data of $\mathrm{H_2O}$")
    plt.hist(ang_g, bins=40, density=True, alpha=0.6, label=r"Gemerated data of $\mathrm{H_2O}$")
    plt.title(r"$\mathrm{H-O-H}$ Angle Distribution")
    plt.xlabel(r"Angle [deg]")

    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()

def plot_bond_angle_distributions_with_kde(dist_original, dist_rec, dist_gen,
                                           outfile="bond_angle_distributions_kde.pdf"):
    """
    Creates a separate figure with histogram + KDE curves
    for Original / Reconstructed / Generated distributions.
    """

    (r1_o, r2_o, ang_o) = dist_original
    (r1_r, r2_r, ang_r) = dist_rec
    (r1_g, r2_g, ang_g) = dist_gen

    fig = plt.figure(figsize=(14,5))

    legend_handles = []

    def plot_hist_kde(data_list, labels, xlabel, title, subplot):
        ax = plt.subplot(1,3,subplot)

        colors = ["blue", "orange", "green"]

        local_handles = []

        for data, label, color in zip(data_list, labels, colors):

            # Histogram
            h = ax.hist(data, bins=70, density=True, alpha=0.3, color=color, label=f"{label} (hist)")

            # KDE-Fit (smooth curve)
            kde = gaussian_kde(data)
            xs = np.linspace(min(data), max(data), 300)
            p, = ax.plot(xs, kde(xs), color=color, lw=2, label=f"{label} (fit)")

            # collect hist + fit handles only once (from first subplot)
            if subplot == 1:
                legend_handles.append(p)

        ax.set_xlabel(xlabel)
        ax.set_title(title)


    # ---- O-H(1) ----
    plot_hist_kde(
        data_list=[r1_o, r1_r, r1_g],
        labels=[r"Original $\mathrm{H_2O}$", r"Reconstructed $\mathrm{H_2O}$", r"Generated $\mathrm{H_2O}$"],
        xlabel=r"Bond length [Å]",
        title=r"$\mathrm{O-H(1)}$ Bond Length Distribution",
        subplot=1
    )

    # ---- O-H(2) ----
    plot_hist_kde(
        data_list=[r2_o, r2_r, r2_g],
        labels=[r"Original $\mathrm{H_2O}$", r"Reconstructed $\mathrm{H_2O}$", r"Generated $\mathrm{H_2O}$"],
        xlabel=r"Bond length [Å]",
        title=r"$\mathrm{O-H(2)}$ Bond Length Distribution",
        subplot=2
    )

    # ---- Angle ----
    plot_hist_kde(
        data_list=[ang_o, ang_r, ang_g],
        labels=[r"Original $\mathrm{H_2O}$", r"Reconstructed $\mathrm{H_2O}$", r"Generated $\mathrm{H_2O}$"],
        xlabel=r"Angle [deg]",
        title=r"$\mathrm{H-O-H}$ Angle Distribution",
        subplot=3
    )
    fig.legend(
        legend_handles,
        [r"Original $\mathrm{H_2O}$", r"Reconstructed $\mathrm{H_2O}$", r"Generated $\mathrm{H_2O}$"],
        loc="lower center",
        ncol=3,
        fontsize=12,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])  # add space for legend
    plt.savefig(outfile)
    plt.show()


def print_distribution_means(dist_original, dist_recon, dist_generated):
    names = ["O-H(1) bond", "O-H(2) bond", "H-O-H angle"]

    sets = {
        "Original": dist_original,
        "Reconstructed": dist_recon,
        "Generated": dist_generated
    }

    print("\n=== Mean values of distributions ===\n")

    for i, name in enumerate(names):
        print(f"--- {name} ---")
        for label, dist in sets.items():
            mean_value = np.mean(dist[i])
            print(f"{label:15s}: {mean_value:.5f}")
        print()


def plot_bond_angle_distributions_mix(dist_original, dist_rec, dist_gen,
                                      outfile="bond_angle_distributions_mix.pdf"):

    (r1_o, r2_o, ang_o) = dist_original
    (r1_r, r2_r, ang_r) = dist_rec
    (r1_g, r2_g, ang_g) = dist_gen

    fig = plt.figure(figsize=(14,5))

    legend_handles = []

    def plot_hist_kde(data_orig, data_rec, data_gen, xlabel, title, subplot):
        ax = plt.subplot(1,3,subplot)

        # Colors: original, rec, gen
        colors = ["blue", "orange", "green"]
        labels = [r"Original $\mathrm{H_2O}$",
                  r"Reconstructed $\mathrm{H_2O}$",
                  r"Generated $\mathrm{H_2O}$"]

        # --- Plot Histograms (always) ---
        h1 = ax.hist(data_orig, bins=70, density=True, alpha=0.3, color=colors[0])
        h2 = ax.hist(data_rec,  bins=70, density=True, alpha=0.3, color=colors[1])
        h3 = ax.hist(data_gen,  bins=70, density=True, alpha=0.3, color=colors[2])

        # --- Only Original gets a KDE Fit ---
        kde = gaussian_kde(data_orig)
        xs = np.linspace(min(data_orig), max(data_orig), 300)
        p, = ax.plot(xs, kde(xs), color=colors[0], lw=2)

        # Only add this handle once (subplot 1)
        if subplot == 1:
            legend_handles.extend([
                plt.Line2D([0], [0], color=colors[0], lw=2),   # original fit
                plt.Rectangle((0,0),1,1, fc=colors[1], alpha=0.3),  # rec hist
                plt.Rectangle((0,0),1,1, fc=colors[2], alpha=0.3)   # gen hist
            ])

        ax.set_xlabel(xlabel)
        ax.set_title(title)

    # ---- O-H(1) ----
    plot_hist_kde(
        r1_o, r1_r, r1_g,
        xlabel=r"Bond length [Å]",
        title=r"$\mathrm{O-H(1)}$ Bond Length Distribution",
        subplot=1
    )

    # ---- O-H(2) ----
    plot_hist_kde(
        r2_o, r2_r, r2_g,
        xlabel=r"Bond length [Å]",
        title=r"$\mathrm{O-H(2)}$ Bond Length Distribution",
        subplot=2
    )

    # ---- Angle ----
    plot_hist_kde(
        ang_o, ang_r, ang_g,
        xlabel=r"Angle [deg]",
        title=r"$\mathrm{H-O-H}$ Angle Distribution",
        subplot=3
    )

    # Global legend BELOW the figure
    fig.legend(
        legend_handles,
        [r"Original Fit (KDE)", r"Reconstructed Histogram", r"Generated Histogram"],
        loc="lower center",
        ncol=3,
        fontsize=12,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(outfile)
    plt.show()



def print_distribution_means(dist_original, dist_recon, dist_generated):
    names = ["O-H(1) bond", "O-H(2) bond", "H-O-H angle"]

    sets = {
        "Original": dist_original,
        "Reconstructed": dist_recon,
        "Generated": dist_generated
    }

    print("\n=== Mean values of distributions ===\n")

    for i, name in enumerate(names):
        print(f"--- {name} ---")
        for label, dist in sets.items():
            mean_value = np.mean(dist[i])
            print(f"{label:15s}: {mean_value:.5f}")
        print()

def print_distribution_stats(dist_original, dist_recon, dist_generated, ddof=1, robust=True):
    """
    Prints mean, variance, std (and optionally IQR) for each distribution:
    O-H(1), O-H(2), angle(H-O-H).
    ddof=1 -> sample variance/std
    robust=True -> prints IQR as robust width measure
    """
    names = ["O-H(1) bond", "O-H(2) bond", "H-O-H angle"]

    sets = {
        "Original": dist_original,
        "Reconstructed": dist_recon,
        "Generated": dist_generated
    }

    print("\n=== Distribution statistics ===\n")
    for i, name in enumerate(names):
        print(f"--- {name} ---")
        for label, dist in sets.items():
            x = np.asarray(dist[i])
            mean = np.mean(x)
            var  = np.var(x, ddof=ddof)
            std  = np.std(x, ddof=ddof)

            if robust:
                q25, q75 = np.percentile(x, [25, 75])
                iqr = q75 - q25
                print(f"{label:15s}: mean={mean:.5f} | var={var:.6e} | std={std:.5f} | IQR={iqr:.5f}")
            else:
                print(f"{label:15s}: mean={mean:.5f} | var={var:.6e} | std={std:.5f}")
        print()

def plot_bond_angle_kde_only(dist_original, dist_rec, dist_gen,
                            outfile="bond_angle_kde_only.pdf",
                            bw_method=None, npoints=400, xlim_quantiles=(0.5, 99.5)):
    """
    KDE-only plot: only smooth KDE lines (no histogram).
    bw_method: None/"scott"/"silverman" or float
    xlim_quantiles: robust shared x-range based on combined data
    """

    (r1_o, r2_o, ang_o) = dist_original
    (r1_r, r2_r, ang_r) = dist_rec
    (r1_g, r2_g, ang_g) = dist_gen

    fig = plt.figure(figsize=(14,5))

    def kde_line(ax, data, label):
        data = np.asarray(data)
        kde = gaussian_kde(data, bw_method=bw_method)
        return kde, label

    def plot_one(ax, datasets, xlabel, title):
        # shared x-range for fair comparison
        all_data = np.concatenate([np.asarray(d) for d, _ in datasets])
        lo, hi = np.percentile(all_data, xlim_quantiles)
        xs = np.linspace(lo, hi, npoints)

        for data, label in datasets:
            kde = gaussian_kde(np.asarray(data), bw_method=bw_method)
            ax.plot(xs, kde(xs), lw=2, label=label)

        ax.set_xlabel(xlabel)
        ax.set_title(title)

    labels = [r"Original $\mathrm{H_2O}$", r"Reconstructed $\mathrm{H_2O}$", r"Generated $\mathrm{H_2O}$"]

    # O-H(1)
    ax1 = plt.subplot(1,3,1)
    plot_one(ax1,
             [(r1_o, labels[0]), (r1_r, labels[1]), (r1_g, labels[2])],
             xlabel=r"Bond length [Å]",
             title=r"KDE: $\mathrm{O-H(1)}$ Bond Length")

    # O-H(2)
    ax2 = plt.subplot(1,3,2)
    plot_one(ax2,
             [(r2_o, labels[0]), (r2_r, labels[1]), (r2_g, labels[2])],
             xlabel=r"Bond length [Å]",
             title=r"KDE: $\mathrm{O-H(2)}$ Bond Length")

    # Angle
    ax3 = plt.subplot(1,3,3)
    plot_one(ax3,
             [(ang_o, labels[0]), (ang_r, labels[1]), (ang_g, labels[2])],
             xlabel=r"Angle [deg]",
             title=r"KDE: $\mathrm{H-O-H}$ Angle")

    # one common legend
    handles, labs = ax1.get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=3, fontsize=12, frameon=False)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    plt.savefig(outfile)
    plt.show()
