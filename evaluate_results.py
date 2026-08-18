import torch
import numpy as np
import matplotlib.pyplot as plt

########################################## Function to compute bond lengths #################################################################
def compute_bond_lengths(
        x: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
    """Function to compute the bond lengths for each bead.

    Function assumes the atom ordering [H1, O, H2]

    Args:
        x (np.ndarray): Cartesian coordinates

    Returns:
        tuple[np.ndarray, np.ndarray]:
            r1: Bond length between H1 and O
            r2: Bond length between H2 and O
    """
    O  = x[:,1,:]
    H1 = x[:,0,:]
    H2 = x[:,2,:]

    r1 = np.linalg.norm(O - H1, axis=1)
    r2 = np.linalg.norm(O - H2, axis=1)
    return r1, r2


########################################### Function to compute the angle ###################################################################
def compute_angle(
        a: np.ndarray, 
        b: np.ndarray,
        c: np.ndarray
        ) -> float:
    """Function to compute the angle of the H2O molecule

    Args:
        a (np.ndarray): Cartesian coordinate of first atom
        b (np.ndarray): Cartesian coordinate of second atom
        c (np.ndarray): Cartesian coordinate of third atom

    Returns:
        float: Angle of the molecule in radians
    """
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
    return np.arccos(np.clip(cosang, -1.0, 1.0))


############################################# Function to compute angle distribution ###########################################################
def compute_angle_distribution(
        x: np.ndarray
        ) -> np.ndarray:
    """Function to compute the angle of the H2O molecule for every PIMC bead 

    Args:
        x (np.ndarray): Cartesian coordinates

    Returns:
        np.ndarray: H2O angles in degrees
    """
    O  = x[:,1,:]
    H1 = x[:,0,:]
    H2 = x[:,2,:]

    angles = []
    for i in range(len(O)):
        ang = compute_angle(H1[i], O[i], H2[i])
        angles.append(np.degrees(ang))
    return np.array(angles)

############################################### Function to obtain distributions #######################################################################
def get_distributions(
        data_tensor: torch.Tensor, 
        P: int, 
        num_atoms: int, 
        model = None, 
        mode = "original", 
        n_samples: int = 3500, 
        device = "cpu", 
        denorm_func = None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Function to extract all the bond lengths and bond angles 

    Args:
        data_tensor (torch.Tensor): Dataset containing the PIMC configurations
        P (int): Number of beads per configuration
        num_atoms (int): Number of atoms per molecule
        model (Optional[torch.nn.Module], optional): Trained VAE, needed if mode is set to reconstructed or generated. Defaults to None.
        mode (str, optional): Source of the evaluated configurations, options are original reconstructed or generated. Defaults to "original".
        n_samples (int, optional): Number of generated configurations if mode is generated. Defaults to 3500.
        device (str, optional): Device used. Defaults to "cpu".
        denorm_func (_type_, optional): Function to convert normalized coordinates back. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 
            bond1: O - H1 Bond length
            bond2: O - H2 Bond length
            angles: H-O-H angle
    """
    bond1 = []
    bond2 = []
    angles = []

    if mode == "original":
        iterator = range(len(data_tensor))
        def get_x(i):
            x = denorm_func(data_tensor[i].cpu().numpy())
            return x.reshape(P, num_atoms, 3)

    elif mode == "reconstructed":
        model.eval()
        iterator = range(len(data_tensor))
        def get_x(i):
            with torch.no_grad():
                x_flat = data_tensor[i].unsqueeze(0).to(device)
                x_hat, _, _ = model(x_flat)
                x = denorm_func(x_hat.squeeze(0).cpu().numpy())
                return x.reshape(P, num_atoms, 3)

    elif mode == "generated":
        model.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, model.encoder.fc_mu.out_features).to(device)
            x_gen = model.decoder(z).cpu().numpy()

        iterator = range(n_samples)
        def get_x(i):
            x = denorm_func(x_gen[i])
            return x.reshape(P, num_atoms, 3)

    for i in iterator:
        x = get_x(i)
        r1, r2 = compute_bond_lengths(x)
        bond1.extend(r1)
        bond2.extend(r2)
        angles.extend(compute_angle_distribution(x))

    return np.array(bond1), np.array(bond2), np.array(angles)


####################################################### Functions for plots ##################################################################################
font1 = {'family':'sans-serif','color':'black','size':12}
font2 = {'family':'sans-serif','color':'black','size':20}

plt.rcParams['text.usetex'] = True #LaTeX

def plot_bond_angle_distributions(dist_original, dist_rec, dist_gen, outfile="bond_angle_distributions_30Beads_geomloss_long.pdf"):
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


############################################ Function to print the results ###################################################################
def summary_table(dist_original, dist_recon, dist_generated, ddof=1):
    quantities = ["O-H(1)", "O-H(2)", "H-O-H"]
    datasets = [
        ("Original", dist_original),
        ("Reconstructed", dist_recon),
        ("Generated", dist_generated),
    ]

    print(f"{'Quantity':10s} {'Dataset':15s} {'Mean':>10s} {'Std':>10s}")
    print("-" * 49)

    for i, quantity in enumerate(quantities):
        for label, distributions in datasets:
            values = np.asarray(distributions[i])

            mean = np.mean(values)
            std = np.std(values, ddof=ddof)

            print(
                f"{quantity:10s} "
                f"{label:15s} "
                f"{mean:10.5f} "
                f"{std:10.5f}"
            )