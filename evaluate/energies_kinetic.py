"""
Code to estimate the kinetic energy -> spring between the beads
The part of the code for the kinetic estimator is taken from the PIMC code from Michael Hütter
Source: https://github.com/michael-huetter/PIMC/blob/main/main.py
"""
import numpy as np
import csv

"""
Definition and various settings
"""
# input and output files
input_data = "/Users/zarahaigner/Documents/Arbeit/VAE/data/input_generated.xyz"
output_data_file = "Energies_generated_kinetic_15beads_new.csv" 

simulation_dim = 3  # three dimensions -> 3 coordinates per atom
numParticles = 3   # number of beads per configuration
numTimeSlices = 15   # number of atoms per bead/molecule
T = 1               # Kelvin

# Boltzman constant
kB = 3.166811563e-6 # Hartree/K

# masses 
mH = 1837.1527
mO = 29156.0

lam = np.array([
    1/(2*mH),   # H
    1/(2*mO),   # O
    1/(2*mH)],   # H
    dtype=float
)

beta = 1.0 / (kB * T)
tau = beta / numTimeSlices

# kinetic estimator -> from the PIMC code
def kinetic_estimator(beads: np.array, tau: float, lam: float, numTimeSlices: int, numParticles: int) -> float:
    """
    Thermodynamic estimator for the kinetic energy. 
    """
    tot = 0.0
    for tslice in range(numTimeSlices):
        tslicep1 = (tslice + 1) % numTimeSlices
        for ptcl in range(numParticles):
            norm = 1.0/(4.0*lam[ptcl]*tau*tau)
            delR = beads[tslicep1,ptcl] - beads[tslice,ptcl]
            tot = tot - norm*np.dot(delR, delR)
        
    return (simulation_dim/2) * numParticles/tau + tot/numTimeSlices


# reading in the xyz file
def read_bead_blocks_from_file(path):
    """
    Returns a list of bead blocks.
    Each block: np.array shape (3,3) in order (H,O,H) as in the file.
    We ignore lines:
      - '3'
      - 'Time step: ...'
    """
    blocks = []
    cur = []

    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s == "3":
                # end of one block (if we have 3 atoms collected)
                if len(cur) == 3:
                    blocks.append(np.array(cur, dtype=float))
                cur = []
                continue
            if s.lower().startswith("time step"):
                continue

            parts = s.split()
            if len(parts) >= 4 and parts[0] in ("H", "O"):
                x, y, z = map(float, parts[1:4])
                cur.append([x, y, z])

    # just in case file doesn't end with '3'
    if len(cur) == 3:
        blocks.append(np.array(cur, dtype=float))

    return blocks


"""
Main part of the code
"""
blocks = read_bead_blocks_from_file(input_data)

# group into configurations of P beads
n_cfg = len(blocks) // numTimeSlices
leftover = len(blocks) % numTimeSlices
if n_cfg == 0:
    raise RuntimeError(f"Not enough bead blocks: found {len(blocks)} blocks, need at least P={numTimeSlices}.")

if leftover != 0:
    print(f"[warn] Ignoring {leftover} bead-block(s) at end (not a full configuration of P={numTimeSlices}).")

kinetic_values = []
for c in range(n_cfg):
    beads = np.stack(blocks[c*numTimeSlices:(c+1)*numTimeSlices], axis=0)  # shape (P,3,3)
    K = kinetic_estimator(beads, tau, lam, numTimeSlices, numParticles)
    kinetic_values.append((c, K))

# write CSV
with open(output_data_file, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["config_index", "K_hartree"])
    w.writerows(kinetic_values)

print(f"Done. Wrote {len(kinetic_values)} kinetic values to {output_data_file}")
print(f"tau = {tau:.6e} Eh^-1  (T={T} K, P={numTimeSlices})")
print(f"mean(K) = {np.mean([k for _, k in kinetic_values]):.6e} Eh")