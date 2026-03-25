"""
Code to combine the potential energies (computed by gaussian) and the kinetic energy (computed by the kinetic estimator) together
To combine the kinetic energy of one configuration is added to the mean value of the potential energy of the single beads
"""
import numpy as np
import csv
import re

P = 15

data_potential_energy = "/Users/zarahaigner/Documents/Arbeit/VAE/data/energies_new_generated.txt"
#data_potential_energy = "/Users/zarahaigner/Documents/Arbeit/Code/Vagrant_VAE/data/Energies_generated_data_gaussian_15beads_1000.txt"
data_kinetic_energy = "/Users/zarahaigner/Documents/Arbeit/VAE/Energies_generated_kinetic_15beads_new.csv"
output_file = "Total_energy_generated_15bead_new.csv"

# reading in the kinetic energy values
K_by_cfg = {}
with open(data_kinetic_energy, "r", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        idx = int(row["config_index"])
        K_by_cfg[idx] = float(row["K_hartree"])

# reading in the potential energy values
V_by_cfg = {}  # cfg -> list of 15 bead energies
pat = re.compile(r"ts(\d+)_bead(\d+)\.log")

with open(data_potential_energy, "r") as f:
    for line in f:
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        path, e_str = parts[0], parts[1]
        m = pat.search(path)
        if not m:
            continue
        ts = int(m.group(1))       # ts0001 -> 1
        bead = int(m.group(2))     # bead01 -> 1
        cfg = ts - 1               # cfg index 0-based to match kinetic_values.csv
        e = float(e_str)

        V_by_cfg.setdefault(cfg, []).append((bead, e))

# sorting the potential energy and computing the average over one bead condiguration
Vavg_by_cfg = {}
for cfg, bead_list in V_by_cfg.items():
    bead_list.sort(key=lambda x: x[0])  # sort by bead number
    energies = [e for _, e in bead_list]
    if len(energies) < P:
        print(f"[warn] cfg {cfg}: only {len(energies)} bead energies found (expected {P})")
    Vavg_by_cfg[cfg] = float(np.mean(energies[:P]))

# combining both the kinetic and the potential energies together
common_cfgs = sorted(set(K_by_cfg.keys()) & set(Vavg_by_cfg.keys()))
missing_K = sorted(set(Vavg_by_cfg.keys()) - set(K_by_cfg.keys()))
missing_V = sorted(set(K_by_cfg.keys()) - set(Vavg_by_cfg.keys()))

if missing_K:
    print(f"[warn] Missing K for configs: {missing_K[:10]}{'...' if len(missing_K)>10 else ''}")
if missing_V:
    print(f"[warn] Missing V for configs: {missing_V[:10]}{'...' if len(missing_V)>10 else ''}")

with open(output_file, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["config_index", "K_Eh", "Vavg_Eh", "Etotal_Eh"])
    for cfg in common_cfgs:
        K = K_by_cfg[cfg]
        Vavg = Vavg_by_cfg[cfg]
        Etot = K + Vavg
        w.writerow([cfg, K, Vavg, Etot])

print(f"Done. Wrote {len(common_cfgs)} rows to {output_file}")
print(f"Example row: cfg={common_cfgs[0]}  K={K_by_cfg[common_cfgs[0]]:.12e}  "
      f"Vavg={Vavg_by_cfg[common_cfgs[0]]:.12e}  E={K_by_cfg[common_cfgs[0]]+Vavg_by_cfg[common_cfgs[0]]:.12e}")