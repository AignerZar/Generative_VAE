"""
Code to generate the input file (Data Tensor), to train the neural network
As an input file use the xyz file produced by the PIMC simulation as an output file a csv file is obtianed
"""
import csv

input_file = '/Users/zarahaigner/Documents/Arbeit/VAE_H2O_30Beads/data/movie.xyz'
output_file = "H2O_30Beads.csv"

P = 30 
num_atoms = 3
node_feat_dim = 3

configs = []
current_config = []
current_bead = []

with open(input_file, "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    if lines[i].strip() == "3":
        i += 2 
        bead_coords = []

        for _ in range (num_atoms):
            parts = lines[i].split()
            coords = [float(parts[1]), float(parts[2]), float(parts[3])]
            bead_coords.extend(coords)
            i += 1

        current_bead.append(bead_coords)

        if len(current_bead) == P:
            flat = []
            for bead in current_bead:
                flat.extend(bead)

            configs.append(flat)
            current_bead = []

    else:
        i += 1

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(configs)

print(f"{len(configs)} Configurations written in {output_file}")