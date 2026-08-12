import csv

# Pfade für den Rückweg (Pässe an deine Ordnerstruktur an)
input_csv = "/Users/zarahaigner/Documents/Arbeit/VAE_H2O_30Beads/generated_denorm_30Beads_geomloss_long.csv"
output_xyz = "/Users/zarahaigner/Documents/Arbeit/VAE_H2O_30Beads/data/VAE_data_reconstructed_30Beads.xyz"

# Deine System-Parameter aus dem Original-Code
P = 30
num_atoms = 3
node_feat_dim = 3  # X, Y, Z

atom_labels = ["H", "O", "H"]

expected_elements = P * num_atoms * node_feat_dim  # 270

configs_count = 0

with open(input_csv, "r") as f_in, open(output_xyz, "w") as f_out:
    reader = csv.reader(f_in)

    for row in reader:
        if not row:
            continue

        coords = [float(val) for val in row]

        if len(coords) != expected_elements:
            print(
                f"Warning: Row {configs_count} has {len(coords)} instead of {expected_elements} values. Skip."
            )
            continue

        time_step = configs_count + 1
        sim_time = time_step * 10.0

        # 30 Beads pro CSV-Zeile
        for b in range(P):
            f_out.write(f"{num_atoms:12d}\n")
            f_out.write(
                f"Time step:{time_step:20d} Sim. Time [au]{sim_time:15.2f}\n"
            )

            # pro Bead: 3 Atome × 3 Koordinaten = 9 Werte
            for a in range(num_atoms):
                idx = b * num_atoms * node_feat_dim + a * node_feat_dim

                x = coords[idx]
                y = coords[idx + 1]
                z = coords[idx + 2]

                label = atom_labels[a]
                f_out.write(f"{label:<2s} {x:16.8E} {y:16.8E} {z:16.8E}\n")

        configs_count += 1

print(
    f"Done! {configs_count} Konfigurationen (mit je {P} Beads) wurden erfolgreich in {output_xyz} geschrieben."
)