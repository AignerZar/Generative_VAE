"""
The following code is used to plot the energy distributions
"""
import numpy as np
import matplotlib.pyplot as plt

# Latex
font1 = {'family':'sans-serif','color':'black','size':12}
font2 = {'family':'sans-serif','color':'black','size':20}

plt.rcParams['text.usetex'] = True #LaTeX

# data
data_original = "/Users/zarahaigner/Documents/Arbeit/VAE/Total_energy_original_15bead_new.csv"
data_generated = "/Users/zarahaigner/Documents/Arbeit/VAE/Total_energy_generated_15bead_new.csv"

# extracting the data from the files -> different columns
col = 3

original = np.loadtxt(data_original, delimiter=",", skiprows=1)
generated = np.loadtxt(data_generated, delimiter=",", skiprows=1)

E_original = original[:, col]
E_generated = generated[:, col]

# plotting the histogram
all_E = np.concatenate([E_original, E_generated])
bins = 300
hist_range = (all_E.min(), all_E.max())

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

axes[0].hist(
    E_original,
    bins=bins,
    range=hist_range,
    density=True,
)
axes[0].set_title(r"\textbf{Original data}")
axes[0].set_xlabel(r"Energy $E\;[\mathrm{Ha}]$")
axes[0].set_ylabel(r"Probability density $p(E)$")

axes[1].hist(
    E_generated,
    bins=bins,
    range=hist_range,
    density=True,
)
axes[1].set_title(r"\textbf{Reconstructed data}")
axes[1].set_xlabel(r"Energy $E\;[\mathrm{Ha}]$")

plt.tight_layout()
plt.savefig("Energy_distributions_original_vs_reconstructed.pdf")
plt.show()

E_mean_original = np.mean(E_original)
E_mean_generated = np.mean(E_generated)

print(f"mena(E) original   = {E_mean_original:.6f} Ha")
print(f"mean(E) generated  = {E_mean_generated:.6f} Ha")
print(f"difference⟨E⟩           = {E_mean_generated - E_mean_original:.6e} Ha")

E_std_original = np.std(E_original, ddof=1)
E_std_generated = np.std(E_generated, ddof=1)

print(f"sigma(E) original = {E_std_original:.6f} Ha")
print(f"Sigma(E) generated = {E_std_generated:.6f} Ha")

error_percent = ((E_mean_generated - E_mean_original) / E_mean_generated) * 100
error_percent_sigma = ((E_std_generated - E_std_original) / E_std_generated) * 100

print(f"Deviation of the mean values = {error_percent:.6f} Percent")
print(f"Deviation of variance = {error_percent_sigma:.6f} Percent")

#E_sem_original = E_std_original / np.sqrt(len(E_original))
#E_sem_generated = E_std_generated / np.sqrt(len(E_generated))

#print(f"⟨E⟩ original  = {E_mean_original:.6f} ± {E_sem_original:.6e} Ha")
#print(f"⟨E⟩ generated = {E_mean_generated:.6f} ± {E_sem_generated:.6e} Ha")