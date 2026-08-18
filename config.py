"""
Config file: parameters can be adjusted in this file
"""
import torch
# in comments the config input for the diatomic molecule can be seen 
P = 30 
num_atoms = 3  
seed = 42   
batch_size = 256
n_epochs = 2000
num_samples = 3500 
latent_dimension = 270  # keeping tahe same -> or is less possible but wouldnt really change the running time
input_dim = 270  # Number of beads * number of atoms * number of coordinates (3: xyz)
learning_rate = 1e-3    # keeping the same 
validation_split = 0.2  # keeping the same should be still 80/20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
use_bootstrap = True    # set to true or false, depends if using Bootstrap or not

input_file = "H2O_30Beads.csv"


