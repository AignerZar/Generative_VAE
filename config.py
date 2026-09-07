import torch

# architecture and input 
P = 30 
num_atoms = 3  
seed = 42   
batch_size = 256
input_dim = 270 # Number of beads * number of atoms * number of coordinates (3: xyz)
latent_dimension = 48
learning_rate = 5e-5    
validation_split = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# training and evaluation
ae_epochs = 100
ae_learning_rate = 1e-3

evaluation_batch_size = 128
n_epochs = 1000 # hier vielleicht besser 500 probieren?????
beta = 0.01
beta_max = 0.02
gamma = 1.0
vae_learning_rate = 1e-4
kl_warmup_epochs = 50
num_samples = 3500 
save_loss_plot = True


# files and checkpoints
input_file = "H2O_30Beads.csv"
vae_checkpoint = "vae_h2o_beta001.pt"

ae_checkpoint = "ae_h2o_best.pt"
preprocessing_file = "preprocessing_h2o_30Beads.npz"

save_loss_plot = True
ae_loss_plot = "loss_ae.pdf"

evaluation_plot = "vae_geometry_evaluation.pdf"

generated_aggregated_file = "generated_aggregated_aligned_angstrom.csv"
generated_prior_file = "generated_prior_aligned_angstrom.csv"