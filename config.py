"""
Config file: parameters can be adjusted in this file
"""
import torch

P = 15
num_atoms = 3
seed = 42
batch_size = 256
node_feat_dim = 3
n_epochs = 5000
latent_dimension = 150
latent_dimension_test = 150 #for testing a different set of hyperparameters
input_dim = 135
num_samples = 2238
learning_rate = 1e-3
validation_split = 0.2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
ensemble_size = 5   #number of VAEs to be used in Ensemble
use_bootstrap = True    # set to true or false, depends if using Bootstrap or not

# Path to the files -> replace for own use, the test files were used for testing different model architectures
input_file = "/Users/zarahaigner/Documents/Arbeit/Code/Vagrant_VAE/input_H2O.csv"
latent_output_file = "/Users/zarahaigner/Documents/Arbeit/Code/Vagrant_VAE/latent_variables.csv"
generated_output_file = "/Users/zarahaigner/Documents/Arbeit/Code/Vagrant_VAE/generated_configurations.csv"
generated_output_denorm_file = "/Users/zarahaigner/Documents/Arbeit/Code/Vagrant_VAE/generated_configurations_denorm.csv"
generated_output_denorm_file_test = "/Users/zarahaigner/Documents/Arbeit/Code/Vagrant_VAE/generated_configurations_denorm_test.csv"
generated_output_file_test = "/Users/zarahaigner/Documents/Arbeit/Code/Vagrant_VAE/generated_configurations_test.csv"
latent_output_file_test = "/Users/zarahaigner/Documents/Arbeit/Code/Vagrant_VAE/latent_variables_test.csv"
ensemble_file = "/Users/zarahaigner/Documents/Arbeit/Code/Vagrant_VAE/ensemble_file.csv"