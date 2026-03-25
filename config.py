"""
Config file: parameters can be adjusted in this file
"""
import torch
# in comments the config input for the diatomic molecule can be seen 
P = 15 
num_atoms = 3  #2
seed = 42   
batch_size = 256
node_feat_dim = 3
n_epochs = 10000  # 5000
latent_dimension = 250  # keeping the same -> or is less possible but wouldnt really change the running time
latent_dimension_test = 250 #for testing a different set of hyperparameters
input_dim = 135  # 15*2*3 = 90
num_samples = 11000  # more samples -> 5000
learning_rate = 1e-3    # keeping the same 
validation_split = 0.2  # keeping the same should be still 80/20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
ensemble_size = 5   #number of VAEs to be used in Ensemble -> later should be done
use_bootstrap = True    # set to true or false, depends if using Bootstrap or not

# Path to the files -> replace for own use, the test files were used for testing different model architectures
input_file = "/Users/zarahaigner/Documents/Arbeit/VAE/final_input_H2O_15_5000MC.csv"
latent_output_file = "/Users/zarahaigner/Documents/Arbeit/VAE/latent_variables.csv"
generated_output_file = "/Users/zarahaigner/Documents/Arbeit/VAE/generated_configurations.csv"
generated_output_denorm_file = "/Users/zarahaigner/Documents/Arbeit/VAE/generated_configurations_denorm.csv"
generated_output_denorm_file_test = "/Users/zarahaigner/Documents/Arbeit/VAE/generated_configurations_denorm_test.csv"
generated_output_file_test = "/Users/zarahaigner/Documents/Arbeit/VAE/generated_configurations_test.csv"
latent_output_file_test = "/Users/zarahaigner/Documents/Arbeit/VAE/latent_variables_test.csv"
ensemble_file = "/Users/zarahaigner/Documents/Arbeit/VAE/ensemble_file.csv"
