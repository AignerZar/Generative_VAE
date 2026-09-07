# Generative_VAE

GitHub Repository for a generative neural network, more precisely a Variational Autoencoder (VAE), to sample Path Integral Monte Carlo configurations. 
By the use of those PIMC simulations it is possible to investigate different properties of quantum mechanical particles by mapping those particles onto a chain of beads connected by springs.

## PIMC configurations
For the training and validation dataset, PIMC configurations are used. Those samples were calculated and generated beforehand by using the PIMC code, developed by Michael Hütter.
The PIMC configurations are computed for H2O with different number of beads per molecule. Whereas here, a H2O molecule was used as an input. An example of such an input sample can be found in file "H2O_30beads.csv".

## Overview
In this work a Variational Autoencoder (VAE) is introduced, whereas before training the actual VAE, a simpler Autoencoder (AE) is trained, whereas the trained weights of the AE were used as and loaded into the VAE, therefore we were able to introduce a two stages approach.

It is important to denote, that the AE does not have to be trained from scratch each time the VAE is trained, only if changes in the architecture or the dimension of the latent space are done, the weights of the AE need to be retrained.

### Autoencoder
For the AE the same architecture as for the VAE was used, whereas only the loss function is much more simpler compared to the VAE. Because of the fact that the output is also computed by drawing random latent vectors $z$ from the latent space, therefore also no KL term or geometry loss is needed, leading to a much faster convergence of the loss function.

### Variational Autoencoder
The VAE is using the same architecture as the AE, whereas now for the sampling process the so called reparametrization trick was used, to be able to perform back propagation. Compared to the AE, the loss function of the VAE consisted of three different terms: The reconstruction term, the representation and the geometry par of the loss function.


## Defining the Input Parameters
In the file "config.py" various variables are defined and used, dependent on the input dataset and the architecture of the network, this file needs to be adjusted properly.

| Variable | Name and Usage | Datatype |
| :--- | :---: | ---: |
| P | Number of beads per configurations (can be defined when training the PIMC network) | int |
| num_atoms | Number of atoms per molecule | int |
| batch_size | Number of samples in one batch | int |
| Input_dim | Dimension of the input data (P * num_atoms * 3) | int |
| Latent_dim | Dimension of the latent space | int |
| Learning_rate | Learning rate, defines the step size during SGD | float |
| Validation_split | How much percent should be in the validation set | float |
| device | Device on which the algorithm and network should run | - |
| ae_epochs | How many epochs the Autoencoder should be trained | int |
| ae_learning_rate | Learning rate of the Autoencoder | float |
|evaluation_batch_size | How many samples should be in one batch for the evaluation | int |
|n_epochs | How many epochs the VAE should be trained | int |
| beta | Factor which determines the influence of the KL loss | float |
| beta_max | Factor which determines the influence of the KL loss for a progressive beta | float|
| gamma | Factor which determines the influence of the geometry loss | - |
| vae_learning_rate | Learning rate of the VAE | float |
|kl_warmup_epochs | How long it should take till beta reaches maximal value, for progressive beta | int |
| num_samples | How many new configurations should be produced | - |



## Workflow
1. Clone the whole repository -> includes input data for training and validation or use your own configurations to train the VAE
2. Download all necessary libaries used in the code -> can all be seen in the file "requirements.txt"
3. Open the file "config.py" and determine various parameters like the epoch size, the number of beads your input has, number of atoms per molecule, latent dimension, input dimension (dependent on your own input) and so on. 
4. It is also important to denote that the code works on GPU and CPU therefore if you activate a conda environment the code runs on GPU automatically if you do not have a conda environment the code runs on the CPU automatically.
5. The architecture of the whole model can be found in file "model.py", whereas here also the number of layers can be adjusted.
6. If you defined your VAE and put in all the parameters just run file "main_train.py"
7. If your training ended and stopped automatically you can run file "main_evaluate.py"
8. Now everything is finished and you should see your results and plots.

## Additional information
The code is still under work, therefore the GitHub Repository will be updated from time to time. 
