# Generative_VAE

GitHub Repository for a generative neural network, more precisely a Variational Autoencoder (VAE), to sample Path Integral Monte Carlo configurations. 
By the use of those PIMC simulations it is possible to investigate different properties of quantum mechanical particles by mapping those particles onto a chain of beads connected by springs.

## Input
For the training and validation dataset, PIMC configurations are used. Those samples were calculated and generated beforehand by using the PIMC code, developed by Michael Hütter.
The PIMC configurations are computed for H2O with different number of beads per molecule. Whereas here, a H2O molecule was used as an input. An example of such an input sample can be found in file "H2O_30beads.csv".

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
