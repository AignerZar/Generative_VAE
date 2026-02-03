# Generative_VAE
GitHub repository for a generative Variational Autoencoder to sample Path Integral Monte Carlo configurations.
In the VAE various EGCL layers are implemented, whereas here I want to cite the following papers:


## Input
For the training and validation dataset, Path Integral Monte Carlo (PIMC) configurations are used, which were calculated beforehand by Michael Hütter. The PIMC configurations are computed for H2O with 4 and 15 beads per molecule. Whereas the H2O molecule with 4 beads can be found in the file "final_input_H2O_4.csv" and the H2O molecule with 15 beads can be found in the file "input_H2O.csv".

## Workflow
1. Clone the whole repository -> includes input data for training and validation or use your own configurations to train the VAE
2. Download all necessary libaries used in the code -> can all be seen in the file "requirements.txt"
3. Open the file "config.py" and determine various parameters like the epoch size, the number of beads your input has, number of atoms per molecule, latent dimension, input dimension (dependent on your own input) and also the path names where your input data is and where you want your results to be.
4. It is also important to denote that the code works on GPU and CPU therefore if you activate a conda environment the code runs on GPU automatically if you do not have a conda environment the code runs on the CPU automatically.
5. In the files "decoder.py", "encoder.py", "VAE.py" you can also determine the number of layers your encoder and decoder should have
6. If you defined your VAE and put in all the parameters just run file "main_train.py"
7. If your training ended and stopped automatically you can run file "main_evaluate.py"
8. Now everything is finished and you should see your results and plots.

## Additional information
The code is still under work, therefore the GitHub Repository will be updated from time to time. 
