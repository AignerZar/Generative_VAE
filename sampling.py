"""
Code for the definition used for posterior sampling, prior sampling and decoding as well as encoding
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple



def encode_in_batches(
    model,
    data: torch.Tensor,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to put PIMC data through the encoder, putting PIMC data through in batches

    Args:
        model: VAE model
        data (torch.Tensor): PIMC data
        batch_size (int): number of configurations per batch 
        device (str): 

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
    """
    all_mu = []
    all_logvar = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = data[start : start + batch_size].to(device)
            mu, logvar = model.encoder(batch)
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())

    return torch.cat(all_mu, dim=0), torch.cat(all_logvar, dim=0)


def decode_in_batches(
    model,
    latent: torch.Tensor,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Function to put the PIMC data through the decoder

    Args:
        model: VAE model
        latent (torch.Tensor): latent space
        batch_size (int): Number of confgiurations per batch
        device (str):

    Returns:
        torch.Tensor: 
    """
    outputs = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(latent), batch_size):
            batch = latent[start : start + batch_size].to(device)
            decoded = model.decoder(batch)
            outputs.append(decoded.reshape(decoded.size(0), -1).cpu())

    return torch.cat(outputs, dim=0)


def sample_posterior(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """Function for the reparametrization trick, introduce noise!

    Args:
        mu (torch.Tensor): Mean value
        logvar (torch.Tensor): Logirthm of the variance

    Returns:
        torch.Tensor: z = mu + sigma * epsilon
    """
    posterior_std = torch.exp(0.5 * logvar) #compute sigma
    return mu + posterior_std * torch.randn_like(posterior_std)


def sample_aggregated_posterior(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    number_of_samples: int,
) -> torch.Tensor:
    """Function takes randomly posterior distributions

    Args:
        mu (torch.Tensor): Mean values
        logvar (torch.Tensor): Logarithm of the variance
        number_of_samples (int): Number of samples

    Returns:
        torch.Tensor: random sample of a posterior distribution
    """
    indices = torch.randint(0, len(mu), (number_of_samples,))
    return sample_posterior(mu[indices], logvar[indices])


def sample_standard_normal(
    number_of_samples: int,
    latent_dimension: int,
) -> torch.Tensor:
    """Producing independent PIMC samples

    Args:
        number_of_samples (int): Number of samples to be generated
        latent_dimension (int): Dimension of latent space

    Returns:
        torch.Tensor: Returnign randomly drawn samples
    """
    return torch.randn(number_of_samples, latent_dimension)

