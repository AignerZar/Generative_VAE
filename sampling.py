import torch

######################################## Sampling Function #################################################################
def sample_from_aggregated_posterior(
        mu_all, 
        logvar_all,
        n_samples, 
        device
    ):
    # Zufällige mu aus dem gesamten Datensatz auswählen
    idx = torch.randint(
        low = 0, 
        high = mu_all.size(0), 
        size=(n_samples,), 
        device=mu_all.device,
    )

    mu = mu_all[idx].to(device)
    logvar = logvar_all[idx].to(device)

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    z = mu + eps * std

    return z


####################################### Function for generating process ####################################################
def generate_from_latent(model, z, mean, std, device):
    """
    Generate real-space positions from latent vectors z.
    """
    model.eval()
    with torch.no_grad():
        x_gen = model.decoder(z.to(device)).cpu().numpy()

    # denormalize
    x_denorm = x_gen * std + mean
    return x_gen, x_denorm


######################################## Computing the latent space ########################################################
def compute_latent(model, data_tensor, device):
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encoder(data_tensor.to(device))
        z = model.reparameterize(mu, logvar)

    return mu.cpu().numpy(), logvar.cpu().numpy(), z.cpu().numpy()

