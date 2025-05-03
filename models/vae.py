"""
Vanilla VAE

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()

        latent_dim = kwargs.get("latent_dim", 512)
        input_dim = kwargs.get("input_dim", 32 * 32 * 3)
        hidden_dim = kwargs.get("hidden_dim", 1024)

        # Encoder layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(3, 32, 32))
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
    
    def encoder(self, x):
        x = self.flatten(x)
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decoder(self, z):
        h = torch.relu(self.fc2(z))
        x_recon = torch.sigmoid(self.fc3(h))
        x_recon = self.unflatten(x_recon)
        return x_recon
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    @staticmethod
    def vae_loss(output, x, args):
        recon_x, mu, logvar = output
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum') # TODO: split
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
