"""
Convolutional VAE

"""

import numpy as np
import torch
import torch.nn as nn


class CVAEEncoder(nn.Module):

    def __init__(self, latent_dim: int, input_dim: int, input_channels: int):
        super(CVAEEncoder, self).__init__()

        ks = (input_dim / input_channels) ** 0.5

        ks_s = [4, 2]
        strides = [2, 1]

        # (Input dim + padding - kernel dim) / stride + 1
        for i in range(0, 2):
          ks = int(np.floor((ks + 2 - ks_s[i]) / strides[i]) + 1)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, kernel_size=ks_s[0], stride=strides[0], padding=1, out_channels=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, kernel_size=ks_s[1], stride=strides[1], padding=1, out_channels=32),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(in_features=32 * ks ** 2, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=32 * ks ** 2, out_features=latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        mu = self.fc_mu(z)
        logvar = self.fc_logvar(z)

        return mu, logvar


class CVAEDecoder(nn.Module):

    def __init__(self, latent_dim: int, input_dim: int, output_channels: int):
        super(CVAEDecoder, self).__init__()

        ks = (input_dim / output_channels) ** 0.5
        
        ks_s = [4, 2]
        strides = [2, 1]

        # (Input dim + padding - kernel dim) / stride + 1
        for i in range(0, 2):
          ks = int(np.floor((ks + 2 - ks_s[i]) / strides[i]) + 1)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=32 * ks ** 2),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(32, ks, ks)),
            nn.ConvTranspose2d(in_channels=32, kernel_size=ks_s[1], stride=strides[1], padding=1, out_channels=16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, kernel_size=ks_s[0], stride=strides[0], padding=1, out_channels=output_channels),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.decoder(z)


class CVAE(nn.Module):
    """
    Convolutional Variational Autoencoder

    This network uses convolutions for the encoder 
    and transpose convolutions + linear for the decoder
    """
    
    def __init__(self, **kwargs):
        super(CVAE, self).__init__()
        self.device = kwargs.get("device", torch.device("cpu"))
        self.latent_dim = kwargs.get("latent_dim", 512)

        input_dim = kwargs.get("input_dim", 32 * 32 * 3)
        input_channels = kwargs.get("input_channels", 3)

        self.encoder = CVAEEncoder(latent_dim=self.latent_dim, input_dim=input_dim, input_channels=input_channels)
        self.decoder = CVAEDecoder(latent_dim=self.latent_dim, input_dim=input_dim, output_channels=input_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)
    
    def generate(self, n: int = 1):
        z = torch.randn((n, self.latent_dim)).to(self.device)
        x_recon = self.decoder(z)
        return x_recon
    
    @staticmethod
    def vae_loss(output, x, args):
        """
        KL-Divergence loss with BCE

        output is a tuple of the following: (reconstructed x, mu/mean, log variance)
        """
        b = 0.25
        x_recon, mu, logvar = output

        loss = nn.functional.binary_cross_entropy(x_recon, x, reduction="sum") # TODO: split
        kl_divergence = -b * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss + kl_divergence