"""
VTAE

"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from vtae_helper import Identity, get_transformer


class MLPEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(MLPEncoder, self).__init__()
        self.flat_dim = np.prod(input_shape).item()
        self.encoder_mu = nn.Sequential(
            nn.BatchNorm1d(self.flat_dim),
            nn.Linear(self.flat_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, latent_dim),
        )
        self.encoder_var = nn.Sequential(
            nn.BatchNorm1d(self.flat_dim),
            nn.Linear(self.flat_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, latent_dim),
            nn.Softplus(),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        z_mu = self.encoder_mu(x)
        z_var = self.encoder_var(x)
        return z_mu, z_var


class MLPDecoder(nn.Module):
    def __init__(self, output_shape, latent_dim, outputnonlin):
        super(MLPDecoder, self).__init__()
        self.flat_dim = np.prod(output_shape).item()
        self.output_shape = output_shape
        self.decoder_mu = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.flat_dim),
            outputnonlin,
        )
        self.decoder_var = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, self.flat_dim),
            nn.Softplus(),
        )

    def forward(self, z):
        x_mu = self.decoder_mu(z).reshape(-1, *self.output_shape)
        x_var = self.decoder_var(z).reshape(-1, *self.output_shape)
        return x_mu, x_var


class VTAE(nn.Module):
    def __init__(
        self,
        input_shape,
        latent_dim,
        outputdensity,
        ST_type,
        **kwargs,
    ):
        super(VTAE, self).__init__()
        # Constants
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.latent_spaces = 2
        self.outputdensity = outputdensity

        # Spatial transformer
        self.stn = get_transformer(ST_type)(input_shape)
        self.ST_type = ST_type

        # Define outputdensities
        outputnonlin = None
        if outputdensity == "bernoulli":
            outputnonlin = nn.Sigmoid()
        elif outputdensity == "gaussian":
            outputnonlin = Identity()
        else:
            ValueError("Unknown output density")

        # Define encoder and decoder
        self.encoder1 = MLPEncoder(input_shape, latent_dim)
        self.decoder1 = MLPDecoder((self.stn.dim(),), latent_dim, Identity())

        self.encoder2 = MLPEncoder(input_shape, latent_dim)
        self.decoder2 = MLPDecoder(input_shape, latent_dim, outputnonlin)

    def reparameterize(self, mu, var, eq_samples=1, iw_samples=1):
        batch_size, latent_dim = mu.shape
        eps = torch.randn(
            batch_size, eq_samples, iw_samples, latent_dim, device=var.device
        )
        return (mu[:, None, None, :] + var[:, None, None, :].sqrt() * eps).reshape(
            -1, latent_dim
        )

    def forward(self, x, eq_samples=1, iw_samples=1, switch=1.0):
        # Encode/decode transformer space
        mu1, var1 = self.encoder1(x)
        z1 = self.reparameterize(mu1, var1, eq_samples, iw_samples)
        theta_mean, theta_var = self.decoder1(z1)

        # Transform input
        x_new = self.stn(
            x.repeat(eq_samples * iw_samples, 1, 1, 1), theta_mean, inverse=True
        )

        # Encode/decode semantic space
        mu2, var2 = self.encoder2(x_new)
        z2 = self.reparameterize(mu2, var2, 1, 1)
        x_mean, x_var = self.decoder2(z2)

        # "Detransform" output
        x_mean = self.stn(x_mean, theta_mean, inverse=False)
        x_var = self.stn(x_var, theta_mean, inverse=False)
        x_var = switch * x_var + (1 - switch) * 0.02**2

        return x_mean, x_var, [z1, z2], [mu1, mu2], [var1, var2]


if __name__ == "__main__":
    input_shape = (3, 32, 32)
    latent_dim = 10
    outputdensity = "gaussian"
    ST_type = "affine"

    model = VTAE(input_shape, latent_dim, outputdensity, ST_type)
    x = torch.randn(8, *input_shape)
    x_mean, x_var, z, mu, var = model(x)
    print(x_mean.shape, x_var.shape)
