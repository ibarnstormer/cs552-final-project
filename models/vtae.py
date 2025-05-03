"""
VTAE

Based on https://github.com/pshams55/VTAE/tree/main

"""

import math

import numpy as np
import torch
import torch.nn as nn

from .vtae_helper import Identity, get_transformer


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
    def __init__(self, **kwargs):
        super(VTAE, self).__init__()

        input_shape = kwargs.get("input_shape", (3, 32, 32))
        latent_dim = kwargs.get("latent_dim", 512)
        outputdensity = kwargs.get("outputdensity", "gaussian")
        ST_type = kwargs.get("ST_type", "affine")

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

    @staticmethod
    def vtae_loss(output, x, args):
        loss, recon_term, kl_term = vae_loss(x, *output, latent_dim=args["latent_dim"], outputdensity=args["outputdensity"])
        return -loss


""" ------ Loss Logic ------ """

c = -0.5 * math.log(2 * math.pi)


def vae_loss(
    x,
    x_mu,
    x_var,
    z,
    z_mus,
    z_vars,
    eq_samples=1,
    iw_samples=1,
    latent_dim=512,
    epoch=None,
    warmup=None,
    beta=0.25,
    outputdensity="gaussian",
):
    eps = 1e-5  # to control underflow in variance estimates
    weight = kl_scaling(epoch, warmup) * beta

    batch_size = x.shape[0]
    x = x.view(batch_size, 1, 1, -1)
    x_mu = x_mu.view(batch_size, eq_samples, iw_samples, -1)
    x_var = x_var.view(batch_size, eq_samples, iw_samples, -1)

    if z_mus[-1].shape[0] == batch_size:
        shape = (1, 1)
    else:
        shape = (eq_samples, iw_samples)

    z = [zs.view(-1, eq_samples, iw_samples, latent_dim) for zs in z]
    z_mus = [z_mus[0].view(-1, 1, 1, latent_dim)] + [
        m.view(-1, *shape, latent_dim) for m in z_mus[1:]
    ]
    z_vars = [z_vars[0].view(-1, 1, 1, latent_dim)] + [
        l.view(-1, *shape, latent_dim) for l in z_vars[1:]
    ]

    log_pz = [log_stdnormal(zs) for zs in z]
    log_qz = [
        log_normal2(zs, m, torch.log(l + eps)) for zs, m, l in zip(z, z_mus, z_vars)
    ]

    if outputdensity == "bernoulli":
        x_mu = x_mu.clamp(1e-5, 1 - 1e-5)
        log_px = x * x_mu.log() + (1 - x) * (1 - x_mu).log()
    elif outputdensity == "gaussian":
        log_px = log_normal2(x, x_mu, torch.log(x_var + eps), eps)
    else:
        ValueError("Unknown output density")
    a = log_px.sum(dim=3) + weight * (
        sum([p.sum(dim=3) for p in log_pz]) - sum([p.sum(dim=3) for p in log_qz])
    )
    a_max = torch.max(a, dim=2, keepdim=True)[0]  # (batch_size, nsamples, 1)
    lower_bound = torch.mean(a_max) + torch.mean(
        torch.log(torch.mean(torch.exp(a - a_max), dim=2))
    )
    recon_term = log_px.sum(dim=3).mean()
    kl_term = [(lp - lq).sum(dim=3).mean() for lp, lq in zip(log_pz, log_qz)]
    return lower_bound, recon_term, kl_term


# %%
def log_stdnormal(x):
    return c - x**2 / 2


# %%
def log_normal2(x, mean, log_var, eps=0.0):
    return c - log_var / 2 - (x - mean) ** 2 / (2 * torch.exp(log_var) + eps)


# %%
def kl_scaling(epoch=None, warmup=None):
    if epoch is None or warmup is None:
        return 1
    else:
        return float(np.min([epoch / warmup, 1]))
