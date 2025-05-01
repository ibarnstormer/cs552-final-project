"""
VQ-VTAE

Transformer-based VQ-VAE

VQ-VAE reference https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py as basis

TODO: not complete, major refactors needed

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        flat_x = x.reshape(-1, self.embedding_dim)
        distances = torch.cdist(flat_x, self.embeddings.weight)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embeddings(encoding_indices).reshape(x.shape)
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = x + (quantized - x).detach()
        return quantized, loss, encoding_indices
    

class PatchEmbed(nn.Module):
    """
    Patch embeddings from Vision Transformer
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.patch_embed(x).flatten(2).transpose(1, 2)
    

class VQVTAEEncoder(nn.Module):
    """
    Very similar to Vision Transformer
    """

    def __init__(self, latent_dim: int, input_dim: int, input_channels: int, patch_size: int, embed_dim: int, attn_heads: int):
        super(VQVTAEEncoder, self).__init__()

        self.patch_embed = PatchEmbed(input_dim, patch_size, input_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.n_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.attn = nn.MultiheadAttention(embed_dim, attn_heads, dropout=0.1, batch_first=True)

        self.norm = nn.LayerNorm(embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.linear = nn.Linear(embed_dim, latent_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.norm(x)
        attn, _ = self.attn(x, x, x)
        x = x + attn
        x = x + self.fc(self.norm(x))

        x = self.norm(x)[:, 0]
        x = self.linear(x)

        return x


class VQVTAEDecoder(nn.Module):
    """
    Reconstruct / sample from latent space
    Use VQVAE decoder for now 
    (Regenerating from latent space using some transformer hybrid may not be more beneficial than an ordinary decoder)
    """

    def __init__(self, input_dim, embedding_dim, hidden_dim, out_channels):
        super(VQVTAEDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, input_dim),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(embedding_dim, 1, 1)),
        )

        self.conv1 = nn.Conv2d(embedding_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv_transpose1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=1)
        self.conv_transpose2 = nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv_transpose1(x))
        x = self.conv_transpose2(x)
        return x


class VQVTAE(nn.Module):

    def __init__(self, **kwargs):
        super(VQVTAE, self).__init__()

        latent_dim = kwargs.get("latent_dim", 512)
        input_dim = kwargs.get("input_dim", 32 * 32 * 3)
        hidden_dim = kwargs.get("hidden_dim", 16)
        input_channels = kwargs.get("input_channels", 3)
        patch_size = kwargs.get("patch_size", 4)
        embed_dim = kwargs.get("embed_dim", 768)
        num_embed = kwargs.get("num_embed", 10)
        attn_heads = kwargs.get("attn_heads", 12)

        self.encoder = VQVTAEEncoder(latent_dim, int((input_dim / input_channels) ** 0.5), input_channels, patch_size, embed_dim, attn_heads)
        self.vq_layer = VectorQuantizer(num_embed, latent_dim, 0.25)
        self.decoder = VQVTAEDecoder(latent_dim, hidden_dim, input_channels)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, _ = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss