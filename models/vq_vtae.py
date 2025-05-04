"""
VQ-VTAE

VQ-VAE with CBAM

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation: variant of channel attention

    (Found to reconstruct color hues better than default Channel Attention from CBAM paper)
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
             nn.Linear(in_channels, in_channels // reduction_ratio),
             nn.ReLU(),
             nn.Linear(in_channels // reduction_ratio, in_channels),
             nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        out = self.avg_pool(x).view(batch_size, num_channels)
        out = self.fc(out).view(batch_size, num_channels, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention
    https://arxiv.org/pdf/1807.06521v2
    """

    def __init__(self, ks: int = 7, transpose: bool = False):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, ks, padding = ks // 2, bias=False) if not transpose else nn.ConvTranspose2d(2, 1, ks, padding = ks // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)

        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    https://arxiv.org/pdf/1807.06521v2
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16, ks: int = 7, transpose: bool = False):
        super(CBAM, self).__init__()
        self.channel_attention = SqueezeExcitation(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(ks, transpose)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return x * out


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, inputs):
        # Flatten input
        flat_inputs = inputs.reshape(-1, self.embedding_dim)

        # Calculate distances
        distances = torch.cdist(flat_inputs, self.embedding.weight)

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).reshape(inputs.shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encoding_indices
    

class VQVTAEEncoder(nn.Module):
    """
    VQ-VAE Encoder with CBAM
    """

    def __init__(self, in_channels, hidden_dim):
        super(VQVTAEEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=2, stride=1)
        self.cbam1 = CBAM(hidden_dim)
        self.cbam2 = CBAM(hidden_dim)

    def forward(self, x):
        x = self.cbam1(F.relu(self.conv1(x)))
        x = self.cbam2(F.relu(self.conv2(x)))
        return x


class VQVTAEDecoder(nn.Module):
    """
    VQ-VAE Decoder with transposed CBAM
    """

    def __init__(self, hidden_dim, out_channels):
        super(VQVTAEDecoder, self).__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=1)
        self.conv_transpose2 = nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2)
        self.cbam = CBAM(in_channels=hidden_dim, transpose=True)
        
    def forward(self, x):
        x = self.cbam(F.relu(self.conv_transpose1(x)))
        x = self.conv_transpose2(x)
        return x


class VQVTAE(nn.Module):

    def __init__(self, **kwargs):
        super(VQVTAE, self).__init__()

        self.device = kwargs.get("device", torch.device("cpu"))
        self.embedding_dim = kwargs.get("embedding_dim")
        
        in_channels = kwargs.get("in_channels")
        hidden_dim = kwargs.get("hidden_dim")
        num_embeddings = kwargs.get("num_embeddings")
        commitment_cost = kwargs.get("commitment_cost", 0.25)

        self.encoder = VQVTAEEncoder(in_channels, hidden_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, self.embedding_dim, commitment_cost)
        self.decoder = VQVTAEDecoder(hidden_dim, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, _ = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        return x_recon, loss, perplexity # x_recon, commit loss, perplexity

    def encode(self, x):
      z = self.encoder(x)
      _, _, _, encoding_indices = self.vq_layer(z)
      return encoding_indices.T

    def decode(self, z_q):
      x_recon = self.decoder(z_q)
      return x_recon

    def generate(self, n: int = 1):
      # Note: does not work, embedding_dim needs to equal hidden_dim
      indices = torch.randint(low=0, high=self.embedding_dim, size=(n, 14, 14)).to(self.device)
      codebook = self.vq_layer.embedding.weight
      z_q = codebook[indices].view(-1, self.vq_layer.embedding_dim, 14, 14)
      x_recon = self.decoder(z_q)
      return x_recon
    
    @staticmethod
    def vqvtae_loss(output, x, args):
        
        b = 0.25

        x_recon = output[0]
        commit_loss = output[1]

        criterion = nn.MSELoss()

        recon_loss = criterion(x_recon, x) # TODO: split
        loss = recon_loss + b * commit_loss
        
        return loss