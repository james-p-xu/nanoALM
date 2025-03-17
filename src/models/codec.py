import torch
import torch.nn as nn
from einops import rearrange
from typing import List

class VectorQuantizer(nn.Module):
    """
    VQ-VAE paper: https://arxiv.org/pdf/1711.00937
    Code references:
        - https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
        - https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
    """
    def __init__(self, embedding_dim: int, num_embeddings: int, beta: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_flattened = rearrange(z, "b d t -> b t d")

        # [B, T, 1] + [K] - [B, T, K] -> [B, T, K]
        d = torch.sum(z_flattened ** 2, dim=2, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1) - 2 * z_flattened @ self.embedding.weight.T

        min_encoding_indices = torch.argmin(d, dim=2) # [B, T]
        min_encodings = torch.zeros(min_encoding_indices.shape[0], min_encoding_indices.shape[1], self.num_embeddings, device=z.device)
        min_encodings.scatter_(2, min_encoding_indices.unsqueeze(2), 1) # [B, T, K]

        z_q = self.embedding(min_encoding_indices)
        z_q = rearrange(z_q, "b t d -> b d t")


        codebook_loss = torch.mean((z - z_q.detach()) ** 2)
        commitment_loss = torch.mean((z.detach() - z_q) ** 2)
        vq_loss = codebook_loss + self.beta * commitment_loss

        return z_q, vq_loss, {
            "perplexity": self._calculate_perplexity(min_encodings),
            "encodings": min_encodings,
            "encoding_indices": min_encoding_indices,
        }

    def _calculate_perplexity(self, min_encodings: torch.Tensor) -> torch.Tensor:
        avg_probs = torch.mean(min_encodings, dim=0) # [K]
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity


class VQVAE(nn.Module):
    """
    Deconstruct/reconstruct audio signal into/from a latent space of discrete codes.
    We will train a standard LM over the flattened codes to generate audio (flatten pattern from MusicGen).
    """
    def __init__(self, in_channels: int, embedding_dim: int, num_embeddings: int, hidden_dims: List[int] = [32, 64, 128, 256], beta: float = 0.25):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        # Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        modules.append(nn.Conv1d(hidden_dims[-1], embedding_dim, kernel_size=1))
        self.encoder = nn.Sequential(*modules)
            
        self.vq = VectorQuantizer(embedding_dim, num_embeddings, beta)

        # Decoder
        hidden_dims.reverse()
        modules = []
        modules.append(nn.Conv1d(embedding_dim, hidden_dims[0], kernel_size=1))
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.LeakyReLU()
            )
        )
        modules.append(
            nn.Sequential(
                nn.ConvTranspose1d(hidden_dims[-1], self.in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Tanh()
            )
        )
        self.decoder = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z_q, vq_loss, vq_dict = self.vq(z)
        x_recon = self.decoder(z_q)

        return x_recon, vq_loss, vq_dict
