"""
Diffusion head components for flow matching in OuroMRI translation.

This module provides the diffusion head architecture for predicting clean image
features from noisy inputs conditioned on timesteps.
"""

import torch
import torch.nn as nn
import math


def modulate(x, shift, scale=None):
    """Apply adaptive layer normalization modulation.

    Args:
        x: Input tensor
        shift: Additive shift term
        scale: Multiplicative scale term

    Returns:
        Modulated tensor
    """
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using sinusoidal embeddings.

    Follows GLIDE style timestep embedding with MLP projection.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: A 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.

        Returns:
            An (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """Embed timesteps through sinusoidal encoding and MLP projection."""
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class ResBlock(nn.Module):
    """
    Residual block with adaptive layer normalization (adaLN) modulation.

    Uses LayerNorm with conditioning from timestep embeddings to modulate
    the hidden representations.
    """

    def __init__(self, channels, mlp_ratio=1.0):
        super().__init__()
        self.channels = channels
        self.intermediate_size = int(channels * mlp_ratio)

        self.in_ln = nn.LayerNorm(self.channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, self.intermediate_size),
            nn.SiLU(),
            nn.Linear(self.intermediate_size, self.channels),
        )

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x, y):
        """Apply residual block with conditioning y."""
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer for diffusion output prediction.

    Applies layer normalization followed by linear projection to output dimension.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)

    def forward(self, x):
        """Project from model channels to output channels."""
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class DiffusionHead(nn.Module):
    """
    Diffusion prediction head with adaptive layer normalization.

    Takes backbone hidden states and timestep conditioning to predict
    clean image features for flow matching.

    Args:
        input_dim: Dimension of input hidden states (ouro_hidden_size)
        out_dim: Output dimension (VE patch dimension)
        dim: Model channel dimension for internal computations
        num_res_blocks: Number of residual blocks
    """

    def __init__(
        self,
        input_dim,
        out_dim,
        dim=1536,
        num_res_blocks=12,
        mlp_ratio=1.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.dim = dim
        self.num_res_blocks = num_res_blocks

        # Input projection from Ouro hidden size to model channels
        self.input_proj = nn.Linear(input_dim, dim)

        # Timestep embedder for conditioning
        self.timestep_embed = TimestepEmbedder(dim)

        # Residual blocks with adaLN modulation
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResBlock(dim, mlp_ratio=mlp_ratio))

        self.res_blocks = nn.ModuleList(res_blocks)

        # Final output layer
        self.final_layer = FinalLayer(dim, out_dim)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights with specific zeroing for adaLN modulation layers."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t):
        """
        Apply the diffusion head to input with timestep conditioning.

        Args:
            x: Input hidden states [batch, seq_len, input_dim] or [batch, input_dim]
            t: Timestep tensor [batch]

        Returns:
            Predicted clean features [batch, seq_len, out_dim] or [batch, out_dim]
        """
        # Handle both 2D and 3D input
        is_3d = x.dim() == 3
        if not is_3d:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]

        batch_size, seq_len, _ = x.shape

        # Project input to model dimensions
        x = self.input_proj(x)  # [batch, seq_len, dim]

        # Embed timesteps and expand to sequence length
        t_emb = self.timestep_embed(t)  # [batch, dim]
        y = t_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, dim]

        # Apply residual blocks with conditioning
        for block in self.res_blocks:
            x = block(x, y)

        # Final output projection
        output = self.final_layer(x)  # [batch, seq_len, out_dim]

        if not is_3d:
            output = output.squeeze(1)  # [batch, out_dim]

        return output