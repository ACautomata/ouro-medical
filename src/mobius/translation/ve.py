import torch
import torch.nn as nn
from typing import Optional


def precompute_rope_freqs_sincos(
    dim: int, max_position: int, base: float = 10000.0, device=None
):
    """Precompute RoPE cos/sin values (1D)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_position, device=device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def build_abs_positions_from_grid_hw(grid_hw: torch.Tensor, device=None):
    """Compute patch coordinates (x, y) from grid (H, W) per image."""
    device = grid_hw.device
    B = grid_hw.shape[0]
    H = grid_hw[:, 0]
    W = grid_hw[:, 1]
    N = H * W
    N_total = N.sum()

    patch_to_sample = torch.repeat_interleave(torch.arange(B, device=device), N)
    patch_id_within_image = torch.arange(N_total, device=device)
    patch_id_within_image = patch_id_within_image - torch.cumsum(
        torch.cat([torch.tensor([0], device=device), N[:-1]]), dim=0
    )[patch_to_sample]

    W_per_patch = W[patch_to_sample]
    abs_x = patch_id_within_image % W_per_patch
    abs_y = patch_id_within_image // W_per_patch

    return abs_x, abs_y


def apply_rotary_emb_1d(
    x: torch.Tensor,
    cos_cached: torch.Tensor,
    sin_cached: torch.Tensor,
    positions: torch.Tensor,
):
    """Apply 1D RoPE to a portion of the input tensor."""
    cos = cos_cached[positions]
    sin = sin_cached[positions]

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = rotated_x1
    x_rotated[..., 1::2] = rotated_x2
    return x_rotated


def apply_2d_rotary_pos_emb(
    x: torch.Tensor,
    cos_cached_x: torch.Tensor,
    sin_cached_x: torch.Tensor,
    cos_cached_y: torch.Tensor,
    sin_cached_y: torch.Tensor,
    abs_positions_x: torch.Tensor,
    abs_positions_y: torch.Tensor,
):
    """Apply 2D RoPE: first half of dim gets X rotation, second half gets Y rotation."""
    dim = x.shape[-1]
    dim_half = dim // 2

    x_part_1 = x[..., :dim_half]
    x_part_2 = x[..., dim_half:]

    rotated_part_1 = apply_rotary_emb_1d(
        x_part_1, cos_cached_x, sin_cached_x, abs_positions_x
    )
    rotated_part_2 = apply_rotary_emb_1d(
        x_part_2, cos_cached_y, sin_cached_y, abs_positions_y
    )

    return torch.cat((rotated_part_1, rotated_part_2), dim=-1)


class VisionEmbeddings(nn.Module):
    """
    Vision Embedding module for image-to-image translation.

    Conv2d patch embedding → GELU → 2D RoPE → optional downsample.

    Adapted from NEOVisionEmbeddings in src/mobius/neo/modeling_neo_vit.py:112-191.
    Key change: supports single-channel input (num_channels=1 for grayscale MRI).
    """

    def __init__(
        self,
        num_channels: int = 1,
        patch_size: int = 16,
        hidden_size: int = 1024,
        downsample_ratio: float = 1.0,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 10000,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.downsample_factor = int(1 / downsample_ratio) if downsample_ratio < 1.0 else 1

        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        if self.downsample_factor > 1:
            self.dense_embedding = nn.Conv2d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=self.downsample_factor,
                stride=self.downsample_factor,
            )
        else:
            self.dense_embedding = None

        self.gelu = nn.GELU()

        self.rope_dim_part = hidden_size // 2
        cos_x, sin_x = precompute_rope_freqs_sincos(
            self.rope_dim_part, max_position_embeddings, base=rope_theta
        )
        cos_y, sin_y = precompute_rope_freqs_sincos(
            self.rope_dim_part, max_position_embeddings, base=rope_theta
        )
        self.register_buffer("cos_cached_x", cos_x, persistent=False)
        self.register_buffer("sin_cached_x", sin_x, persistent=False)
        self.register_buffer("cos_cached_y", cos_y, persistent=False)
        self.register_buffer("sin_cached_y", sin_y, persistent=False)

    def _apply_2d_rotary_pos_emb(
        self, patch_embeds: torch.Tensor, grid_hw: torch.Tensor
    ) -> torch.Tensor:
        abs_pos_x, abs_pos_y = build_abs_positions_from_grid_hw(
            grid_hw, device=patch_embeds.device
        )
        embeddings = apply_2d_rotary_pos_emb(
            patch_embeds.to(torch.float32),
            self.cos_cached_x,
            self.sin_cached_x,
            self.cos_cached_y,
            self.sin_cached_y,
            abs_pos_x,
            abs_pos_y,
        ).to(self.patch_embedding.weight.dtype)
        return embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_hw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, N, num_channels, patch_size, patch_size]
                or [N_total, num_channels, patch_size, patch_size]
                Flattened patch tensor where each patch is a (C, P, P) image.
            grid_hw: [B, 2] tensor with (H, W) patch counts per image.
                If None, assumes a single square image (H=W=sqrt(N)).

        Returns:
            embeddings: [N_total, hidden_size] — patch embeddings
        """
        N_total = pixel_values.shape[0]

        if pixel_values.dim() == 5:
            B, N, C, P, _ = pixel_values.shape
            pixel_values = pixel_values.view(B * N, C, P, P)
            N_total = B * N
        else:
            B = 1

        if grid_hw is None:
            H = W = int(N_total ** 0.5)
            grid_hw = torch.tensor([[H, W]], device=pixel_values.device)

        patch_embeds = self.gelu(self.patch_embedding(pixel_values)).view(
            -1, self.hidden_size
        )

        # Move RoPE buffers to correct device
        self.cos_cached_x = self.cos_cached_x.to(patch_embeds.device)
        self.sin_cached_x = self.sin_cached_x.to(patch_embeds.device)
        self.cos_cached_y = self.cos_cached_y.to(patch_embeds.device)
        self.sin_cached_y = self.sin_cached_y.to(patch_embeds.device)

        patch_embeds = self._apply_2d_rotary_pos_emb(patch_embeds, grid_hw)

        if self.dense_embedding is not None:
            patches_list = []
            cur_position = 0
            for i in range(grid_hw.shape[0]):
                h, w = grid_hw[i]
                patches_per_img = (
                    patch_embeds[cur_position : cur_position + h * w]
                    .view(h, w, -1)
                    .unsqueeze(0)
                )
                patches_per_img = self.dense_embedding(
                    patches_per_img.permute(0, 3, 1, 2)
                )
                patches_per_img = patches_per_img.permute(0, 2, 3, 1)
                patches_list.append(
                    patches_per_img.reshape(-1, patches_per_img.shape[-1])
                )
                cur_position += h * w
            embeddings = torch.cat(patches_list, dim=0)
        else:
            embeddings = patch_embeds

        return embeddings
