from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

from .config import OuroMRIConfig
from .ve import VisionEmbeddings
from .modeling_backbone import OuroImageBackbone
from .modeling_diffusion import DiffusionHead


def patchify(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert [B, C, H, W] image to [B, N, C, patch_size, patch_size] patches.

    Args:
        image: [B, C, H, W] input image
        patch_size: size of each square patch

    Returns:
        patches: [B, N, C, patch_size, patch_size]
    """
    B, C, H, W = image.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"Image dimensions ({H}x{W}) must be divisible by patch_size ({patch_size})"
    h_patches = H // patch_size
    w_patches = W // patch_size
    patches = image.reshape(B, C, h_patches, patch_size, w_patches, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = patches.reshape(B, h_patches * w_patches, C, patch_size, patch_size)
    return patches


def unpatchify(patches: torch.Tensor, patch_size: int, H: int, W: int) -> torch.Tensor:
    """
    Convert [B, N, C, patch_size, patch_size] patches back to [B, C, H, W] image.

    Args:
        patches: [B, N, C, patch_size, patch_size]
        patch_size: size of each square patch
        H: output image height
        W: output image width

    Returns:
        image: [B, C, H, W]
    """
    B, N, C, P, _ = patches.shape
    h_patches = H // patch_size
    w_patches = W // patch_size
    patches = patches.reshape(B, h_patches, w_patches, C, patch_size, patch_size)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    image = patches.reshape(B, C, H, W)
    return image


class OuroForImageTranslation(PreTrainedModel):
    """
    Ouro-MRI image translation model.

    Combines VE (Vision Embeddings), Ouro UT loop backbone, and
    Flow Matching diffusion head for one-to-one MRI contrast translation.

    Training: source image + noisy target → predict x_0 → MSE loss
    Inference: source image + random noise → iterative denoising (Euler)
    """

    config_class = OuroMRIConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: OuroMRIConfig):
        super().__init__(config)
        self.config = config
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.t_eps = config.t_eps
        self.num_inference_steps = config.fm_steps
        self.total_ut_steps = config.total_ut_steps

        # Compute patch dimension: C * patch_size * patch_size
        patch_dim = config.num_channels * config.patch_size * config.patch_size

        # Vision Embeddings: projects patches to Ouro hidden size
        self.ve = VisionEmbeddings(
            num_channels=config.num_channels,
            patch_size=config.patch_size,
            hidden_size=config.ve_hidden_size,
            downsample_ratio=config.downsample_ratio,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )

        # Project VE output to Ouro backbone hidden size
        self.ve_proj = nn.Linear(config.ve_hidden_size, config.hidden_size, bias=False)

        # Ouro UT loop backbone
        self.backbone = OuroImageBackbone(config)

        # Flow Matching diffusion head
        self.diffusion_head = DiffusionHead(
            input_dim=config.hidden_size,
            out_dim=patch_dim,
            dim=config.hidden_size,
            num_res_blocks=4,
            mlp_ratio=1.0,
        )

        self.post_init()

    def _get_grid_hw(self, image_shape: tuple[int, int], batch_size: int = 1) -> torch.Tensor:
        """Compute (H, W) patch grid for given image shape, repeated per image."""
        H, W = image_shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        grid = torch.tensor([[h_patches, w_patches]], device=self.device, dtype=torch.long)
        return grid.expand(batch_size, -1)

    def forward(
        self,
        source_image: torch.Tensor,
        target_image: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass with flow matching.

        Args:
            source_image: [B, C, H, W] — source contrast image
            target_image: [B, C, H, W] — target contrast image (x_0)
            t: [B] — timesteps in [0, 1]. Sampled uniformly if None.

        Returns:
            dict with 'loss' (MSE between predicted and ground-truth x_0)
        """
        B, C, H, W = source_image.shape
        device = source_image.device
        grid_hw = self._get_grid_hw((H, W), batch_size=B)

        if t is None:
            t = torch.rand(B, device=device)

        # Encode source image
        source_patches = patchify(source_image, self.patch_size)  # [B, N, C, P, P]
        N_patches = source_patches.shape[1]
        source_patches_flat = source_patches.view(B * N_patches, C, self.patch_size, self.patch_size)
        source_embeds = self.ve(source_patches_flat, grid_hw=grid_hw)  # [B*N, ve_hidden]
        source_embeds = self.ve_proj(source_embeds)  # [B*N, hidden_size]
        source_embeds = source_embeds.view(B, N_patches, self.hidden_size)

        # Patchify target and add noise: x_t = (1-t)·x_0 + t·ε
        target_patches = patchify(target_image, self.patch_size)  # [B, N, C, P, P]
        noise = torch.randn_like(target_patches)
        t_reshaped = t[:, None, None, None, None]
        noisy_patches = (1 - t_reshaped) * target_patches + t_reshaped * noise

        # Encode noisy target
        noisy_flat = noisy_patches.view(B * N_patches, C, self.patch_size, self.patch_size)
        target_embeds = self.ve(noisy_flat, grid_hw=grid_hw)  # [B*N, ve_hidden]
        target_embeds = self.ve_proj(target_embeds)  # [B*N, hidden_size]
        target_embeds = target_embeds.view(B, N_patches, self.hidden_size)

        # Add timestep embedding to target embeddings
        t_emb = self.diffusion_head.timestep_embed(t)  # [B, hidden_size]
        t_emb = t_emb.unsqueeze(1).expand(-1, N_patches, -1)
        target_embeds = target_embeds + t_emb

        # Concatenate and run backbone
        combined_embeds = torch.cat([source_embeds, target_embeds], dim=1)  # [B, 2N, C]
        outputs, hidden_states_list, gate_list = self.backbone(
            inputs_embeds=combined_embeds,
            attention_mask=None,  # bidirectional
            use_cache=False,
        )

        # Compute exit probability distribution from gates (same as OuroForCausalLM)
        pdf_list = []
        remaining_prob = torch.ones(B, combined_embeds.shape[1], device=device)
        for idx, gate_tensor in enumerate(gate_list):
            lambda_i = torch.sigmoid(gate_tensor.squeeze(-1))  # [B, 2N]
            if idx < len(gate_list) - 1:
                p_i = lambda_i * remaining_prob
                remaining_prob = remaining_prob * (1.0 - lambda_i)
            else:
                p_i = remaining_prob
            pdf_list.append(p_i)

        # Weighted x_0 prediction across UT steps
        x_0_expected = torch.zeros(
            B, N_patches, C, self.patch_size, self.patch_size, device=device
        )
        for step_idx, hidden_states in enumerate(hidden_states_list):
            target_hidden = hidden_states[:, N_patches:, :]  # [B, N, C]
            x_0_step = self.diffusion_head(target_hidden, t)
            x_0_step = x_0_step.view(B, N_patches, C, self.patch_size, self.patch_size)
            # Use target-position exit probabilities as weights
            weight = pdf_list[step_idx][:, N_patches:]  # [B, N]
            weight = weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1, 1]
            x_0_expected = x_0_expected + weight * x_0_step

        # Loss: MSE between weighted prediction and ground truth target patches
        loss = nn.functional.mse_loss(x_0_expected, target_patches)

        return {"loss": loss, "x_0_pred": x_0_expected}

    @torch.no_grad()
    def translate(
        self,
        source_image: torch.Tensor,
        num_steps: Optional[int] = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Inference: translate source image to target contrast via iterative denoising.

        Args:
            source_image: [B, C, H, W] — source contrast image
            num_steps: number of Euler integration steps (default: config.num_inference_steps)
            verbose: if True, print progress

        Returns:
            generated_image: [B, C, H, W] — translated target contrast image
        """
        B, C, H, W = source_image.shape
        device = source_image.device
        grid_hw = self._get_grid_hw((H, W), batch_size=B)
        num_steps = num_steps if num_steps is not None else self.num_inference_steps

        # Encode source image once
        source_patches = patchify(source_image, self.patch_size)
        N_patches = source_patches.shape[1]
        source_flat = source_patches.view(B * N_patches, C, self.patch_size, self.patch_size)
        source_embeds = self.ve(source_flat, grid_hw=grid_hw)
        source_embeds = self.ve_proj(source_embeds)
        source_embeds = source_embeds.view(B, N_patches, self.hidden_size)

        # Initialize from noise (x_1 in the flow: x_t = (1-t)*x_0 + t*ε)
        z = torch.randn(B, N_patches, C, self.patch_size, self.patch_size, device=device)

        # Integrate from t=1 (noise) to t=0 (clean)
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for step_i in range(num_steps):
            t = timesteps[step_i]
            t_next = timesteps[step_i + 1]
            t_batch = t.expand(B)

            # Encode current noisy target
            z_flat = z.view(B * N_patches, C, self.patch_size, self.patch_size)
            target_embeds = self.ve(z_flat, grid_hw=grid_hw)
            target_embeds = self.ve_proj(target_embeds)
            target_embeds = target_embeds.view(B, N_patches, self.hidden_size)

            # Add timestep embedding
            t_emb = self.diffusion_head.timestep_embed(t_batch)
            t_emb = t_emb.unsqueeze(1).expand(-1, N_patches, -1)
            target_embeds = target_embeds + t_emb

            # Run backbone
            combined_embeds = torch.cat([source_embeds, target_embeds], dim=1)
            outputs, _, _ = self.backbone(
                inputs_embeds=combined_embeds,
                attention_mask=None,
                use_cache=False,
            )

            # Predict x_0
            target_hidden = outputs.last_hidden_state[:, N_patches:, :]
            x_0_pred = self.diffusion_head(target_hidden, t_batch)
            x_0_pred = x_0_pred.view(B, N_patches, C, self.patch_size, self.patch_size)

            # Velocity: v = dx/dt = (x_t - x_0) / t
            denom = t.clamp_min(self.t_eps)
            v_pred = (z - x_0_pred) / denom

            # Euler step (dt < 0 since t_next < t)
            z = z + (t_next - t) * v_pred

            if verbose and (step_i % 10 == 0 or step_i == num_steps - 1):
                print(f"  Step {step_i+1}/{num_steps}, t={t:.4f}")

        # Unpatchify to get final image
        generated_image = unpatchify(z, self.patch_size, H, W)
        return generated_image


__all__ = ["OuroForImageTranslation", "patchify", "unpatchify"]
