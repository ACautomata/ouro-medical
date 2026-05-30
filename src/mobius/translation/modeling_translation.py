from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

from .config import OuroMRIConfig
from .ve import VisionEmbeddings
from .modeling_backbone import OuroImageBackbone
from .modeling_diffusion import DiffusionHead, TimestepEmbedder
from .modeling_meanflow import MeanFlowHead


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
        if config.downsample_ratio != 1.0:
            raise ValueError(
                "OuroForImageTranslation currently requires downsample_ratio=1.0 because "
                "the diffusion head predicts one output patch per input patch."
            )
        self.config = config
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.t_eps = config.t_eps
        self.num_inference_steps = config.fm_steps
        self.total_ut_steps = config.total_ut_steps
        self.fm_strategy = config.fm_strategy

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

        if config.fm_strategy == "meanflow":
            # MeanFlow: separate backbone timestep embedder + MeanFlowHead
            self.backbone_timestep_embed = TimestepEmbedder(config.hidden_size)
            self.diffusion_head = MeanFlowHead(
                input_dim=config.hidden_size,
                out_dim=patch_dim,
                dim=config.hidden_size,
                num_res_blocks=4,
                mlp_ratio=1.0,
            )
        else:
            # Standard flow matching: DiffusionHead (unchanged)
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

    def _embed_timestep_for_backbone(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timestep t for backbone conditioning.

        Standard strategy uses the DiffusionHead's embedder (shared with the head).
        MeanFlow strategy uses a dedicated backbone embedder (head has its own for (r,t)).
        """
        if self.fm_strategy == "meanflow":
            return self.backbone_timestep_embed(t)
        return self.diffusion_head.timestep_embed(t)

    def _compute_exit_pdf(
        self, gate_list: list[torch.Tensor], seq_len: int, B: int, device: torch.device,
    ) -> list[torch.Tensor]:
        """Compute exit probability distribution from gate outputs."""
        pdf_list = []
        remaining_prob = torch.ones(B, seq_len, device=device)
        for idx, gate_tensor in enumerate(gate_list):
            lambda_i = torch.sigmoid(gate_tensor.squeeze(-1))  # [B, seq_len]
            if idx < len(gate_list) - 1:
                p_i = lambda_i * remaining_prob
                remaining_prob = remaining_prob * (1.0 - lambda_i)
            else:
                p_i = remaining_prob
            pdf_list.append(p_i)
        return pdf_list

    def forward(
        self,
        source_image: torch.Tensor,
        target_image: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            source_image: [B, C, H, W] — source contrast image
            target_image: [B, C, H, W] — target contrast image (x_0)
            t: [B] — timesteps in [0, 1]. Sampled uniformly if None.

        Returns:
            dict with 'loss' and prediction tensors
        """
        B, C, H, W = source_image.shape
        device = source_image.device
        grid_hw = self._get_grid_hw((H, W), batch_size=B)

        if t is None:
            t = torch.rand(B, device=device)

        # --- Shared: encode source image ---
        source_patches = patchify(source_image, self.patch_size)
        N_patches = source_patches.shape[1]
        source_patches_flat = source_patches.view(B * N_patches, C, self.patch_size, self.patch_size)
        source_embeds = self.ve(source_patches_flat, grid_hw=grid_hw)
        source_embeds = self.ve_proj(source_embeds)
        source_embeds = source_embeds.view(B, N_patches, self.hidden_size)

        # --- Shared: create noise ---
        target_patches = patchify(target_image, self.patch_size)
        noise = torch.randn_like(target_patches)

        # --- Strategy-specific path ---
        if self.fm_strategy == "meanflow":
            return self._forward_meanflow(
                source_embeds, target_patches, noise,
                N_patches, B, C, H, W, t, grid_hw, device,
            )

        # --- Standard FM: encode noisy target and run backbone ---
        t_reshaped = t[:, None, None, None, None]
        noisy_patches = (1 - t_reshaped) * target_patches + t_reshaped * noise

        # Encode noisy target
        noisy_flat = noisy_patches.view(B * N_patches, C, self.patch_size, self.patch_size)
        target_embeds = self.ve(noisy_flat, grid_hw=grid_hw)
        target_embeds = self.ve_proj(target_embeds)
        target_embeds = target_embeds.view(B, N_patches, self.hidden_size)

        # Add timestep embedding for backbone conditioning
        t_emb = self._embed_timestep_for_backbone(t)
        t_emb = t_emb.unsqueeze(1).expand(-1, N_patches, -1)
        target_embeds = target_embeds + t_emb

        # Run backbone
        combined_embeds = torch.cat([source_embeds, target_embeds], dim=1)
        outputs, hidden_states_list, gate_list = self.backbone(
            inputs_embeds=combined_embeds,
            attention_mask=None,
            use_cache=False,
        )

        pdf_list = self._compute_exit_pdf(gate_list, combined_embeds.shape[1], B, device)

        return self._forward_standard(
            hidden_states_list, pdf_list, N_patches, B, C, t, target_patches,
        )

    def _forward_standard(
        self, hidden_states_list, pdf_list, N_patches, B, C, t, target_patches,
    ) -> dict[str, torch.Tensor]:
        """Standard flow matching: weighted x_0 prediction, MSE loss."""
        device = target_patches.device
        P = self.patch_size

        x_0_expected = torch.zeros(B, N_patches, C, P, P, device=device)
        for step_idx, hidden_states in enumerate(hidden_states_list):
            target_hidden = hidden_states[:, N_patches:, :]
            x_0_step = self.diffusion_head(target_hidden, t)
            x_0_step = x_0_step.view(B, N_patches, C, P, P)
            weight = pdf_list[step_idx][:, N_patches:]
            weight = weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x_0_expected = x_0_expected + weight * x_0_step

        loss = nn.functional.mse_loss(x_0_expected, target_patches)
        return {"loss": loss, "x_0_pred": x_0_expected}

    def _forward_meanflow(
        self, source_embeds, target_patches, noise,
        N_patches, B, C, H, W, t, grid_hw, device,
    ) -> dict[str, torch.Tensor]:
        """MeanFlow: JVP through full pipeline → v-loss.

        Defines a pipeline function f(t) → u that encapsulates the full
        forward pass (VE → backbone → head), then uses JVP to compute the
        total derivative du/dt for the MeanFlow Identity.
        """
        P = self.patch_size
        hidden_size = self.hidden_size

        # Sample interval start r
        r = torch.zeros(B, device=device)
        random_mask = torch.rand(B, device=device) < 0.5
        r[random_mask] = torch.rand(random_mask.sum(), device=device) * t[random_mask]

        # Ground-truth velocity: v = ε - x_0 (constant for linear interpolation)
        v_target = (noise - target_patches).view(B, N_patches, -1)

        # Capture float32 copies for JVP precision (M1)
        source_f32 = source_embeds.float()
        target_f32 = target_patches.float()
        noise_f32 = noise.float()

        def pipeline_fn(t_val):
            """Full model pipeline as a function of t."""
            # Interpolate z_t from target and noise
            t_r = t_val[:, None, None, None, None]
            z_t = (1 - t_r) * target_f32 + t_r * noise_f32

            # VE encode z_t
            z_flat = z_t.view(B * N_patches, C, P, P)
            tgt_embeds = self.ve(z_flat, grid_hw=grid_hw)
            tgt_embeds = self.ve_proj(tgt_embeds)
            tgt_embeds = tgt_embeds.view(B, N_patches, hidden_size)

            # Timestep embedding for backbone conditioning
            t_emb = self.backbone_timestep_embed(t_val)
            t_emb = t_emb.unsqueeze(1).expand(-1, N_patches, -1)
            tgt_embeds = tgt_embeds + t_emb

            # Backbone (UT loop)
            combined = torch.cat([source_f32, tgt_embeds], dim=1)
            _outputs, hs_list, gate_list = self.backbone(
                inputs_embeds=combined, attention_mask=None, use_cache=False,
            )

            # Exit PDF weighted combination of hidden states
            pdf_list = self._compute_exit_pdf(gate_list, combined.shape[1], B, device)
            combined_hidden = torch.zeros(B, N_patches, hidden_size, device=device)
            for step_idx, hs in enumerate(hs_list):
                target_hs = hs[:, N_patches:]
                weight = pdf_list[step_idx][:, N_patches:].unsqueeze(-1)
                combined_hidden = combined_hidden + weight * target_hs

            # Head predicts average velocity u for interval [r, t]
            u = self.diffusion_head(combined_hidden, t_val, r)
            return u

        # JVP through the full pipeline → correct total derivative
        loss, u = self.diffusion_head.compute_vloss(pipeline_fn, t, r, v_target)

        # x_0 prediction for logging: x_0 = x_t - (t-r) * u  (valid when r=0)
        u_patches = u.view(B, N_patches, C, P, P)
        t_reshaped = t[:, None, None, None, None]
        noisy_patches = (1 - t_reshaped) * target_patches + t_reshaped * noise
        dt_reshaped = (t - r)[:, None, None, None, None]
        x_0_pred = noisy_patches - dt_reshaped * u_patches

        return {"loss": loss, "x_0_pred": x_0_pred}

    @torch.no_grad()
    def translate(
        self,
        source_image: torch.Tensor,
        num_steps: Optional[int] = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Inference: translate source image to target contrast.

        For "standard": multi-step Euler ODE denoising.
        For "meanflow": MeanFlow sampling (1-step or multi-step).

        Args:
            source_image: [B, C, H, W] — source contrast image
            num_steps: number of sampling steps (default: config.fm_steps)
            verbose: if True, print progress

        Returns:
            generated_image: [B, C, H, W]
        """
        if self.fm_strategy == "meanflow":
            return self._translate_meanflow(source_image, num_steps, verbose)
        return self._translate_standard(source_image, num_steps, verbose)

    def _translate_standard(
        self, source_image: torch.Tensor, num_steps: Optional[int], verbose: bool,
    ) -> torch.Tensor:
        """Multi-step Euler ODE denoising (standard flow matching)."""
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

        z = torch.randn(B, N_patches, C, self.patch_size, self.patch_size, device=device)
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

            t_emb = self.diffusion_head.timestep_embed(t_batch)
            t_emb = t_emb.unsqueeze(1).expand(-1, N_patches, -1)
            target_embeds = target_embeds + t_emb

            combined_embeds = torch.cat([source_embeds, target_embeds], dim=1)
            outputs, hidden_states_list, gate_list = self.backbone(
                inputs_embeds=combined_embeds, attention_mask=None, use_cache=False,
            )
            pdf_list = self._compute_exit_pdf(gate_list, combined_embeds.shape[1], B, device)

            x_0_pred = torch.zeros(B, N_patches, C, self.patch_size, self.patch_size, device=device)
            for step_idx, hidden_states in enumerate(hidden_states_list):
                target_hidden = hidden_states[:, N_patches:, :]
                x_0_step = self.diffusion_head(target_hidden, t_batch)
                x_0_step = x_0_step.view(B, N_patches, C, self.patch_size, self.patch_size)
                weight = pdf_list[step_idx][:, N_patches:]
                weight = weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                x_0_pred = x_0_pred + weight * x_0_step

            denom = t.clamp_min(self.t_eps)
            v_pred = (z - x_0_pred) / denom
            z = z + (t_next - t) * v_pred

            if verbose and (step_i % 10 == 0 or step_i == num_steps - 1):
                print(f"  Step {step_i+1}/{num_steps}, t={t:.4f}")

        return unpatchify(z, self.patch_size, H, W)

    def _translate_meanflow(
        self, source_image: torch.Tensor, num_steps: Optional[int], verbose: bool,
    ) -> torch.Tensor:
        """MeanFlow sampling: x_r = x_t - (t-r) * u_θ(x_t, r, t).

        For num_steps=1: one-step generation (x_0 = ε - u_θ(ε, 0, 1)).
        For num_steps=N: split [0,1] into N intervals, apply u_θ per interval.
        """
        B, C, H, W = source_image.shape
        device = source_image.device
        grid_hw = self._get_grid_hw((H, W), batch_size=B)
        num_steps = num_steps if num_steps is not None else 1  # MeanFlow default: 1-step
        P = self.patch_size

        # Encode source image once
        source_patches = patchify(source_image, P)
        N_patches = source_patches.shape[1]
        source_flat = source_patches.view(B * N_patches, C, P, P)
        source_embeds = self.ve(source_flat, grid_hw=grid_hw)
        source_embeds = self.ve_proj(source_embeds)
        source_embeds = source_embeds.view(B, N_patches, self.hidden_size)

        # Start from noise
        z = torch.randn(B, N_patches, C, P, P, device=device)

        # Split [0, 1] into num_steps intervals
        dt = 1.0 / num_steps

        for i in range(num_steps):
            r_i = torch.full((B,), i * dt, device=device)
            t_i = torch.full((B,), (i + 1) * dt, device=device)

            # Encode current z
            z_flat = z.view(B * N_patches, C, P, P)
            target_embeds = self.ve(z_flat, grid_hw=grid_hw)
            target_embeds = self.ve_proj(target_embeds)
            target_embeds = target_embeds.view(B, N_patches, self.hidden_size)

            # Backbone timestep conditioning uses t (endpoint)
            t_emb = self.backbone_timestep_embed(t_i)
            t_emb = t_emb.unsqueeze(1).expand(-1, N_patches, -1)
            target_embeds = target_embeds + t_emb

            # Run backbone
            combined_embeds = torch.cat([source_embeds, target_embeds], dim=1)
            outputs, hidden_states_list, gate_list = self.backbone(
                inputs_embeds=combined_embeds, attention_mask=None, use_cache=False,
            )
            pdf_list = self._compute_exit_pdf(gate_list, combined_embeds.shape[1], B, device)

            # Combine hidden states across UT steps
            combined_hidden = torch.zeros(B, N_patches, self.hidden_size, device=device)
            for step_idx, hidden_states in enumerate(hidden_states_list):
                target_hidden = hidden_states[:, N_patches:, :]
                weight = pdf_list[step_idx][:, N_patches:].unsqueeze(-1)
                combined_hidden = combined_hidden + weight * target_hidden

            # Predict average velocity u for interval [r_i, t_i]
            u = self.diffusion_head(combined_hidden, t_i, r_i)  # [B, N, C*P*P]
            u = u.view(B, N_patches, C, P, P)

            # Update: x_r = x_t - (t - r) * u
            z = z - dt * u

            if verbose:
                print(f"  Step {i+1}/{num_steps}, r={r_i[0]:.4f}, t={t_i[0]:.4f}")

        return unpatchify(z, P, H, W)


__all__ = ["OuroForImageTranslation", "patchify", "unpatchify"]
