"""MeanFlow head: one-step flow matching via average velocity prediction.

Implements the MeanFlow Identity from "Mean Flows for One-step Generative Modeling"
(Geng et al., NeurIPS 2025 Oral, arXiv:2505.13447) with the v-loss reparameterization
from "Improved Mean Flows" (iMF, arXiv:2512.02012).

Core idea:
    The head predicts the average velocity u(z_t, r, t) for the time interval [r, t].
    The MeanFlow Identity provides the training objective:
        V_θ = u_θ + (t - r) · du_θ/dt    (via JVP)
        Loss = ||V_θ - v_gt||²            where v_gt = ε - x_0

    At inference, one-step generation:
        x_0 = x_t - (t - r) · u_θ(x_t, r, t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp

from .modeling_diffusion import TimestepEmbedder, ResBlock, FinalLayer


class MeanFlowHead(nn.Module):
    """Flow matching head for MeanFlow average velocity prediction.

    Takes backbone hidden states and dual timestep (r, t) as input,
    predicts the average velocity u for the interval [r, t].

    Args:
        input_dim: Dimension of backbone hidden states.
        out_dim: Output dimension (patch dimension: C * P * P).
        dim: Internal channel dimension for residual blocks.
        num_res_blocks: Number of adaLN residual blocks.
        mlp_ratio: MLP expansion ratio in residual blocks.
    """

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        dim: int = 1024,
        num_res_blocks: int = 4,
        mlp_ratio: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.dim = dim

        # Input projection from backbone hidden size to head channels
        self.input_proj = nn.Linear(input_dim, dim)

        # Separate embedders for endpoint t and interval start r
        self.timestep_embed = TimestepEmbedder(dim)
        self.interval_embed = TimestepEmbedder(dim)

        # Project concatenated (t_emb, r_emb) → conditioning vector
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * dim, dim),
        )

        # AdaLN residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(dim, mlp_ratio=mlp_ratio)
            for _ in range(num_res_blocks)
        ])

        # Output projection
        self.final_layer = FinalLayer(dim, out_dim)

        self._init_weights()

    def _init_weights(self):
        """Xavier init + zero-init for adaLN modulation and output layers."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """Predict average velocity u for interval [r, t].

        Args:
            x: Backbone hidden states [B, N, input_dim] or [B, input_dim].
            t: Endpoint timestep [B].
            r: Interval start timestep [B].

        Returns:
            Average velocity prediction, same leading dims as out_dim.
        """
        is_3d = x.dim() == 3
        if not is_3d:
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)

        # Embed (r, t) jointly → conditioning
        t_emb = self.timestep_embed(t)  # [B, dim]
        r_emb = self.interval_embed(r)  # [B, dim]
        cond = self.time_proj(torch.cat([t_emb, r_emb], dim=-1))  # [B, dim]
        cond = cond.unsqueeze(1).expand(-1, seq_len, -1)  # [B, N, dim]

        # Residual blocks with adaLN modulation
        for block in self.res_blocks:
            x = block(x, cond)

        # Output projection
        output = self.final_layer(x)

        if not is_3d:
            output = output.squeeze(1)

        return output

    def compute_vloss(
        self,
        pipeline_fn,
        t: torch.Tensor,
        r: torch.Tensor,
        v_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MeanFlow v-loss via JVP through the full model pipeline.

        The total derivative d/dt u_θ is computed by JVP through the entire
        pipeline (VE → backbone → head), capturing all t-dependent paths:
        z_t interpolation, timestep embeddings, and the explicit t input.

        Args:
            pipeline_fn: function(t: [B]) -> u: [B, N, out_dim].
                Encapsulates the full forward pass from timestep t to
                average velocity prediction u.
            t: [B] endpoint timestep.
            r: [B] interval start timestep.
            v_target: [B, N, out_dim] ground-truth velocity (ε - x_0).

        Returns:
            (loss, u) — loss is differentiable, u is the primal prediction.
        """
        # Disable autocast for JVP precision (bf16 tangents lose ~16% relative
        # error on full model; FP32 JVP is required for correct du/dt).
        with torch.amp.autocast(device_type=t.device.type, enabled=False):
            u, du_dt = jvp(pipeline_fn, (t.float(),), (torch.ones_like(t).float(),))

        # MeanFlow Identity: V_θ = u + (t - r) · du/dt
        dt = (t - r)
        V_theta = u + dt.view(-1, 1, 1) * du_dt.detach()

        loss = F.mse_loss(V_theta, v_target.float())
        return loss, u

    def compute_uloss(
        self,
        pipeline_fn,
        t: torch.Tensor,
        r: torch.Tensor,
        v_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MeanFlow u-loss (原始 MF 论文，梯度流过 du/dt).

        与 compute_vloss 的区别：不对 du/dt 做 stop_gradient，梯度完整
        流过 JVP 计算的二阶路径。目标函数为：
            u_tgt = v_gt - (t - r) * du/dt
            loss  = ||u - u_tgt||²

        由于 du/dt 未 detach，优化过程包含对 θ 的二阶梯度项。
        这可能导致训练不稳定（iMF 论文推荐使用 v-loss 替代）。

        Args:
            pipeline_fn: function(t: [B]) -> u: [B, N, out_dim].
                封装从 t 到平均速度 u 的完整前向传播。
            t: [B] 终点时间步。
            r: [B] 区间起点时间步。
            v_target: [B, N, out_dim] 真实速度 (ε - x₀)。

        Returns:
            (loss, u) — loss 可微，u 为原始预测。
        """
        # Disable autocast for JVP precision (bf16 tangents lose ~16% relative
        # error on full model; FP32 JVP is required for correct du/dt).
        with torch.amp.autocast(device_type=t.device.type, enabled=False):
            u, du_dt = jvp(pipeline_fn, (t.float(),), (torch.ones_like(t).float(),))

        # u-loss: target = v_gt - (t - r) * du/dt, 无 detach
        dt = (t - r)
        u_target = v_target.float() - dt.view(-1, 1, 1) * du_dt

        loss = F.mse_loss(u, u_target)
        return loss, u


__all__ = ["MeanFlowHead"]
