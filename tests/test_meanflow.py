"""Tests for MeanFlow head and strategy-based model dispatch."""

import torch

from mobius.translation import OuroMRIConfig, OuroForImageTranslation, MeanFlowHead
from mobius.translation.modeling_diffusion import TimestepEmbedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config(**overrides) -> OuroMRIConfig:
    defaults = dict(
        image_size=32,
        patch_size=16,
        num_channels=1,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        ve_hidden_size=64,
        downsample_ratio=1.0,
        total_ut_steps=2,
        fm_steps=2,
    )
    defaults.update(overrides)
    return OuroMRIConfig(**defaults)


# ---------------------------------------------------------------------------
# MeanFlowHead unit tests
# ---------------------------------------------------------------------------

class TestMeanFlowHead:
    def test_forward_output_shape(self):
        """MeanFlowHead produces correct output shape with dual timestep input."""
        B, N, dim, out_dim = 2, 4, 64, 16
        head = MeanFlowHead(input_dim=dim, out_dim=out_dim, dim=32, num_res_blocks=2)

        x = torch.randn(B, N, dim)
        t = torch.rand(B)
        r = torch.zeros(B)

        out = head(x, t, r)
        assert out.shape == (B, N, out_dim)

    def test_forward_2d_input(self):
        """Works with 2D (unbatched-seq) input."""
        B, dim, out_dim = 2, 64, 16
        head = MeanFlowHead(input_dim=dim, out_dim=out_dim, dim=32, num_res_blocks=2)

        x = torch.randn(B, dim)
        t = torch.rand(B)
        r = torch.zeros(B)

        out = head(x, t, r)
        assert out.shape == (B, out_dim)

    def test_vloss_backward(self):
        """MeanFlow v-loss produces finite gradients via full-pipeline JVP."""
        B, N, dim, out_dim = 1, 4, 64, 16
        head = MeanFlowHead(input_dim=dim, out_dim=out_dim, dim=32, num_res_blocks=2)

        backbone_hidden = torch.randn(B, N, dim)
        t = torch.rand(B) * 0.8 + 0.1  # avoid t near 0 or 1
        r = torch.zeros(B)
        v_target = torch.randn(B, N, out_dim)

        def pipeline_fn(t_val):
            return head(backbone_hidden, t_val, r)

        loss, u = head.compute_vloss(pipeline_fn, t, r, v_target)

        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert u.shape == (B, N, out_dim)

        loss.backward()
        params_without_grad = [
            name for name, p in head.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert not params_without_grad, f"Missing grads: {params_without_grad[:5]}"

    def test_vloss_gradient_flows_to_backbone(self):
        """Gradient flows through to backbone hidden states via pipeline JVP."""
        B, N, dim, out_dim = 1, 4, 64, 16
        head = MeanFlowHead(input_dim=dim, out_dim=out_dim, dim=32, num_res_blocks=2)

        backbone_hidden = torch.randn(B, N, dim, requires_grad=True)
        t = torch.rand(B) * 0.8 + 0.1
        r = torch.zeros(B)
        v_target = torch.randn(B, N, out_dim)

        def pipeline_fn(t_val):
            return head(backbone_hidden, t_val, r)

        loss, _ = head.compute_vloss(pipeline_fn, t, r, v_target)
        loss.backward()

        assert backbone_hidden.grad is not None, "No gradient on backbone_hidden"
        assert torch.isfinite(backbone_hidden.grad).all()

    def test_vloss_with_random_r(self):
        """v-loss works with r > 0 (not just r=0)."""
        B, N, dim, out_dim = 1, 4, 64, 16
        head = MeanFlowHead(input_dim=dim, out_dim=out_dim, dim=32, num_res_blocks=2)

        backbone_hidden = torch.randn(B, N, dim)
        t = torch.tensor([0.8])
        r = torch.tensor([0.3])
        v_target = torch.randn(B, N, out_dim)

        def pipeline_fn(t_val):
            return head(backbone_hidden, t_val, r)

        loss, _ = head.compute_vloss(pipeline_fn, t, r, v_target)
        assert torch.isfinite(loss)
        loss.backward()

    def test_vloss_jvp_numerical_correctness(self):
        """JVP du/dt matches finite difference approximation."""
        B, N, dim, out_dim = 1, 4, 64, 16
        head = MeanFlowHead(input_dim=dim, out_dim=out_dim, dim=32, num_res_blocks=2)
        head.eval()

        backbone_hidden = torch.randn(B, N, dim)
        t = torch.tensor([0.5])
        r = torch.tensor([0.2])

        def pipeline_fn(t_val):
            return head(backbone_hidden, t_val, r)

        # JVP
        with torch.no_grad():
            u_jvp, du_dt_jvp = torch.func.jvp(
                pipeline_fn, (t.float(),), (torch.ones_like(t).float(),)
            )

        # Finite difference: du/dt ≈ (u(t+eps) - u(t-eps)) / (2*eps)
        eps = 1e-4
        with torch.no_grad():
            u_plus = pipeline_fn((t + eps).float())
            u_minus = pipeline_fn((t - eps).float())
            du_dt_fd = (u_plus - u_minus) / (2 * eps)

        # The JVP computes the total derivative. Since backbone_hidden
        # is a constant (no t-dependence), JVP should match the finite diff.
        max_diff = (du_dt_jvp - du_dt_fd).abs().max().item()
        assert max_diff < 1e-2, (
            f"JVP du/dt does not match finite difference: max_diff={max_diff:.6f}"
        )

    def test_vloss_boundary_cases(self):
        """v-loss handles boundary t values without NaN/Inf."""
        B, N, dim, out_dim = 1, 4, 64, 16
        head = MeanFlowHead(input_dim=dim, out_dim=out_dim, dim=32, num_res_blocks=2)

        backbone_hidden = torch.randn(B, N, dim)

        for t_val in [0.01, 0.5, 0.99]:
            for r_val in [0.0, t_val * 0.5]:
                t = torch.tensor([t_val])
                r = torch.tensor([r_val])
                v_target = torch.randn(B, N, out_dim)

                def pipeline_fn(tv, _r=r):
                    return head(backbone_hidden, tv, _r)

                loss, u = head.compute_vloss(pipeline_fn, t, r, v_target)
                assert torch.isfinite(loss), f"Non-finite loss at t={t_val}, r={r_val}"
                assert torch.isfinite(u).all(), f"Non-finite u at t={t_val}, r={r_val}"


# ---------------------------------------------------------------------------
# Strategy dispatch tests
# ---------------------------------------------------------------------------

class TestStrategyDispatch:
    def test_standard_strategy_uses_diffusion_head(self):
        """Standard strategy creates DiffusionHead (not MeanFlowHead)."""
        config = _tiny_config(fm_strategy="standard")
        model = OuroForImageTranslation(config)
        from mobius.translation.modeling_diffusion import DiffusionHead
        assert isinstance(model.diffusion_head, DiffusionHead)

    def test_meanflow_strategy_uses_meanflow_head(self):
        """MeanFlow strategy creates MeanFlowHead + backbone_timestep_embed."""
        config = _tiny_config(fm_strategy="meanflow")
        model = OuroForImageTranslation(config)
        assert isinstance(model.diffusion_head, MeanFlowHead)
        assert hasattr(model, "backbone_timestep_embed")
        assert isinstance(model.backbone_timestep_embed, TimestepEmbedder)

    def test_standard_forward_produces_loss(self):
        """Standard forward pass returns finite loss."""
        config = _tiny_config(fm_strategy="standard")
        model = OuroForImageTranslation(config)
        model.train()

        source = torch.randn(1, 1, 32, 32)
        target = torch.randn(1, 1, 32, 32)
        result = model(source, target)

        assert "loss" in result
        assert torch.isfinite(result["loss"])
        assert "x_0_pred" in result

    def test_meanflow_forward_produces_loss(self):
        """MeanFlow forward pass returns finite loss."""
        config = _tiny_config(fm_strategy="meanflow")
        model = OuroForImageTranslation(config)
        model.train()

        source = torch.randn(1, 1, 32, 32)
        target = torch.randn(1, 1, 32, 32)
        t = torch.tensor([0.5])

        result = model(source, target, t)

        assert "loss" in result
        assert torch.isfinite(result["loss"])
        assert "x_0_pred" in result

    def test_meanflow_forward_backward(self):
        """MeanFlow loss backpropagates to all model parameters."""
        config = _tiny_config(fm_strategy="meanflow")
        model = OuroForImageTranslation(config)
        model.train()

        source = torch.randn(1, 1, 32, 32)
        target = torch.randn(1, 1, 32, 32)

        result = model(source, target)
        result["loss"].backward()

        params_without_grad = [
            name for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert not params_without_grad, f"Missing grads: {params_without_grad[:5]}"

    def test_meanflow_translate_one_step(self):
        """MeanFlow translate with num_steps=1 produces valid output."""
        config = _tiny_config(fm_strategy="meanflow")
        model = OuroForImageTranslation(config)
        model.eval()

        source = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            generated = model.translate(source, num_steps=1)

        assert generated.shape == (1, 1, 32, 32)
        assert torch.isfinite(generated).all()

    def test_meanflow_translate_multi_step(self):
        """MeanFlow translate with num_steps=2 produces valid output."""
        config = _tiny_config(fm_strategy="meanflow")
        model = OuroForImageTranslation(config)
        model.eval()

        source = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            generated = model.translate(source, num_steps=2)

        assert generated.shape == (1, 1, 32, 32)
        assert torch.isfinite(generated).all()


# ---------------------------------------------------------------------------
# Training loop tests
# ---------------------------------------------------------------------------

class TestMeanFlowTrainingLoop:
    def test_single_training_step(self):
        """One full training step with MeanFlow strategy."""
        config = _tiny_config(fm_strategy="meanflow")
        model = OuroForImageTranslation(config)
        model.train()

        source = torch.randn(1, 1, 32, 32)
        target = torch.randn(1, 1, 32, 32)
        t = torch.rand(1) * 0.8 + 0.1

        result = model(source, target, t)
        loss = result["loss"]

        assert torch.isfinite(loss)
        loss.backward()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        optimizer.zero_grad()

        # Second step
        t2 = torch.rand(1) * 0.8 + 0.1
        result2 = model(source, target, t2)
        result2["loss"].backward()
        assert torch.isfinite(result2["loss"])

    def test_loss_decreases(self):
        """MeanFlow loss decreases over multiple training steps."""
        config = _tiny_config(fm_strategy="meanflow")
        model = OuroForImageTranslation(config)
        model.train()

        source = torch.randn(1, 1, 32, 32)
        target = torch.randn(1, 1, 32, 32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(20):
            t = torch.rand(1) * 0.8 + 0.1
            result = model(source, target, t)
            loss = result["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        assert all(torch.isfinite(torch.tensor(v)) for v in losses), "NaN in losses"
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_strategy_is_standard(self):
        config = OuroMRIConfig()
        assert config.fm_strategy == "standard"

    def test_meanflow_strategy_config(self):
        config = OuroMRIConfig(fm_strategy="meanflow")
        assert config.fm_strategy == "meanflow"
