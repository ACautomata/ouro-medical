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
        """MeanFlow v-loss produces finite gradients."""
        B, N, dim, out_dim = 1, 4, 64, 16
        head = MeanFlowHead(input_dim=dim, out_dim=out_dim, dim=32, num_res_blocks=2)

        backbone_hidden = torch.randn(B, N, dim)
        t = torch.rand(B) * 0.8 + 0.1  # avoid t near 0 or 1
        r = torch.zeros(B)
        v_target = torch.randn(B, N, out_dim)

        loss, u = head.compute_vloss(backbone_hidden, t, r, v_target)

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
        """Gradient flows through to backbone hidden states."""
        B, N, dim, out_dim = 1, 4, 64, 16
        head = MeanFlowHead(input_dim=dim, out_dim=out_dim, dim=32, num_res_blocks=2)

        backbone_hidden = torch.randn(B, N, dim, requires_grad=True)
        t = torch.rand(B) * 0.8 + 0.1
        r = torch.zeros(B)
        v_target = torch.randn(B, N, out_dim)

        loss, _ = head.compute_vloss(backbone_hidden, t, r, v_target)
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

        loss, _ = head.compute_vloss(backbone_hidden, t, r, v_target)
        assert torch.isfinite(loss)
        loss.backward()


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
