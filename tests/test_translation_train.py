"""
System tests: verify OuroMRI translation training + inference pipeline on a single sample.

No frameworks — raw PyTorch only.
"""

import torch

from mobius.translation import OuroMRIConfig, OuroForImageTranslation


def _tiny_config() -> OuroMRIConfig:
    return OuroMRIConfig(
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


def test_single_sample_training_step():
    """Full training step on one synthetic (source, target) pair."""
    config = _tiny_config()
    model = OuroForImageTranslation(config)
    model.train()

    # Single sample: 1-channel 32×32 images
    source = torch.randn(1, 1, 32, 32)
    target = torch.randn(1, 1, 32, 32)
    t = torch.rand(1)

    # --- Forward ---
    result = model(source, target, t)
    loss = result["loss"]

    assert loss.dim() == 0, f"loss should be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
    assert loss.item() >= 0, f"MSE loss should be non-negative: {loss.item()}"

    # --- Backward ---
    loss.backward()

    # Verify gradients exist on every trainable parameter
    params_without_grad = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not params_without_grad, (
        f"Missing gradients for: {params_without_grad[:5]}"
    )

    # --- Optimizer step ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad()

    # --- Second step to confirm we can keep training ---
    t2 = torch.rand(1)
    result2 = model(source, target, t2)
    result2["loss"].backward()
    optimizer.step()

    assert torch.isfinite(result2["loss"]), "loss diverged on second step"


def test_loss_decreases_over_multiple_steps():
    """Run several steps and confirm loss decreases on average."""
    config = _tiny_config()
    model = OuroForImageTranslation(config)
    model.train()

    source = torch.randn(1, 1, 32, 32)
    target = torch.randn(1, 1, 32, 32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(20):
        t = torch.rand(1)
        result = model(source, target, t)
        loss = result["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(l)) for l in losses), "NaN/Inf in losses"
    # Over 20 steps with AdamW on a single sample, loss should trend down
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


def _train_to_convergence(model, source, target, lr=1e-3, max_steps=2000, threshold=1e-3):
    """Helper: train model on a single (source, target) pair until convergence."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    initial_loss = None
    for step in range(max_steps):
        t = torch.rand(1)
        result = model(source, target, t)
        loss = result["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step == 0:
            initial_loss = loss.item()
        if loss.item() < threshold:
            return step, initial_loss, loss.item()
    return max_steps - 1, initial_loss, loss.item()


def test_single_sample_convergence():
    """Slow test: train on one sample until loss converges to near-zero."""
    config = _tiny_config()
    model = OuroForImageTranslation(config)
    model.train()

    torch.manual_seed(0)
    source = torch.randn(1, 1, 32, 32)
    target = torch.randn(1, 1, 32, 32)

    step, initial, final = _train_to_convergence(model, source, target)
    print(f"  Initial loss: {initial:.6f}")
    print(f"  Converged at step {step}, loss: {final:.6f}")
    assert final < 1e-3, f"Did not converge: {final:.6f}"


def test_translate_matches_training():
    """Slow test: after training on one sample, translate(source) should ≈ target."""
    config = _tiny_config()
    model = OuroForImageTranslation(config)
    model.train()

    torch.manual_seed(0)
    source = torch.randn(1, 1, 32, 32)
    target = torch.randn(1, 1, 32, 32)

    step, initial, final = _train_to_convergence(model, source, target)
    print(f"  Training: initial={initial:.6f}, converged at step {step}, loss={final:.6f}")

    model.eval()
    torch.manual_seed(99)
    generated = model.translate(source, num_steps=50)

    mae = (generated - target).abs().mean().item()
    print(f"  Inference MAE vs target: {mae:.6f}")
    assert mae < 0.1, f"translate output too far from target: MAE={mae:.4f}"


if __name__ == "__main__":
    import sys
    slow = "--slow" in sys.argv

    print("Running test_single_sample_training_step ...")
    test_single_sample_training_step()
    print("PASSED\n")

    print("Running test_loss_decreases_over_multiple_steps ...")
    test_loss_decreases_over_multiple_steps()
    print("PASSED\n")

    if slow:
        print("Running test_single_sample_convergence (slow) ...")
        test_single_sample_convergence()
        print("PASSED\n")

        print("Running test_translate_matches_training (slow) ...")
        test_translate_matches_training()
        print("PASSED\n")

    print("All tests passed.")
