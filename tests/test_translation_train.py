"""
System test: verify OuroMRI translation training pipeline on a single sample.

Runs forward → loss → backward → optimizer step with a tiny model on CPU.
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


def test_single_sample_convergence():
    """Slow test: train on one sample until loss converges to near-zero."""
    config = _tiny_config()
    model = OuroForImageTranslation(config)
    model.train()

    torch.manual_seed(0)
    source = torch.randn(1, 1, 32, 32)
    target = torch.randn(1, 1, 32, 32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    convergence_threshold = 1e-3
    max_steps = 2000

    for step in range(max_steps):
        t = torch.rand(1)
        result = model(source, target, t)
        loss = result["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step == 0:
            initial_loss = loss.item()
            print(f"  Initial loss: {initial_loss:.6f}")

        if loss.item() < convergence_threshold:
            print(f"  Converged at step {step}, loss: {loss.item():.6f}")
            return

    raise AssertionError(
        f"Did not converge after {max_steps} steps (initial: {initial_loss:.6f}, last: {loss.item():.6f})"
    )


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

    print("All tests passed.")
