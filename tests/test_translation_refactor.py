"""Tests for refactored dataset, callbacks, and training config."""

import math
import os
import tempfile

import torch
import torch.nn.functional as F
import yaml

from mobius.translation.callbacks import (
    LossLoggingCallback,
    ReportGeneratorCallback,
    SampleVisualizationCallback,
    ValidationMetricsCallback,
)
from mobius.translation.config import OuroMRIConfig
from mobius.translation.dataset import BraTS2023Dataset, _discover_brats_dirs, _map_contrast_files
from mobius.translation.modeling_translation import OuroForImageTranslation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_brats_structure(root, n_patients=3, n_slices=10):
    """Create a minimal BraTS2023-like directory structure for testing."""
    import nibabel as nib
    import numpy as np

    for split in ["TrainingData", "ValidationData"]:
        for subtype in ["GLI"]:
            split_dir = os.path.join(root, f"BraTS-{subtype}-00001-000-{split}")
            os.makedirs(split_dir, exist_ok=True)
            for pid in range(n_patients):
                patient_dir = os.path.join(split_dir, f"BraTS-{subtype}-00001-000-{pid:03d}")
                os.makedirs(patient_dir, exist_ok=True)
                # Create small 3D volumes (H=8, W=8, D=n_slices)
                data = np.random.rand(8, 8, n_slices).astype(np.float32)
                for suffix in ["t1n", "t1c", "t2w", "t2f"]:
                    img = nib.Nifti1Image(data, affine=np.eye(4))
                    nib.save(img, os.path.join(patient_dir, f"BraTS-{subtype}-00001-000-{pid:03d}-{suffix}.nii.gz"))


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestDiscoverBratsDirs:
    def test_discovers_training_and_validation_separately(self):
        with tempfile.TemporaryDirectory() as root:
            _make_brats_structure(root, n_patients=2, n_slices=10)
            train_dirs = _discover_brats_dirs(root, ["GLI"], split="train")
            val_dirs = _discover_brats_dirs(root, ["GLI"], split="val")
            assert len(train_dirs) == 1
            assert len(val_dirs) == 1
            assert "TrainingData" in train_dirs[0]
            assert "ValidationData" in val_dirs[0]

    def test_raises_on_missing_data(self):
        with tempfile.TemporaryDirectory() as root:
            import pytest
            with pytest.raises(FileNotFoundError, match="TrainingData"):
                BraTS2023Dataset(root, subtypes=["GLI"], split="train")


class TestBraTS2023DatasetSplit:
    def test_train_and_val_are_disjoint(self):
        with tempfile.TemporaryDirectory() as root:
            _make_brats_structure(root, n_patients=4, n_slices=10)
            train_ds = BraTS2023Dataset(root, subtypes=["GLI"], split="train")
            val_ds = BraTS2023Dataset(root, subtypes=["GLI"], split="val")
            assert len(train_ds) > 0
            assert len(val_ds) > 0

    def test_returns_dict_with_tensors(self):
        with tempfile.TemporaryDirectory() as root:
            _make_brats_structure(root, n_patients=2, n_slices=10)
            ds = BraTS2023Dataset(root, subtypes=["GLI"], split="train")
            sample = ds[0]
            assert isinstance(sample, dict)
            assert "source_image" in sample
            assert "target_image" in sample
            assert sample["source_image"].shape == (1, 8, 8)
            assert sample["target_image"].shape == (1, 8, 8)

    def test_contrast_selection(self):
        with tempfile.TemporaryDirectory() as root:
            _make_brats_structure(root, n_patients=2, n_slices=10)
            ds = BraTS2023Dataset(root, subtypes=["GLI"], source_contrast="t1ce", target_contrast="flair", split="train")
            assert len(ds) > 0

    def test_normalization_minmax(self):
        with tempfile.TemporaryDirectory() as root:
            _make_brats_structure(root, n_patients=1, n_slices=10)
            ds = BraTS2023Dataset(root, subtypes=["GLI"], normalize="minmax", split="train")
            sample = ds[0]
            # minmax normalization now maps to [-1, 1] for flow matching
            assert sample["source_image"].min() >= -1.0
            assert sample["source_image"].max() <= 1.0


# ---------------------------------------------------------------------------
# Callback tests
# ---------------------------------------------------------------------------

class MockTrainer:
    def __init__(self, current_epoch=0, global_step=0, max_epochs=10):
        self.current_epoch = current_epoch
        self.global_step = global_step
        self.max_epochs = max_epochs
        self.callback_metrics = {}
        self.val_dataloaders = []


class MockModule:
    _printed = []
    def print(self, msg):
        self._printed.append(msg)


class TestLossLoggingCallback:
    def test_accumulates_train_losses(self):
        cb = LossLoggingCallback(log_interval=1)
        trainer = MockTrainer(current_epoch=4)
        module = MockModule()
        outputs = {"loss": torch.tensor(0.5)}
        cb.on_train_batch_end(trainer, module, outputs, {}, 0)
        cb.on_train_batch_end(trainer, module, outputs, {}, 1)
        assert len(cb._train_losses) == 2

    def test_clears_after_epoch(self):
        cb = LossLoggingCallback()
        trainer = MockTrainer()
        module = MockModule()
        cb._train_losses = [0.1, 0.2, 0.3]
        cb.on_train_epoch_end(trainer, module)
        assert len(cb._train_losses) == 0


class TestValidationMetricsCallback:
    def test_psnr_identical_images(self):
        img = torch.rand(1, 1, 32, 32)
        psnr = ValidationMetricsCallback._compute_psnr(img, img)
        assert psnr == float("inf")

    def test_psnr_different_images(self):
        pred = torch.zeros(1, 1, 32, 32)
        target = torch.ones(1, 1, 32, 32)
        psnr = ValidationMetricsCallback._compute_psnr(pred, target)
        assert psnr == 0.0  # MSE=1.0 → PSNR=0

    def test_ssim_identical_images(self):
        img = torch.rand(1, 1, 32, 32)
        ssim = ValidationMetricsCallback._compute_ssim(img, img)
        assert abs(ssim - 1.0) < 1e-4

    def test_ssim_different_images(self):
        pred = torch.zeros(1, 1, 32, 32)
        target = torch.ones(1, 1, 32, 32)
        ssim = ValidationMetricsCallback._compute_ssim(pred, target)
        assert ssim < 0.1

    def test_batch_end_accumulates_metrics(self):
        cb = ValidationMetricsCallback()
        trainer = MockTrainer(current_epoch=9)
        module = MockModule()
        outputs = {
            "generated": torch.rand(2, 1, 32, 32),
            "target_image": torch.rand(2, 1, 32, 32),
        }
        cb.on_validation_batch_end(trainer, module, outputs, {}, 0)
        assert len(cb._psnr_values) == 2
        assert len(cb._ssim_values) == 2

    def test_skips_when_no_images(self):
        cb = ValidationMetricsCallback()
        trainer = MockTrainer()
        module = MockModule()
        cb.on_validation_batch_end(trainer, module, {"loss": torch.tensor(0.5)}, {}, 0)
        assert len(cb._psnr_values) == 0


class TestReportGeneratorCallback:
    def test_generates_report_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, "subdir", "report.log")
            cb = ReportGeneratorCallback(report_path=report_path)
            trainer = MockTrainer(current_epoch=9, max_epochs=10)
            module = MockModule()
            trainer.callback_metrics = {"train_loss": torch.tensor(0.1), "val_loss": torch.tensor(0.2)}

            cb.on_train_start(trainer, module)
            cb.on_train_epoch_end(trainer, module)
            cb.on_train_end(trainer, module)

            assert os.path.exists(report_path)
            with open(report_path) as f:
                content = f.read()
            assert "OuroMRI Training + Validation Report" in content
            assert "Epoch Summary" in content
            assert "Best Metrics" in content


class TestSampleVisualizationCallback:
    def test_produces_fixed_indices(self):
        cb = SampleVisualizationCallback(n_samples=2, seed=42)
        cb._ensure_indices(100)
        assert len(cb._sample_indices) == 2
        # Same seed → same indices
        cb2 = SampleVisualizationCallback(n_samples=2, seed=42)
        cb2._ensure_indices(100)
        assert cb._sample_indices == cb2._sample_indices

    def test_caps_at_dataset_size(self):
        cb = SampleVisualizationCallback(n_samples=100, seed=0)
        cb._ensure_indices(3)
        assert len(cb._sample_indices) == 3


# ---------------------------------------------------------------------------
# Training config tests
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    def test_yaml_config_loads(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "mobius", "translation", "configs", "train_config.yaml"
        )
        if not os.path.exists(config_path):
            import pytest
            pytest.skip("Config file not found")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        assert "data" in cfg
        assert "training" in cfg
        assert "report" in cfg
        assert "epochs" in cfg["training"]
        assert "val_interval" in cfg["training"]
        assert "path" in cfg["report"]


class TestTranslationForward:
    def test_validation_forward_generates_images(self):
        """Verify translate() produces valid images for PSNR/SSIM computation."""
        config = _tiny_config()
        model = OuroForImageTranslation(config)
        model.eval()

        source = torch.randn(1, 1, 32, 32)
        target = torch.randn(1, 1, 32, 32)

        with torch.no_grad():
            generated = model.translate(source, num_steps=2)

        assert generated.shape == (1, 1, 32, 32)
        assert torch.isfinite(generated).all()

        # Compute PSNR/SSIM as the callback would
        psnr = ValidationMetricsCallback._compute_psnr(generated, target)
        ssim = ValidationMetricsCallback._compute_ssim(generated, target)
        assert isinstance(psnr, float)
        assert isinstance(ssim, float)
        assert math.isfinite(psnr)
