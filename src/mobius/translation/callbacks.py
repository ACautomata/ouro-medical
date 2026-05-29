"""Single-function monitoring callbacks for OuroMRI training.

Each callback has one responsibility:
- LossLoggingCallback: log train/val loss and epoch averages
- ValidationMetricsCallback: compute and log PSNR/SSIM on validation samples
- SampleVisualizationCallback: print sample info during validation
- ReportGeneratorCallback: write a final training+validation report log
"""

import math
import os
from datetime import datetime

import lightning as pl
import torch
import torch.nn.functional as F


class LossLoggingCallback(pl.Callback):
    """Logs per-epoch training and validation loss averages."""

    def __init__(self, log_interval: int = 10):
        super().__init__()
        self.log_interval = log_interval
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if isinstance(outputs, dict) and "loss" in outputs:
            self._train_losses.append(outputs["loss"].item())
            if batch_idx % self.log_interval == 0:
                pl_module.print(
                    f"  [Step {batch_idx}] loss: {outputs['loss'].item():.6f}"
                )

    def on_train_epoch_end(self, trainer, pl_module):
        if self._train_losses:
            avg = sum(self._train_losses) / len(self._train_losses)
            pl_module.print(
                f"Epoch {trainer.current_epoch + 1} train avg_loss: {avg:.6f} "
                f"({len(self._train_losses)} batches)"
            )
        self._train_losses.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if isinstance(outputs, dict) and "loss" in outputs:
            self._val_losses.append(outputs["loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._val_losses:
            avg = sum(self._val_losses) / len(self._val_losses)
            pl_module.print(
                f"Epoch {trainer.current_epoch + 1} val avg_loss: {avg:.6f} "
                f"({len(self._val_losses)} batches)"
            )
        self._val_losses.clear()


class ValidationMetricsCallback(pl.Callback):
    """Computes PSNR and SSIM on generated images during validation.

    Expects the forward function to return 'generated', 'target_image'
    during validation, where both are [B, C, H, W] tensors.
    """

    def __init__(self):
        super().__init__()
        self._psnr_values: list[float] = []
        self._ssim_values: list[float] = []

    @staticmethod
    def _compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
        """PSNR between two [B, C, H, W] images. Assumes values in [0, 1]."""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float("inf")
        return 10.0 * math.log10(1.0 / mse.item())

    @staticmethod
    def _compute_ssim(
        pred: torch.Tensor, target: torch.Tensor,
        window_size: int = 7, pad: int = 3,
    ) -> float:
        """Mean SSIM between two [B, C, H, W] images. Single-channel only."""
        C = pred.shape[1]
        if C != 1:
            return 0.0

        # 1D Gaussian kernel → 2D outer product
        sigma = 1.5
        coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - pad
        g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g1d = g1d / g1d.sum()
        window = g1d.unsqueeze(1) @ g1d.unsqueeze(0)  # [k, k]
        window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_pred = F.conv2d(pred, window, padding=pad)
        mu_target = F.conv2d(target, window, padding=pad)
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=pad) - mu_target_sq
        sigma_cross = F.conv2d(pred * target, window, padding=pad) - mu_cross

        ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / (
            (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
        )
        return ssim_map.mean().item()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not isinstance(outputs, dict):
            return
        generated = outputs.get("generated")
        target = outputs.get("target_image")
        if generated is None or target is None:
            return

        B = generated.shape[0]
        for i in range(B):
            self._psnr_values.append(self._compute_psnr(generated[i : i + 1], target[i : i + 1]))
            self._ssim_values.append(self._compute_ssim(generated[i : i + 1], target[i : i + 1]))

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._psnr_values:
            avg_psnr = sum(self._psnr_values) / len(self._psnr_values)
            avg_ssim = sum(self._ssim_values) / len(self._ssim_values)
            pl_module.print(
                f"Epoch {trainer.current_epoch + 1} val metrics: "
                f"PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f} "
                f"({len(self._psnr_values)} samples)"
            )
        self._psnr_values.clear()
        self._ssim_values.clear()


class SampleVisualizationCallback(pl.Callback):
    """Prints a few validation samples each validation run.

    Uses fixed sample indices (seeded) so the same patients/slices are
    printed every validation epoch for consistent comparison.
    """

    def __init__(self, n_samples: int = 4, seed: int = 42):
        super().__init__()
        self.n_samples = n_samples
        self.seed = seed
        self._sample_indices: list[int] | None = None

    def _ensure_indices(self, n_available: int):
        """Compute fixed sample indices once based on dataset size."""
        if self._sample_indices is not None:
            return
        torch.manual_seed(self.seed)
        n = min(self.n_samples, n_available)
        self._sample_indices = torch.randperm(n_available)[:n].sort().values.tolist()

    def on_validation_epoch_end(self, trainer, pl_module):
        val_dl = trainer.val_dataloaders
        if not val_dl:
            return
        if isinstance(val_dl, (list, tuple)):
            val_dl = val_dl[0]

        dataset = val_dl.dataset
        self._ensure_indices(len(dataset))

        pl_module.print(f"\n  --- Validation samples (epoch {trainer.current_epoch + 1}) ---")
        for i, idx in enumerate(self._sample_indices):
            sample = dataset[idx]
            src = sample["source_image"]
            tgt = sample["target_image"]
            pl_module.print(
                f"  Sample {i}: idx={idx}, "
                f"src range=[{src.min():.3f}, {src.max():.3f}], "
                f"tgt range=[{tgt.min():.3f}, {tgt.max():.3f}], "
                f"shape={src.shape}"
            )
        pl_module.print("  ---\n")


class ReportGeneratorCallback(pl.Callback):
    """Generates a training + validation report log file.

    Accumulates epoch-level metrics and writes a summary report
    at the end of training to the configured path.
    """

    def __init__(self, report_path: str):
        super().__init__()
        self.report_path = report_path
        self._records: list[dict] = []
        self._start_time: datetime | None = None

    def on_train_start(self, trainer, pl_module):
        self._start_time = datetime.now()
        os.makedirs(os.path.dirname(self.report_path) or ".", exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        record = {
            "epoch": trainer.current_epoch + 1,
            "global_step": trainer.global_step,
        }
        metrics = trainer.callback_metrics
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                record[key] = value.item()
            else:
                record[key] = value
        self._records.append(record)

    def on_train_end(self, trainer, pl_module):
        end_time = datetime.now()
        duration = end_time - self._start_time if self._start_time else None

        lines = []
        lines.append("=" * 60)
        lines.append("OuroMRI Training + Validation Report")
        lines.append("=" * 60)
        lines.append(f"Generated: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if duration:
            lines.append(f"Duration: {duration}")
        lines.append(f"Total epochs: {trainer.current_epoch + 1}")
        lines.append(f"Total steps: {trainer.global_step}")
        lines.append(f"Max epochs: {trainer.max_epochs}")
        lines.append("")

        lines.append("-" * 60)
        lines.append("Epoch Summary")
        lines.append("-" * 60)
        for rec in self._records:
            epoch = rec.get("epoch", "?")
            parts = [f"Epoch {epoch:>3d}"]
            for metric_key in ["train_loss", "val_loss", "psnr", "ssim"]:
                if metric_key in rec:
                    parts.append(f"{metric_key}={rec[metric_key]:.6f}")
            lines.append("  ".join(parts))

        lines.append("")
        lines.append("-" * 60)
        lines.append("Best Metrics")
        lines.append("-" * 60)
        for metric_key in ["train_loss", "val_loss"]:
            vals = [r[metric_key] for r in self._records if metric_key in r]
            if vals:
                best = min(vals)
                best_epoch = self._records[vals.index(best)]["epoch"]
                lines.append(f"  Best {metric_key}: {best:.6f} (epoch {best_epoch})")
        for metric_key in ["psnr", "ssim"]:
            vals = [r[metric_key] for r in self._records if metric_key in r]
            if vals:
                best = max(vals)
                best_epoch = self._records[vals.index(best)]["epoch"]
                lines.append(f"  Best {metric_key}: {best:.6f} (epoch {best_epoch})")

        lines.append("")
        lines.append("=" * 60)

        with open(self.report_path, "w") as f:
            f.write("\n".join(lines))

        pl_module.print(f"Report saved to: {self.report_path}")


__all__ = [
    "LossLoggingCallback",
    "ValidationMetricsCallback",
    "SampleVisualizationCallback",
    "ReportGeneratorCallback",
]
