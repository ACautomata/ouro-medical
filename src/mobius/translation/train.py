"""Training script for OuroMRI flow matching based image translation.

Uses the stable-pretraining (spt) framework built on PyTorch Lightning.
Configure via YAML config file — see configs/train_config.yaml for reference.
"""

import argparse
import math

import lightning as pl
import torch
import torch.nn.functional as F
import yaml

import stable_pretraining as spt

from .callbacks import (
    LossLoggingCallback,
    ReportGeneratorCallback,
    SampleVisualizationCallback,
    ValidationMetricsCallback,
)
from .config import OuroMRIConfig
from .dataset import BraTS2023Dataset
from .modeling_translation import OuroForImageTranslation


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(model_cfg: dict) -> OuroForImageTranslation:
    import inspect
    valid_keys = set(inspect.signature(OuroMRIConfig.__init__).parameters.keys()) - {"self", "kwargs"}
    filtered = {k: v for k, v in model_cfg.items() if k in valid_keys}
    config = OuroMRIConfig(**filtered)
    return OuroForImageTranslation(config)


def translation_forward(self, batch, stage):
    """Forward function for spt.Module — handles both train and validation."""
    source = batch["source_image"]
    target = batch["target_image"]

    if stage == "fit":
        B = source.shape[0]
        t = torch.rand(B, device=source.device)
        result = self.model(source, target, t)
        return {**result, "target_image": target, "source_image": source}
    else:
        # Validation: run full inference to generate images
        generated = self.model.translate(source)
        loss = F.mse_loss(generated, target)
        return {
            "loss": loss,
            "generated": generated,
            "target_image": target,
            "source_image": source,
        }


def main():
    parser = argparse.ArgumentParser(description="Train OuroMRI with stable-pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from Lightning checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    report_cfg = cfg.get("report", {})

    # Seed
    pl.seed_everything(train_cfg.get("seed", 42), workers=True)

    # Model
    model = build_model(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Datasets — created before module to compute exact training steps
    train_dataset = BraTS2023Dataset(
        data_root=data_cfg["data_root"],
        subtypes=data_cfg.get("subtypes"),
        source_contrast=data_cfg.get("source_contrast", "t1"),
        target_contrast=data_cfg.get("target_contrast", "t2"),
        split="train",
    )
    val_dataset = BraTS2023Dataset(
        data_root=data_cfg["data_root"],
        subtypes=data_cfg.get("subtypes"),
        source_contrast=data_cfg.get("source_contrast", "t1"),
        target_contrast=data_cfg.get("target_contrast", "t2"),
        split="val",
    )

    print(f"Train: {len(train_dataset)} slices | Val: {len(val_dataset)} slices")

    # Compute exact number of optimizer steps for scheduler
    grad_accum = train_cfg.get("gradient_accumulation_steps", 4)
    batch_size = train_cfg["batch_size"]
    steps_per_epoch = math.ceil(len(train_dataset) / batch_size) // grad_accum
    total_training_steps = steps_per_epoch * train_cfg["epochs"]
    warmup_steps = min(train_cfg.get("warmup_steps", 1000), total_training_steps // 10)

    def warmup_cosine_lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_training_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    # Optimizer config with warmup + cosine decay
    optim_cfg = {
        "optimizer": {
            "type": "AdamW",
            "lr": train_cfg["learning_rate"],
            "weight_decay": train_cfg.get("weight_decay", 0.01),
        },
        "scheduler": {
            "type": "LambdaLR",
            "lr_lambda": warmup_cosine_lr_lambda,
        },
        "interval": "step",
    }

    # spt.Module
    module = spt.Module(
        model=model,
        forward=translation_forward,
        optim=optim_cfg,
    )

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )

    data = spt.data.DataModule(train=train_loader, val=val_loader)

    # Callbacks
    report_path = report_cfg.get("path", "outputs/report.log")
    val_interval = train_cfg.get("val_interval", 10)
    n_val_samples = report_cfg.get("n_samples", 4)

    callbacks = [
        LossLoggingCallback(log_interval=train_cfg.get("log_interval", 10)),
        ValidationMetricsCallback(),
        SampleVisualizationCallback(n_samples=n_val_samples),
        ReportGeneratorCallback(report_path=report_path),
        pl.callbacks.ModelCheckpoint(
            dirpath=train_cfg.get("output_dir", "outputs/checkpoints"),
            filename="ouro-mri-{epoch:03d}",
            every_n_epochs=train_cfg.get("save_interval", 1),
            save_last=True,
        ),
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_cfg["epochs"],
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if train_cfg.get("bf16", True) else "32-true",
        accumulate_grad_batches=grad_accum,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        check_val_every_n_epoch=val_interval,
        callbacks=callbacks,
        log_every_n_steps=train_cfg.get("log_interval", 10),
        default_root_dir=train_cfg.get("output_dir", "outputs"),
    )

    # Manager
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data,
    )

    print(f"Steps/epoch: {steps_per_epoch} | Total steps: {total_training_steps}")
    print(f"Validation every {val_interval} epochs")
    print(f"Report will be saved to: {report_path}")
    print(f"Starting training for {train_cfg['epochs']} epochs...")

    manager(ckpt_path=args.resume)


if __name__ == "__main__":
    main()
