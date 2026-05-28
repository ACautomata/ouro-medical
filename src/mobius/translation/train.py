"""Training script for OuroMRI flow matching based image translation."""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from .config import OuroMRIConfig
from .modeling_translation import OuroForImageTranslation
from .dataset import BraTS2023Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train OuroMRI image translation model")
    parser.add_argument("--data_root", type=str, required=True, help="BraTS2023 dataset path")
    parser.add_argument("--output_dir", type=str, required=True, help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument("--save_interval", type=int, default=1, help="Checkpoint save frequency")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging frequency")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    """Cosine schedule with linear warmup."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item()))

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, epoch, args, filename=None):
    """Save model checkpoint."""
    os.makedirs(args.output_dir, exist_ok=True)
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pt"

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
    }

    filepath = os.path.join(args.output_dir, filename)
    torch.save(checkpoint, filepath)
    return filepath


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    # weights_only=False is required because checkpoints contain optimizer/scheduler state
    # (non-tensor objects). Only load checkpoints you generated yourself.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0)


def sample_random_timesteps(batch_size, device):
    """Sample random timesteps t ∈ [0, 1] for flow matching."""
    return torch.rand(batch_size, device=device)


def train_step(model, source, target, optimizer, scaler, args, global_step):
    """Single training step with flow matching."""
    batch_size = source.shape[0]

    # Sample random timesteps t ∈ [0, 1]
    t = sample_random_timesteps(batch_size, source.device)

    # Forward pass with mixed precision
    with autocast(enabled=args.device == "cuda"):
        result = model(source, target, t)
        loss = result["loss"]

    # Scale loss for gradient accumulation
    scaled_loss = loss / args.gradient_accumulation_steps

    # Backward pass
    scaler.scale(scaled_loss).backward()

    return loss.item()


def train_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, args, writer=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        source = batch["source_image"].to(args.device)
        target = batch["target_image"].to(args.device)

        # Training step
        loss = train_step(model, source, target, optimizer, scaler, args, epoch)

        # Gradient accumulation
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss
        num_batches += 1

        # Logging
        if batch_idx % args.log_interval == 0:
            avg_loss = total_loss / max(num_batches, 1)
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Step [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss:.4f} "
                f"Avg Loss: {avg_loss:.4f} "
                f"LR: {current_lr:.2e}"
            )

            if writer is not None:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar("train/loss", loss, global_step)
                writer.add_scalar("train/loss_avg", avg_loss, global_step)
                writer.add_scalar("train/learning_rate", current_lr, global_step)

    return total_loss / max(num_batches, 1)


def validate(model, val_loader, args):
    """Validation step."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            source = batch["source_image"].to(args.device)
            target = batch["target_image"].to(args.device)
            batch_size = source.shape[0]

            t = sample_random_timesteps(batch_size, source.device)

            with autocast(enabled=args.device == "cuda"):
                result = model(source, target, t)
                loss = result["loss"]

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    print(f"Using device: {device}")

    # Initialize model
    config = OuroMRIConfig()
    model = OuroForImageTranslation(config)
    model.to(device)

    # Initialize optimizer (AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Calculate training steps
    train_dataset = BraTS2023Dataset(args.data_root, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    num_training_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps

    # Scheduler (cosine with warmup)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, num_training_steps
    )

    # Mixed precision scaler
    scaler = GradScaler(enabled=args.device == "cuda")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # Resume from checkpoint
    start_epoch = 0
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train one epoch
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, args, writer)

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = save_checkpoint(model, optimizer, scheduler, epoch + 1, args)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Log epoch metrics
        print(f"Epoch {epoch + 1} - Average Loss: {train_loss:.4f}")
        writer.add_scalar("train/epoch_loss", train_loss, epoch)
        writer.add_scalar("train/epoch", epoch + 1, epoch)

    # Final checkpoint
    final_checkpoint = save_checkpoint(model, optimizer, scheduler, args.epochs, args, filename="final_model.pt")
    print(f"Training complete. Final model saved to: {final_checkpoint}")

    writer.close()


if __name__ == "__main__":
    main()