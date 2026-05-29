"""Training script for OuroMRI flow matching based image translation.

Supports single-GPU and multi-GPU DDP training via `torchrun`.
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from .config import OuroMRIConfig
from .modeling_translation import OuroForImageTranslation
from .dataset import BraTS2023Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train OuroMRI image translation model")
    parser.add_argument("--data_root", type=str, required=True, help="BraTS2023 dataset root path")
    parser.add_argument(
        "--subtypes", type=str, nargs="+",
        default=["GLI", "MEN", "MET", "PED", "SSA"],
        help="BraTS subtypes to include",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument("--save_interval", type=int, default=1, help="Checkpoint save frequency")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging frequency")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


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

    state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
    }

    filepath = os.path.join(args.output_dir, filename)
    torch.save(checkpoint, filepath)
    return filepath


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    raw = model.module if isinstance(model, DDP) else model
    raw.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0)


def train_step(model, source, target, scaler, grad_accum_steps, use_bf16):
    """Single training step with flow matching."""
    batch_size = source.shape[0]
    t = torch.rand(batch_size, device=source.device)

    with autocast("cuda", enabled=use_bf16, dtype=torch.bfloat16 if use_bf16 else torch.float32):
        result = model(source, target, t)
        loss = result["loss"] / grad_accum_steps

    scaler.scale(loss).backward()
    return loss.item() * grad_accum_steps


def train_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, args, writer=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    use_bf16 = args.bf16

    if dist.is_initialized():
        train_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        source = batch["source_image"].cuda(non_blocking=True)
        target = batch["target_image"].cuda(non_blocking=True)

        loss = train_step(model, source, target, scaler, args.gradient_accumulation_steps, use_bf16)

        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss
        num_batches += 1

        if is_main() and batch_idx % args.log_interval == 0:
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


def main():
    args = parse_args()
    set_seed(args.seed)

    # DDP setup
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main():
        print(f"Rank {rank}/{world_size}, device: {device}")

    # Model
    config = OuroMRIConfig()
    model = OuroForImageTranslation(config).to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if isinstance(model, DDP) else model
    n_params = sum(p.numel() for p in raw_model.parameters())
    if is_main():
        print(f"Model params: {n_params:,} ({n_params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # Dataset & DataLoader
    train_dataset = BraTS2023Dataset(
        args.data_root, subtypes=args.subtypes, split="train",
    )
    sampler = DistributedSampler(train_dataset, shuffle=True) if dist.is_initialized() else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Scheduler
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    num_training_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, num_training_steps
    )

    # Scaler (BF16 doesn't need GradScaler)
    scaler = GradScaler("cuda", enabled=not args.bf16)

    # TensorBoard (rank 0 only)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs")) if is_main() else None

    # Resume
    start_epoch = 0
    if args.resume is not None:
        if is_main():
            print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)
        if is_main():
            print(f"Resumed from epoch {start_epoch}")

    if is_main():
        print(f"Dataset: {len(train_dataset)} slices, {steps_per_epoch} steps/epoch")
        print(f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"Starting training for {args.epochs} epochs...")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if is_main():
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, args, writer)

        if is_main():
            print(f"Epoch {epoch + 1} - Average Loss: {train_loss:.4f}")
            writer.add_scalar("train/epoch_loss", train_loss, epoch)

        if is_main() and (epoch + 1) % args.save_interval == 0:
            checkpoint_path = save_checkpoint(model, optimizer, scheduler, epoch + 1, args)
            print(f"Saved checkpoint: {checkpoint_path}")

    if is_main():
        save_checkpoint(model, optimizer, scheduler, args.epochs, args, filename="final_model.pt")
        print("Training complete.")
        writer.close()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
