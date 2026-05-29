"""
Inference script for Euler integration based MRI translation.
Translates source MRI to target contrast using trained OuroMRI model.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .modeling_translation import OuroForImageTranslation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Euler integration based MRI translation inference"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        help="Source MRI image path (.nii.gz or .png). Required unless using --input_dir.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path for generated image. Required unless using --output_dir.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing input images for batch processing.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for batch processing.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of Euler integration steps (default: 50)",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale (default: 1.0, no CFG)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)",
    )
    parser.add_argument(
        "--target_image",
        type=str,
        help="Ground truth target image for metrics computation.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Patch size used in the model (default: 16)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size for processing (default: 256)",
    )
    return parser.parse_args()


def load_image(image_path: str, image_size: int = 256, num_channels: int = 1) -> torch.Tensor:
    """Load image from path, supporting .nii.gz and standard image formats.

    Args:
        image_path: Path to the image file.
        image_size: Target spatial size (square).
        num_channels: Number of output channels (1 for grayscale MRI, 3 for RGB).
    """
    path = Path(image_path)
    is_nifti = path.name.endswith(".nii.gz") or path.suffix == ".nii"

    if is_nifti:
        try:
            import nibabel as nib

            nii_img = nib.load(str(path))
            data = nii_img.get_fdata()

            # Handle 3D volumes - take middle slice
            if len(data.shape) == 3:
                mid_slice = data.shape[2] // 2
                data = data[:, :, mid_slice]

            # Normalize to [0, 1]
            data = data.astype(np.float32)
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)

            # Convert to tensor
            img_tensor = torch.from_numpy(data).float()

            # Resize to target size
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            img_tensor = F.interpolate(
                img_tensor, size=(image_size, image_size), mode="bilinear", align_corners=False
            )

            # Expand to requested channel count
            img_tensor = img_tensor.repeat(1, num_channels, 1, 1)

            return img_tensor

        except ImportError:
            raise ImportError("nibabel is required for .nii.gz files. Install with: pip install nibabel")

    else:
        # Standard image formats (.png, .jpg, etc.)
        mode = "RGB" if num_channels == 3 else "L"
        img = Image.open(str(path)).convert(mode)
        img = img.resize((image_size, image_size), Image.BILINEAR)
        img_array = np.array(img).astype(np.float32) / 255.0
        if num_channels == 1:
            img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        else:
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        return img_tensor


def save_image(tensor: torch.Tensor, output_path: str, normalize: bool = True):
    """Save tensor as image, supporting both .nii.gz and standard formats."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_nifti = path.name.endswith(".nii.gz") or path.suffix == ".nii"

    # Move to CPU and convert to numpy
    img = tensor.detach().cpu()

    # Handle different tensor shapes
    if img.dim() == 4:
        img = img[0]  # Remove batch dimension

    if normalize:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    img_np = img.permute(1, 2, 0).numpy()  # [H, W, C]

    if is_nifti:
        try:
            import nibabel as nib

            # Handle both 3-channel and 1-channel images
            if img_np.shape[-1] == 3:
                # Convert RGB to single channel (use average or take first channel)
                img_np = img_np[:, :, 0]

            nii_img = nib.Nifti1Image(img_np.astype(np.float32), affine=np.eye(4))
            nib.save(nii_img, str(path))

        except ImportError:
            raise ImportError("nibabel is required for .nii.gz files. Install with: pip install nibabel")

    else:
        # Standard image formats
        img_np = (img_np * 255).astype(np.uint8)
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)  # Grayscale
            Image.fromarray(img_np).save(str(path))
        else:
            Image.fromarray(img_np).save(str(path))


def unpatchify(x: torch.Tensor, H: int, W: int, patch_size: int = 16) -> torch.Tensor:
    """Convert patch tokens back to spatial format."""
    B, N, D = x.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    x = x.reshape(B, num_patches_h, num_patches_w, patch_size, patch_size, D)
    x = x.permute(0, 5, 1, 3, 2, 4)  # [B, D, num_patches_h, patch_size, num_patches_w, patch_size]
    x = x.reshape(B, D, H, W)
    return x


def translate_image(
    model,
    source_image: torch.Tensor,
    num_steps: int = 50,
    cfg_scale: float = 1.0,
    device: str = "cuda",
    patch_size: int = 16,
) -> torch.Tensor:
    """
    Translate source MRI to target contrast using Euler integration.

    Args:
        model: The translation model (OuroForImageTranslation)
        source_image: Source image tensor [B, C, H, W]
        num_steps: Number of Euler integration steps
        cfg_scale: Classifier-free guidance scale
        device: Device to run on
        patch_size: Patch size used in the model

    Returns:
        Generated target image tensor [B, C, H, W]
    """
    model.eval()
    with torch.no_grad():
        # Use the model's translate() method directly
        result = model.translate(source_image.to(device), num_steps=num_steps)
        return result


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float("inf")
    psnr = 20 * torch.log10(torch.tensor(data_range) / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> float:
    """Compute Structural Similarity Index (simplified version)."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Ensure 4D tensors [B, C, H, W]
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    # Create Gaussian window
    sigma = 1.5
    gauss = torch.tensor(
        [
            np.exp(-((x - window_size // 2) ** 2) / (2 * sigma**2))
            for x in range(window_size)
        ],
        dtype=img1.dtype,
        device=img1.device,
    )
    gauss = gauss / gauss.sum()
    window = gauss.unsqueeze(-1) * gauss.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0)

    channels = img1.shape[1]
    window = window.expand(channels, 1, window_size, window_size)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1**2, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean().item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute all metrics between prediction and target."""
    # Normalize both to [0, 1]
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    target_norm = (target - target.min()) / (target.max() - target.min() + 1e-8)

    psnr = compute_psnr(pred_norm, target_norm)
    ssim = compute_ssim(pred_norm, target_norm)

    return {"psnr": psnr, "ssim": ssim}


def process_single(
    model,
    input_path: str,
    output_path: str,
    num_steps: int,
    cfg_scale: float,
    device: str,
    patch_size: int,
    target_path: str = None,
    image_size: int = 256,
) -> dict:
    """Process a single image and return results."""
    logger.info(f"Processing: {input_path}")

    # Load source image
    source = load_image(input_path, image_size=image_size)

    # Translate
    result = translate_image(model, source, num_steps, cfg_scale, device, patch_size)

    # Save result
    save_image(result, output_path)
    logger.info(f"Saved to: {output_path}")

    # Compute metrics if target provided
    metrics = {}
    if target_path:
        target = load_image(target_path, image_size=image_size)
        metrics = compute_metrics(result, target)
        logger.info(f"Metrics - PSNR: {metrics['psnr']:.4f}, SSIM: {metrics['ssim']:.4f}")

    return {"output_path": output_path, "metrics": metrics}


def process_batch(
    model,
    input_dir: str,
    output_dir: str,
    num_steps: int,
    cfg_scale: float,
    device: str,
    patch_size: int,
    image_size: int = 256,
):
    """Process multiple images from a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all image files
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.nii.gz", "*.nii"]
    image_files = []
    for ext in extensions:
        image_files.extend(list(input_path.glob(ext)))

    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        return

    logger.info(f"Found {len(image_files)} images to process")

    results = []
    for img_file in tqdm(image_files, desc="Batch processing"):
        output_file = output_path / f"generated_{img_file.name}"
        result = process_single(
            model,
            str(img_file),
            str(output_file),
            num_steps,
            cfg_scale,
            device,
            patch_size,
            image_size=image_size,
        )
        results.append(result)

    # Summary
    psnr_values = [r["metrics"].get("psnr", 0) for r in results if r["metrics"]]
    ssim_values = [r["metrics"].get("ssim", 0) for r in results if r["metrics"]]

    if psnr_values:
        logger.info("=" * 50)
        logger.info("Batch Processing Summary:")
        logger.info(f"  PSNR: {np.mean(psnr_values):.4f} +/- {np.std(psnr_values):.4f}")
        logger.info(f"  SSIM: {np.mean(ssim_values):.4f} +/- {np.std(ssim_values):.4f}")
        logger.info("=" * 50)

    return results


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Using device: {device}")

    # Load model from checkpoint
    logger.info(f"Loading model from: {args.checkpoint_path}")
    model = OuroForImageTranslation.from_pretrained(args.checkpoint_path)
    model.to(device)
    model.eval()

    # Determine mode: single or batch
    if args.input_dir and args.output_dir:
        # Batch processing
        process_batch(
            model,
            args.input_dir,
            args.output_dir,
            args.num_steps,
            args.cfg_scale,
            device,
            args.patch_size,
            args.image_size,
        )
    elif args.input_image and args.output_path:
        # Single image processing
        process_single(
            model,
            args.input_image,
            args.output_path,
            args.num_steps,
            args.cfg_scale,
            device,
            args.patch_size,
            args.target_image,
            args.image_size,
        )
    else:
        raise ValueError(
            "Either (--input_image and --output_path) or (--input_dir and --output_dir) must be provided."
        )

    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
