"""Utility functions for NEO-Unify models."""

from pathlib import Path
from typing import Tuple, Union

import torch
from PIL import Image

DEFAULT_IMAGE_PATCH_SIZE = 14
DEFAULT_VRAM_MODE = "high"
SYSTEM_MESSAGE_FOR_GEN = "You are a helpful assistant."


# VRAM modes and their prefetch counts
VRAM_MODE_PREFETCH = {
    "low": 2,
    "medium": 1,
    "high": 0,
}


def vram_mode_to_prefetch_count(vram_mode: str) -> int:
    """Convert VRAM mode to prefetch count."""
    return VRAM_MODE_PREFETCH.get(vram_mode.lower(), 0)


def best_available_device() -> str:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def infer_input_device(model, fallback: str = "cuda") -> torch.device:
    """Infer the primary input device from model parameters."""
    try:
        first_param = next(model.parameters())
        return first_param.device
    except StopIteration:
        return torch.device(fallback)


def add_offload_args(parser):
    """Add VRAM offloading arguments to an argparse parser."""
    parser.add_argument(
        "--vram_mode",
        default=DEFAULT_VRAM_MODE,
        choices=["low", "medium", "high"],
        help="VRAM management mode: low (aggressive offload), medium, high (no offload).",
    )
    parser.add_argument(
        "--device_map",
        default=None,
        help="Device map for model loading (e.g., 'auto', 'balanced', 'sequential').",
    )
    parser.add_argument(
        "--max_memory",
        default=None,
        help="Max memory per device in JSON format (e.g., '{\"0\": \"20GiB\", \"cpu\": \"30GiB\"}').",
    )


def make_offload_ctx(model, prefetch_count: int, device: str):
    """Create a context manager for VRAM offloading."""
    return OffloadContext(model, prefetch_count, device)


class OffloadContext:
    """Context manager for model offloading during generation."""

    def __init__(self, model, prefetch_count: int, device: str):
        self.model = model
        self.prefetch_count = prefetch_count
        self.device = device

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def chat(self, *args, **kwargs):
        """Forward to model's chat/generate method."""
        return self.model.chat(*args, **kwargs)


def load_model_and_tokenizer(
    model_path: str,
    dtype=None,
    device="cuda",
    gguf_checkpoint=None,
    for_offload=False,
    device_map=None,
    max_memory=None,
):
    """Load model and tokenizer from path.

    This is a placeholder. In production, this should use the transformers
    library to load the actual model.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if dtype is None:
        dtype = torch.bfloat16

    if device_map:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)

    return model, tokenizer


class InferenceProfiler:
    """Simple profiler for inference timing and memory stats."""

    def __init__(self, enabled: bool = False, device: str = "cuda", config: dict = None):
        self.enabled = enabled
        self.device = device
        self.config = config or {}
        self.load_time = 0.0
        self.generate_times = []
        self.peak_memory = 0

    def time_load(self):
        return _LoadTimer(self)

    def time_generate(self, width=1, height=1, batch=1):
        return _GenerateTimer(self, width, height, batch)

    def report(self):
        if not self.enabled:
            return
        avg_time = sum(self.generate_times) / len(self.generate_times) if self.generate_times else 0
        print(f"Average generation time: {avg_time:.3f}s")


class _LoadTimer:
    def __init__(self, profiler):
        self.profiler = profiler
        self.start = None

    def __enter__(self):
        import time
        self.start = time.time()
        return self

    def __exit__(self, *args):
        import time
        self.profiler.load_time = time.time() - self.start


class _GenerateTimer:
    def __init__(self, profiler, width, height, batch):
        self.profiler = profiler
        self.start = None

    def __enter__(self):
        import time
        self.start = time.time()
        return self

    def __exit__(self, *args):
        import time
        elapsed = time.time() - self.start
        self.profiler.generate_times.append(elapsed)


def load_image_native(
    image: Union[str, Path, Image.Image],
    patch_size: int = DEFAULT_IMAGE_PATCH_SIZE,
    downsample_ratio: float = 0.5,
    min_pixels: int = 512 * 512,
    max_pixels: int = 2048 * 2048,
    upscale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and preprocess an image for vision model input.

    Args:
        image: Image path or PIL Image.
        patch_size: Patch size for the vision model.
        downsample_ratio: Downsampling ratio.
        min_pixels: Minimum number of pixels.
        max_pixels: Maximum number of pixels.
        upscale: Whether to upscale small images.

    Returns:
        pixel_values: Preprocessed image tensor.
        grid_hw: Grid size tensor (H, W).
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    orig_width, orig_height = image.size

    # Calculate pixels
    pixels = orig_width * orig_height
    if pixels < min_pixels and upscale:
        scale = (min_pixels / pixels) ** 0.5
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    elif pixels > max_pixels:
        scale = (max_pixels / pixels) ** 0.5
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        image = image.resize((new_width, new_height), Image.LANCZOS)

    # Convert to tensor
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pixel_values = transform(image)
    pixel_values = pixel_values.unsqueeze(0)  # (1, 3, H, W)

    # Calculate grid size
    grid_h = pixel_values.shape[2] // patch_size
    grid_w = pixel_values.shape[3] // patch_size
    grid_hw = torch.tensor([[grid_h, grid_w]], dtype=torch.long)

    return pixel_values, grid_hw


__all__ = [
    "load_image_native",
    "smart_resize",
    "SYSTEM_MESSAGE_FOR_GEN",
    "DEFAULT_IMAGE_PATCH_SIZE",
    "DEFAULT_VRAM_MODE",
    "vram_mode_to_prefetch_count",
    "best_available_device",
    "infer_input_device",
    "add_offload_args",
    "make_offload_ctx",
    "load_model_and_tokenizer",
    "load_and_merge_lora_weight_from_safetensors",
    "save_compare",
    "seed_all_accelerators",
    "InferenceProfiler",
    "OffloadContext",
]

def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 512 * 512,
    max_pixels: int = 2048 * 2048,
):
    """Resize dimensions to be divisible by factor while respecting pixel bounds."""
    if height * width > max_pixels:
        scale = (max_pixels / (height * width)) ** 0.5
        height, width = int(height * scale), int(width * scale)
    if height * width < min_pixels:
        scale = (min_pixels / (height * width)) ** 0.5
        height, width = int(height * scale), int(width * scale)
    h = max(round(height / factor) * factor, factor)
    w = max(round(width / factor) * factor, factor)
    return h, w


def load_and_merge_lora_weight_from_safetensors(model, lora_path: str, device: str = "cpu"):
    """Load LoRA weights from safetensors and merge into model."""
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("safetensors is required for LoRA loading: pip install safetensors")
    lora_state = load_file(lora_path, device=device)
    model_state = model.state_dict()
    for key, value in lora_state.items():
        if key in model_state:
            model_state[key] += value.to(model_state[key].device, model_state[key].dtype)
    model.load_state_dict(model_state)
    return model


def save_compare(
    images,
    output_path: str,
    nrow: int = 8,
    padding: int = 2,
):
    """Save a comparison grid of images."""
    import torchvision.utils as vutils
    from PIL import Image as PILImage
    grid = vutils.make_grid(images, nrow=nrow, padding=padding, normalize=True, value_range=(-1, 1))
    grid = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    PILImage.fromarray(grid).save(output_path)


def seed_all_accelerators(seed: int = 0):
    """Set random seeds across all accelerators for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
