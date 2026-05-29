"""BraTS2023 dataset loader for MRI contrast translation.

Handles raw NIfTI (.nii.gz) files from the official BraTS2023 challenge data.
Supports all subtypes: GLI, MEN, MET, PED, SSA.
"""

import os
import random
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

# BraTS2023 file suffix → internal contrast name
BRATS_SUFFIX_MAP = {
    "t1n": "t1",
    "t1c": "t1ce",
    "t2w": "t2",
    "t2f": "flair",
}

AVAILABLE_SUBTYPES = ["GLI", "MEN", "MET", "PED", "SSA"]


def _discover_brats_dirs(data_root: str, subtypes: list[str]) -> list[str]:
    """Find all BraTS2023 training directories for the given subtypes."""
    dirs = []
    for st in subtypes:
        # Match pattern: ASNR-MICCAI-BraTS2023-{ST}-Challenge-TrainingData
        for entry in os.listdir(data_root):
            if st in entry and "TrainingData" in entry and not entry.endswith(".zip"):
                full = os.path.join(data_root, entry)
                if os.path.isdir(full):
                    dirs.append(full)
    return sorted(dirs)


def _find_contrast_file(patient_dir: str, contrast: str) -> Optional[str]:
    """Find the NIfTI file for a given contrast in a patient directory.

    Handles BraTS naming: {prefix}-{suffix}.nii.gz where suffix maps to contrast.
    """
    target_suffix = None
    for suf, name in BRATS_SUFFIX_MAP.items():
        if name == contrast:
            target_suffix = suf
            break
    if target_suffix is None:
        return None

    for f in os.listdir(patient_dir):
        if f.endswith(f"-{target_suffix}.nii.gz"):
            return os.path.join(patient_dir, f)
    return None


class BraTS2023Dataset(Dataset):
    """BraTS2023 2D slice dataset for MRI contrast translation.

    Loads raw NIfTI volumes from BraTS2023, extracts axial slices,
    and pairs source→target contrasts. Supports multiple subtypes.

    Args:
        data_root: path to BraTS2023 root (containing subtype directories)
            e.g. "/data72/dataset/ASNR-MICCAI-BraTS2023"
        subtypes: list of BraTS subtypes to include (e.g. ["GLI", "MEN"])
        source_contrast: source modality (e.g. "t1", "t1ce", "t2", "flair")
        target_contrast: target modality
        slice_range: (start, end) slice indices (default: middle 60%)
        normalize: "minmax", "zscore", or None
        split: "train" or "val"
        val_ratio: fraction of patients for validation
        seed: random seed for train/val split
        cache_volumes: cache loaded volumes in memory (faster but uses more RAM)
    """

    def __init__(
        self,
        data_root: str,
        subtypes: Optional[list[str]] = None,
        source_contrast: str = "t1",
        target_contrast: str = "t2",
        slice_range: Optional[tuple[int, int]] = None,
        normalize: Optional[str] = "minmax",
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        cache_volumes: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.subtypes = subtypes or AVAILABLE_SUBTYPES
        self.source_contrast = source_contrast.lower()
        self.target_contrast = target_contrast.lower()
        self.slice_range = slice_range
        self.normalize = normalize
        self.cache_volumes = cache_volumes
        self._cache: dict[str, np.ndarray] = {}

        # Discover patient directories across all subtypes
        subtype_dirs = _discover_brats_dirs(data_root, self.subtypes)
        if not subtype_dirs:
            raise FileNotFoundError(
                f"No BraTS2023 training directories found in {data_root} "
                f"for subtypes {self.subtypes}"
            )

        patient_dirs = []
        for sdir in subtype_dirs:
            for entry in sorted(os.listdir(sdir)):
                full = os.path.join(sdir, entry)
                if os.path.isdir(full):
                    patient_dirs.append(full)

        # Train/val split by patient
        random.seed(seed)
        random.shuffle(patient_dirs)
        n_val = max(1, int(len(patient_dirs) * val_ratio))
        if split == "val":
            patient_dirs = patient_dirs[:n_val]
        else:
            patient_dirs = patient_dirs[n_val:]

        # Build slice index: (patient_dir, slice_idx) pairs
        self.samples: list[tuple[str, int]] = []
        for pdir in patient_dirs:
            src_file = _find_contrast_file(pdir, self.source_contrast)
            tgt_file = _find_contrast_file(pdir, self.target_contrast)
            if src_file is None or tgt_file is None:
                continue

            # Probe volume shape without loading full data
            header = nib.load(src_file).header
            shape = header.get_data_shape()
            n_slices = shape[2] if len(shape) == 3 else shape[0]

            if slice_range is not None:
                start, end = slice_range
                start = max(0, start)
                end = min(n_slices, end)
            else:
                start = int(n_slices * 0.2)
                end = int(n_slices * 0.8)

            for s in range(start, end):
                self.samples.append((pdir, s))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_volume(self, path: str) -> np.ndarray:
        if self.cache_volumes and path in self._cache:
            return self._cache[path]
        vol = np.asarray(nib.load(path).dataobj, dtype=np.float32)
        if self.cache_volumes:
            self._cache[path] = vol
        return vol

    def _normalize_slice(self, data: np.ndarray) -> np.ndarray:
        if self.normalize == "minmax":
            vmin, vmax = data.min(), data.max()
            if vmax > vmin:
                return (data - vmin) / (vmax - vmin)
            return np.zeros_like(data)
        elif self.normalize == "zscore":
            mean, std = data.mean(), data.std()
            if std > 0:
                return (data - mean) / std
            return data - mean
        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pdir, slice_idx = self.samples[idx]

        src_file = _find_contrast_file(pdir, self.source_contrast)
        tgt_file = _find_contrast_file(pdir, self.target_contrast)

        src_vol = self._load_volume(src_file)
        tgt_vol = self._load_volume(tgt_file)

        # BraTS volumes are (H, W, D), axial slices are along D axis
        src_slice = self._normalize_slice(src_vol[:, :, slice_idx])
        tgt_slice = self._normalize_slice(tgt_vol[:, :, slice_idx])

        # (H, W) → (1, H, W)
        src_tensor = torch.from_numpy(src_slice).unsqueeze(0)
        tgt_tensor = torch.from_numpy(tgt_slice).unsqueeze(0)

        pid = os.path.basename(pdir)
        return {
            "source_image": src_tensor,
            "target_image": tgt_tensor,
            "patient_id": pid,
            "slice_idx": slice_idx,
        }


__all__ = ["BraTS2023Dataset", "BRATS_SUFFIX_MAP", "AVAILABLE_SUBTYPES"]
