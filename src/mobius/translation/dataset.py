"""BraTS2023 dataset loader for MRI contrast translation.

Handles raw NIfTI (.nii.gz) files from the official BraTS2023 challenge data.
Supports all subtypes: GLI, MEN, MET, PED, SSA.
"""

import os
import random
from functools import lru_cache
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
        for entry in os.listdir(data_root):
            if st in entry and "TrainingData" in entry and not entry.endswith(".zip"):
                full = os.path.join(data_root, entry)
                if os.path.isdir(full):
                    dirs.append(full)
    return sorted(dirs)


def _map_contrast_files(patient_dir: str) -> dict[str, str]:
    """Map contrast names to file paths for a single patient directory."""
    result = {}
    for f in os.listdir(patient_dir):
        if not f.endswith(".nii.gz"):
            continue
        for suffix, contrast in BRATS_SUFFIX_MAP.items():
            if f.endswith(f"-{suffix}.nii.gz"):
                result[contrast] = os.path.join(patient_dir, f)
                break
    return result


def _get_n_slices(path: str) -> int:
    """Get number of slices from a NIfTI header without loading data."""
    header = nib.load(path).header
    shape = header.get_data_shape()
    return shape[2] if len(shape) == 3 else shape[0]


class BraTS2023Dataset(Dataset):
    """BraTS2023 2D slice dataset for MRI contrast translation.

    Loads raw NIfTI volumes from BraTS2023, extracts axial slices,
    and pairs source/target contrasts. Supports multiple subtypes.

    Volumes are cached in a per-worker LRU cache to avoid redundant I/O.
    With ~2600 patients and 2 contrasts per volume (~36MB each in float32),
    peak RAM per worker is bounded by lru_cache_size.

    Args:
        data_root: path to BraTS2023 root (containing subtype directories)
        subtypes: list of BraTS subtypes to include (e.g. ["GLI", "MEN"])
        source_contrast: source modality (e.g., "t1", "t1ce", "t2", "flair")
        target_contrast: target modality
        slice_range: (start, end) slice indices (default: middle 60%)
        normalize: "minmax", "zscore", or None
        split: "train" or "val"
        val_ratio: fraction of patients for validation
        seed: random seed for train/val split
        lru_cache_size: max cached volumes per worker (default 64)
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
        lru_cache_size: int = 64,
    ):
        super().__init__()
        self.source_contrast = source_contrast.lower()
        self.target_contrast = target_contrast.lower()
        self.slice_range = slice_range
        self.normalize = normalize

        self._load_volume = lru_cache(maxsize=lru_cache_size)(self._load_volume_uncached)

        subtype_dirs = _discover_brats_dirs(data_root, subtypes or AVAILABLE_SUBTYPES)
        if not subtype_dirs:
            raise FileNotFoundError(
                f"No BraTS2023 training directories found in {data_root}"
            )

        # Build (src_path, tgt_path, slice_idx) tuples
        self.samples: list[tuple[str, str, int]] = []
        patient_entries = []
        for sdir in subtype_dirs:
            for entry in sorted(os.listdir(sdir)):
                full = os.path.join(sdir, entry)
                if os.path.isdir(full):
                    patient_entries.append(full)

        random.seed(seed)
        random.shuffle(patient_entries)
        n_val = max(1, int(len(patient_entries) * val_ratio))
        if split == "val":
            patient_entries = patient_entries[:n_val]
        else:
            patient_entries = patient_entries[n_val:]

        for pdir in patient_entries:
            fmap = _map_contrast_files(pdir)
            src_file = fmap.get(self.source_contrast)
            tgt_file = fmap.get(self.target_contrast)
            if src_file is None or tgt_file is None:
                continue

            n_slices = _get_n_slices(src_file)
            if slice_range is not None:
                start, end = max(0, slice_range[0]), min(n_slices, slice_range[1])
            else:
                start, end = int(n_slices * 0.2), int(n_slices * 0.8)

            for s in range(start, end):
                self.samples.append((src_file, tgt_file, s))

    @staticmethod
    def _load_volume_uncached(path: str) -> np.ndarray:
        return np.asarray(nib.load(path).dataobj, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

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
        src_file, tgt_file, slice_idx = self.samples[idx]

        src_vol = self._load_volume(src_file)
        tgt_vol = self._load_volume(tgt_file)

        # BraTS volumes are (H, W, D), axial slices along D axis
        src_slice = self._normalize_slice(src_vol[:, :, slice_idx])
        tgt_slice = self._normalize_slice(tgt_vol[:, :, slice_idx])

        src_tensor = torch.from_numpy(src_slice).unsqueeze(0)
        tgt_tensor = torch.from_numpy(tgt_slice).unsqueeze(0)

        return {
            "source_image": src_tensor,
            "target_image": tgt_tensor,
            "slice_idx": slice_idx,
        }


__all__ = ["BraTS2023Dataset", "BRATS_SUFFIX_MAP", "AVAILABLE_SUBTYPES"]
