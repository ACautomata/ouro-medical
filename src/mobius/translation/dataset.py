import os
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# BraTS2023 contrast naming convention (subject to actual file layout)
CONTRAST_NAMES = ["t1", "t1ce", "t2", "flair"]


class BraTS2023Dataset(Dataset):
    """
    BraTS2023 2D slice dataset for one-to-one MRI contrast translation.

    Extracts axial slices from 3D volumes and pairs source→target contrasts.
    Supports all 6 directed contrast pairs: T1→T2, T1→T1ce, T1→FLAIR, etc.

    Args:
        data_dir: path to BraTS2023 preprocessed .npy files
            Expected layout: data_dir/{patient_id}/{contrast}.npy
            Each .npy file shape: (155, 240, 240) or (H_slices, H, W)
        source_contrast: source modality name (e.g., "t1")
        target_contrast: target modality name (e.g., "t2")
        slice_range: (start, end) slice indices to use (default: middle 60%)
        normalize: normalization method — "minmax", "zscore", or None
        split: "train" or "val"
        val_ratio: fraction of patients for validation
        seed: random seed for train/val split
    """

    def __init__(
        self,
        data_dir: str,
        source_contrast: str = "t1",
        target_contrast: str = "t2",
        slice_range: Optional[tuple[int, int]] = None,
        normalize: Optional[str] = "minmax",
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.source_contrast = source_contrast.lower()
        self.target_contrast = target_contrast.lower()
        self.normalize = normalize
        self.split = split

        # Discover patients
        patient_ids = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        # Train/val split by patient
        random.seed(seed)
        random.shuffle(patient_ids)
        n_val = max(1, int(len(patient_ids) * val_ratio))
        if split == "val":
            patient_ids = patient_ids[:n_val]
        else:
            patient_ids = patient_ids[n_val:]

        # Build slice index: (patient_id, slice_idx) pairs
        self.samples: list[tuple[str, int]] = []
        for pid in patient_ids:
            src_path = os.path.join(data_dir, pid, f"{self.source_contrast}.npy")
            tgt_path = os.path.join(data_dir, pid, f"{self.target_contrast}.npy")

            if not os.path.exists(src_path) or not os.path.exists(tgt_path):
                continue

            volume = np.load(src_path)
            n_slices = volume.shape[0]

            if slice_range is not None:
                start, end = slice_range
                start = max(0, start)
                end = min(n_slices, end)
            else:
                # Default: middle 60% of slices (skipping top/bottom 20%)
                start = int(n_slices * 0.2)
                end = int(n_slices * 0.8)

            for s in range(start, end):
                self.samples.append((pid, s))

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize_slice(self, slice_data: np.ndarray) -> np.ndarray:
        """Normalize a single 2D slice."""
        if self.normalize == "minmax":
            vmin, vmax = slice_data.min(), slice_data.max()
            if vmax > vmin:
                return (slice_data - vmin) / (vmax - vmin)
            return np.zeros_like(slice_data)
        elif self.normalize == "zscore":
            mean, std = slice_data.mean(), slice_data.std()
            if std > 0:
                return (slice_data - mean) / std
            return slice_data - mean
        return slice_data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pid, slice_idx = self.samples[idx]

        src_path = os.path.join(self.data_dir, pid, f"{self.source_contrast}.npy")
        tgt_path = os.path.join(self.data_dir, pid, f"{self.target_contrast}.npy")

        src_volume = np.load(src_path)
        tgt_volume = np.load(tgt_path)

        src_slice = self._normalize_slice(src_volume[slice_idx].astype(np.float32))
        tgt_slice = self._normalize_slice(tgt_volume[slice_idx].astype(np.float32))

        # Add channel dimension: (H, W) → (1, H, W)
        src_tensor = torch.from_numpy(src_slice).unsqueeze(0)
        tgt_tensor = torch.from_numpy(tgt_slice).unsqueeze(0)

        return {
            "source_image": src_tensor,
            "target_image": tgt_tensor,
            "patient_id": pid,
            "slice_idx": slice_idx,
            "source_contrast": self.source_contrast,
            "target_contrast": self.target_contrast,
        }
