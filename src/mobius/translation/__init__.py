"""
OuroMRI: MRI Image Translation using Ouro Looped Language Model.

Architecture: VE (Vision Embeddings) + Ouro UT loop Backbone + Flow Matching Diffusion Head.
Supports one-to-one MRI contrast translation on BraTS2023 dataset.
"""

from .config import OuroMRIConfig
from .modeling_translation import OuroForImageTranslation
from .dataset import BraTS2023Dataset
from .callbacks import (
    LossLoggingCallback,
    ReportGeneratorCallback,
    SampleVisualizationCallback,
    ValidationMetricsCallback,
)

__all__ = [
    "OuroMRIConfig",
    "OuroForImageTranslation",
    "BraTS2023Dataset",
    "LossLoggingCallback",
    "ValidationMetricsCallback",
    "SampleVisualizationCallback",
    "ReportGeneratorCallback",
]