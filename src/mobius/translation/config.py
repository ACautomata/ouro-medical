# coding=utf-8
# Copyright 2025 ByteDance Seed. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OuroMRI translation model configuration."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class OuroMRIConfig(PretrainedConfig):
    r"""
    Configuration class for OuroMRI image-to-image translation model.

    This configuration defines the architecture of a Vision-to-Vision translation model
    that uses Ouro (Looped Language Model) as the backbone for processing MRI images.

    Args:
        image_size (`int`, *optional*, defaults to 256):
            Input/output image size (assumes square images).
        patch_size (`int`, *optional*, defaults to 16):
            Size of each image patch for embedding.
        num_channels (`int`, *optional*, defaults to 1):
            Number of input channels (1 for grayscale MRI).
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of hidden representations in the vision embedding and backbone.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimension of MLP representations in the transformer.
        num_hidden_layers (`int`, *optional*, defaults to 16):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of key/value heads for Grouped Query Attention.
        ve_hidden_size (`int`, *optional*, defaults to 1024):
            Hidden size of the VisionEmbeddings output before projection to Ouro dimension.
        downsample_ratio (`float`, *optional*, defaults to 1.0):
            Downsampling ratio in VisionEmbeddings. Values < 1.0 enable downsampling.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        max_position_embeddings (`int`, *optional*, defaults to 10000):
            Maximum position embeddings for RoPE computation.
        total_ut_steps (`int`, *optional*, defaults to 4):
            Number of Universal Transformer recurrent steps (loop depth) from Ouro.
        early_exit_threshold (`float`, *optional*, defaults to 1.0):
            Cumulative probability threshold for adaptive early exit during inference.
            A value of 1.0 means always use all steps (no early exit).
        fm_steps (`int`, *optional*, defaults to 50):
            Number of flow matching inference steps.
        fm_cfg_guidance_scale (`float`, *optional*, defaults to 1.0):
            Classifier-free guidance scale for flow matching. 1.0 disables CFG.
        fm_strategy (`str`, *optional*, defaults to `"standard"`):
            Flow matching prediction strategy. One of:
            - `"standard"`: x_0-prediction with MSE loss, multi-step Euler sampling.
            - `"meanflow"`: MeanFlow average velocity with v-loss (iMF-style),
              one-step sampling via the MeanFlow Identity.
        fm_loss_type (`str`, *optional*, defaults to `"v_loss"`):
            MeanFlow 损失函数类型，仅在 fm_strategy="meanflow" 时生效。
            - `"v_loss"`: iMF 风格的 v-loss，对 du/dt 做 stop_gradient，
              等价于回归瞬时速度 v = ε - x₀。
            - `"u_loss"`: 原始 MF 论文的 u-loss，梯度流过 du/dt（二阶梯度），
              目标为 u_tgt = v_gt - (t-r) * du/dt。
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by RMS normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for attention probabilities.
        mlp_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for MLP layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer.

    ```python
    >>> from mobius.translation import OuroMRIConfig

    >>> # Initializing a OuroMRI configuration
    >>> config = OuroMRIConfig()

    >>> # Accessing model configuration
    >>> hidden_size = config.hidden_size
    ```
    """

    model_type = "ouro_mri"
    keys_to_ignore_at_inference = ["latent"]

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        num_channels: int = 1,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        ve_hidden_size: int = 1024,
        downsample_ratio: float = 1.0,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 10000,
        total_ut_steps: int = 4,
        early_exit_threshold: float = 1.0,
        fm_steps: int = 50,
        fm_cfg_guidance_scale: float = 1.0,
        fm_strategy: str = "standard",
        fm_loss_type: str = "v_loss",
        t_eps: float = 1e-6,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        initializer_range: float = 0.02,
        hidden_act: str = "silu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if fm_strategy not in ("standard", "meanflow"):
            raise ValueError(
                f"fm_strategy must be 'standard' or 'meanflow', got '{fm_strategy}'"
            )

        if fm_loss_type not in ("v_loss", "u_loss"):
            raise ValueError(
                f"fm_loss_type must be 'v_loss' or 'u_loss', got '{fm_loss_type}'"
            )
        if fm_strategy != "meanflow" and fm_loss_type != "v_loss":
            logger.warning(
                f"fm_loss_type='{fm_loss_type}' only applies when fm_strategy='meanflow'; "
                f"current fm_strategy='{fm_strategy}' will ignore it."
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.ve_hidden_size = ve_hidden_size
        self.downsample_ratio = downsample_ratio
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.total_ut_steps = total_ut_steps
        self.t_eps = t_eps
        self.early_exit_threshold = early_exit_threshold
        self.fm_steps = fm_steps
        self.fm_cfg_guidance_scale = fm_cfg_guidance_scale
        self.fm_strategy = fm_strategy
        self.fm_loss_type = fm_loss_type
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.mlp_dropout = mlp_dropout
        self.initializer_range = initializer_range
        self.hidden_act = hidden_act


__all__ = ["OuroMRIConfig"]