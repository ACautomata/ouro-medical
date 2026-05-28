# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

"""Ouro Looped Language Model - extracted from ByteDance/Ouro-1.4B HuggingFace model files.

Original source: https://huggingface.co/ByteDance/Ouro-1.4B
Paper: "Scaling Latent Reasoning via Looped Language Models" (arXiv:2510.25741)
"""

from .configuration_ouro import OuroConfig
from .modeling_ouro import (
    OuroForCausalLM,
    OuroModel,
    OuroPreTrainedModel,
    OuroForSequenceClassification,
    OuroForTokenClassification,
    OuroForQuestionAnswering,
    UniversalTransformerCache,
)

__all__ = [
    "OuroConfig",
    "OuroPreTrainedModel",
    "OuroModel",
    "OuroForCausalLM",
    "OuroForSequenceClassification",
    "OuroForTokenClassification",
    "OuroForQuestionAnswering",
    "UniversalTransformerCache",
]
