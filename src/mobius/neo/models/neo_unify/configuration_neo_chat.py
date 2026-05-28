import copy

from transformers import Qwen3Config, Qwen3MoeConfig
from transformers.utils import logging
from transformers.configuration_utils import PretrainedConfig

from .configuration_neo_vit import NEOVisionConfig


logger = logging.get_logger(__name__)


class NEOLLMConfig(Qwen3Config):
    """Config for the dense Qwen3 backbone used by NEO-Unify.

    Extends ``Qwen3Config`` with two extra rope knobs used by the spatial
    (height/width) rotary axes that are layered on top of the temporal one.
    """

    def __init__(self, rope_theta_hw=10000.0, max_position_embeddings_hw=10000, **kwargs):
        super().__init__(**kwargs)
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings_hw = max_position_embeddings_hw


class NEOMoELLMConfig(Qwen3MoeConfig):
    """Config for the Qwen3-MoE backbone used by NEO-Unify.

    Extends ``Qwen3MoeConfig`` with the same ``rope_theta_hw`` /
    ``max_position_embeddings_hw`` extras as :class:`NEOLLMConfig`, and adds a
    *generation-path* MoE branch alongside the standard understanding-path one.
    In the A3B unified model every decoder layer carries two parallel sparse
    MoE blocks routed by the per-token ``image_gen_indicators`` mask:

        * ``mlp``           - sparse MoE for the understanding path
                              (``num_experts`` experts, ``num_experts_per_tok`` active,
                              expert width ``moe_intermediate_size``).
        * ``mlp_mot_gen``   - sparse MoE for the image generation path
                              (``gen_num_experts`` experts, ``gen_num_experts_per_tok``
                              active, expert width ``gen_moe_intermediate_size``).

    Each gen-path knob falls back to its understanding-path counterpart when
    unset, so vanilla single-MoE configs keep working without changes.
    """

    def __init__(
        self,
        rope_theta_hw=10000.0,
        max_position_embeddings_hw=10000,
        gen_num_experts=None,
        gen_num_experts_per_tok=None,
        gen_moe_intermediate_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings_hw = max_position_embeddings_hw

        # Generation-path MoE knobs default to the understanding-path values
        # so legacy single-MoE configs (where both branches share the same
        # router width / expert count) keep working unchanged.
        self.gen_num_experts = (
            int(gen_num_experts) if gen_num_experts is not None else int(self.num_experts)
        )
        self.gen_num_experts_per_tok = (
            int(gen_num_experts_per_tok)
            if gen_num_experts_per_tok is not None
            else int(self.num_experts_per_tok)
        )
        self.gen_moe_intermediate_size = (
            int(gen_moe_intermediate_size)
            if gen_moe_intermediate_size is not None
            else int(self.moe_intermediate_size)
        )

        # ``Qwen3Attention`` (used by NEO-Unify MoE layers) reads
        # ``config.layer_types[layer_idx]`` to decide between ``"full_attention"``
        # and ``"sliding_attention"``. Older / vanilla ``Qwen3MoeConfig`` does
        # not populate that field, so we backfill it here mirroring the dense
        # ``Qwen3Config`` behaviour: sliding-attention layers start at
        # ``max_window_layers`` when ``use_sliding_window`` is enabled.
        existing = getattr(self, "layer_types", None)
        if not existing or len(existing) != self.num_hidden_layers:
            use_swa = bool(getattr(self, "use_sliding_window", False)) and getattr(
                self, "sliding_window", None
            ) is not None
            max_window_layers = int(getattr(self, "max_window_layers", 0) or 0)
            self.layer_types = [
                "sliding_attention" if (use_swa and i >= max_window_layers) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]


def _is_moe_llm_config(llm_config) -> bool:
    """Detect whether an ``llm_config`` (dict or object) targets a MoE backbone.

    Order of checks: explicit ``model_type``, ``architectures`` entry that
    contains ``MoE/MoeForCausalLM``, or presence of MoE-specific keys
    (``num_experts``).
    """
    if isinstance(llm_config, dict):
        model_type = llm_config.get("model_type", "")
        archs = llm_config.get("architectures") or []
        has_num_experts = "num_experts" in llm_config
    else:
        model_type = getattr(llm_config, "model_type", "")
        archs = getattr(llm_config, "architectures", None) or []
        has_num_experts = hasattr(llm_config, "num_experts")

    if isinstance(model_type, str) and "moe" in model_type.lower():
        return True
    for arch in archs:
        arch_str = str(arch)
        if "Moe" in arch_str or "MoE" in arch_str:
            return True
    return bool(has_num_experts) and getattr(llm_config, "num_experts", 0) and int(getattr(llm_config, "num_experts", 0)) > 1


def _build_llm_config(llm_config):
    """Instantiate the right LLM config object from a dict or pre-built config."""
    if isinstance(llm_config, dict):
        if _is_moe_llm_config(llm_config):
            return NEOMoELLMConfig(**llm_config)
        return NEOLLMConfig(**llm_config)
    return llm_config


class NEOChatConfig(PretrainedConfig):
    model_type = 'neo_chat'
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        downsample_ratio=0.5,
        template=None,
        concat_time_token_num=0,
        noise_scale=1.0,
        noise_scale_mode="none",
        noise_scale_base_image_seq_len=256,
        noise_scale_max_value=4.0,
        add_noise_scale_embedding=False,
        time_schedule="standard",
        time_shift_type="linear",
        base_shift=0.5,
        max_shift=1.0,
        base_image_seq_len=256,
        max_image_seq_len=4096,
        fm_head_layers=2,
        fm_head_dim=1536,
        fm_head_mlp_ratio=1.0,
        use_pixel_head=False,
        t_eps=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {'architectures': ['NEOVisionModel']}
            logger.info('vision_config is None. Initializing the NEOVisionConfig with default values.')

        if llm_config is None:
            llm_config = {'architectures': ['Qwen3ForCausalLM']}
            logger.info('llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).')
        if isinstance(llm_config, dict):
            assert 'architectures' in llm_config, "Should specify architecture in llm_config"

        if isinstance(vision_config, dict):
            self.vision_config = NEOVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        self.llm_config = _build_llm_config(llm_config)

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings

        self.concat_time_token_num = concat_time_token_num
        self.noise_scale = noise_scale
        self.noise_scale_mode = noise_scale_mode
        self.noise_scale_base_image_seq_len = noise_scale_base_image_seq_len
        self.noise_scale_max_value = noise_scale_max_value
        self.add_noise_scale_embedding = add_noise_scale_embedding
        self.time_schedule = time_schedule
        self.time_shift_type = time_shift_type
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.fm_head_layers = fm_head_layers
        self.fm_head_dim = fm_head_dim
        self.fm_head_mlp_ratio = fm_head_mlp_ratio
        self.use_pixel_head = use_pixel_head
        self.t_eps = t_eps

    @property
    def is_moe_llm(self) -> bool:
        """Convenience flag so callers can switch between dense / MoE LLM."""
        return isinstance(self.llm_config, NEOMoELLMConfig)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        return output
