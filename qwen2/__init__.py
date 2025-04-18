
from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
)
from .modeling_lm import Qwen2ForCausalLM,Qwen2Model
from .modeling_qwen2 import Qwen2ForCausalLM1
from .configuration_qwen2 import Qwen2Config
from .image_processing_vlm import VLMImageProcessor