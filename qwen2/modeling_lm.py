from typing import Callable, Any, Dict,  List, Optional, Tuple, Union

import torch
from torch import nn
import torch_npu
from einops import rearrange, repeat
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from .configuration_qwen2 import Qwen2Config
from .unet import UNetModel

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-qwen2/Qwen2-2-7b-hf"
_CONFIG_FOR_DOC = "Qwen2Config"


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.diffusion_q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.diffusion_k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.diffusion_v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.diffusion_o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)


    def forward(
        self,
        text_embeds: torch.Tensor,
        img_embeds: torch.Tensor,
        temb: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        task=None,
        img_mask=None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bs,il=img_embeds.shape[:-1]
        hidden_shape = (bs,il, -1, self.head_dim)
        v_query_states = self.diffusion_q_proj(img_embeds).view(hidden_shape).transpose(1, 2)
        v_key_states = self.diffusion_k_proj(img_embeds).view(hidden_shape).transpose(1, 2)
        v_value_states = self.diffusion_v_proj(img_embeds).view(hidden_shape).transpose(1, 2)
        bs,tl=text_embeds.shape[:-1]
        hidden_shape = (bs,tl, -1, self.head_dim)
        t_query_states = self.q_proj(text_embeds).view(hidden_shape).transpose(1, 2)
        t_key_states = self.k_proj(text_embeds).view(hidden_shape).transpose(1, 2)
        t_value_states = self.v_proj(text_embeds).view(hidden_shape).transpose(1, 2)
        if task=="image_under":
            temp_q_img_mask=img_mask.unsqueeze(1).expand(-1,t_query_states.shape[1], -1)
            temp_v_img_mask=img_mask.unsqueeze(1).expand(-1,t_value_states.shape[1], -1)
            t_query_states[temp_q_img_mask]=v_query_states.contiguous().view(-1, self.head_dim)
            t_key_states[temp_v_img_mask]=v_key_states.contiguous().view(-1, self.head_dim)
            t_value_states[temp_v_img_mask]=v_value_states.contiguous().view(-1, self.head_dim)
            query_states=t_query_states
            key_states=t_key_states
            value_states=t_value_states
        else:
            query_states=torch.cat([t_query_states,v_query_states],dim=2)
            key_states=torch.cat([t_key_states,v_key_states],dim=2)
            value_states=torch.cat([t_value_states,v_value_states],dim=2)
        
        cos, sin = position_embeddings
        #print(query_states.shape,key_states.shape,cos.shape,sin.shape)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        if task=="image_under":

            print ("bs",bs)
            print ("il",il)
            print ("tl",tl)


            attn_output = attn_output.reshape(bs,tl, -1).contiguous() #b*l*c


            print(" attn_output", attn_output.shape)
            img_embeds=attn_output[img_mask].view(bs,il,-1)
            text_embeds=attn_output*(~img_mask.unsqueeze(-1))
            print("text_embeds",text_embeds.shape)
            print("img_embeds",img_embeds.shape)
        else:
            attn_output = attn_output.reshape(bs,tl+il, -1).contiguous()
            text_embeds,img_embeds=attn_output[:,:tl],attn_output[:,tl:]
        
        text_embeds = self.o_proj(text_embeds)
        img_embeds = self.diffusion_o_proj(img_embeds)
        return text_embeds,img_embeds


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)

        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.diffusion_ada_ln = nn.Linear(512, config.hidden_size * 6)
        self.diffusion_norm = nn.LayerNorm(config.hidden_size, eps=1e-5, elementwise_affine=True)
        nn.init.zeros_(self.diffusion_ada_ln.weight)
        nn.init.zeros_(self.diffusion_ada_ln.bias)

        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.diffusion_norm_post = nn.LayerNorm(config.hidden_size, eps=1e-5, elementwise_affine=True)

        self.mlp = Qwen2MLP(config)
        self.diffusion_mlp = Qwen2MLP(config)
           

    def forward(
        self,
        text_embeds: torch.Tensor,
        img_embeds: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        img_mask: Optional[torch.Tensor] = None,
        task=None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        t_residual = text_embeds
        text_embeds = self.input_layernorm(text_embeds)

        v_residual = img_embeds
        shift, scale, gate,shift_mlp, scale_mlp, gate_mlp= self.diffusion_ada_ln(temb).chunk(6, dim=1)
        img_embeds = self.diffusion_norm(img_embeds) * (1 + scale)[:, None, :] + shift[:, None, :]
        # Self Attention
        text_embeds,img_embeds = self.self_attn(
            text_embeds=text_embeds,
            img_embeds=img_embeds,
            temb=temb,
            attention_mask=attention_mask,
            task=task,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            img_mask=img_mask,
            **kwargs,
        )
        
        img_embeds = v_residual + img_embeds*gate.unsqueeze(1)
        text_embeds = t_residual + text_embeds
        t_residual = text_embeds
        text_embeds = self.post_attention_layernorm(text_embeds)
        text_embeds = self.mlp(text_embeds)
        text_embeds = t_residual + text_embeds

        v_residual = img_embeds
        img_embeds = self.diffusion_norm_post(img_embeds) * (1 + scale_mlp)[:, None, :] + shift_mlp[:, None, :]
        img_embeds = self.diffusion_mlp(img_embeds)
        img_embeds = v_residual + img_embeds*gate_mlp.unsqueeze(1)

        return text_embeds,img_embeds


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


QWEN2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
import einx
def naive_attn_mask(
    seq_len,
    start,
    end,
    device = None,
    pad_mask=None
):
    seq = torch.arange(seq_len, device = device)
    is_causal = einx.greater_equal('i, j -> i j', seq, seq)
    is_modality = ((seq >= start).unsqueeze(1) & (seq < end).unsqueeze(0))
    attn_mask=is_causal | is_modality
    if pad_mask is not None:
        # 将pad_mask从(b, seq_len)扩展到(b, 1, seq_len)
        # 这样广播后，任何查询位置如果对应填充token都会被屏蔽
        pad_mask = pad_mask.unsqueeze(1)  # 形状变为(b, 1, seq_len)
        pad_mask = pad_mask.expand(-1, seq_len, -1)
        attn_mask=attn_mask.expand(pad_mask.shape[0], -1, -1)
        attn_mask = attn_mask & pad_mask  # 广播合并
    return attn_mask
def img_under_attn_mask( input_ids, img_mask,pad_mask=None):
    batch_size, seq_len = input_ids.shape[0],input_ids.shape[1]
    device = input_ids.device
    print(seq_len)

    autoregressive_mask = torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    ).unsqueeze(0)

    autoregressive_mask = autoregressive_mask.expand(batch_size, -1, -1)  # (batch_size, seq_len, seq_len)

    # 构建图像token之间的全连接掩码
    img_mask_row = img_mask.unsqueeze(2)  # (batch_size, seq_len, 1)
    img_mask_col = img_mask.unsqueeze(1)  # (batch_size, 1, seq_len)
    image_full_mask = img_mask_row & img_mask_col  # (batch_size, seq_len, seq_len)

    # 合并掩码：对于图像token的位置，使用自回归掩码或图像全连接掩码
    combined_mask = torch.where(
        img_mask.unsqueeze(-1),  # 条件：当前行是否为图像token (batch_size, seq_len, 1)
        autoregressive_mask | image_full_mask,  # 若为图像token，合并掩码
        autoregressive_mask  # 否则保持自回归掩码
    )
    if pad_mask is not None:
        pad_mask_2d = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)
        combined_mask = combined_mask & pad_mask_2d

    return combined_mask

def ForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss
@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.text_embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.diffusion_time_proj = Timesteps(config.hidden_size, True, 0)
        self.diffusion_time_embedding = TimestepEmbedding(config.hidden_size, 512, "silu")
        self.gradient_checkpointing = False
        self.act=nn.SiLU()

        self.diffusion_ada_ln = nn.Linear(512, config.hidden_size * 2)
        self.diffusion_norm = nn.LayerNorm(config.hidden_size, eps=1e-5, elementwise_affine=True)
        nn.init.zeros_(self.diffusion_ada_ln.weight)
        nn.init.zeros_(self.diffusion_ada_ln.bias)


        # Initialize weights and apply final processing
        self.post_init()


    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        img_embeds: torch.Tensor = None,
        timesteps:torch.LongTensor=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        pad_mask=None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = img_embeds.device if input_ids is not None else img_embeds.device
        dtype = img_embeds.dtype if input_ids is not None else img_embeds.dtype
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        input_ids[input_ids < 0] = 0 
        text_embeds = self.text_embed_tokens(input_ids)
        
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        timesteps=timesteps.to(device=device)
        temb = self.diffusion_time_proj(timesteps)
        temb = temb.to(dtype=dtype)
        temb = self.act(self.diffusion_time_embedding(temb))
        image_length=img_embeds.shape[1]
        text_length=text_embeds.shape[1]
        if kwargs["task"]=="image_under":
            print(kwargs.keys())
            attn_mask = img_under_attn_mask(text_embeds,kwargs["image_mask"],pad_mask=pad_mask).bool() #& attention_mask
            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + text_embeds.shape[1], device=text_embeds.device
                )
            hidden_states = text_embeds

        else:
            attn_mask = naive_attn_mask(text_length+image_length,text_length,text_length+image_length,device=text_embeds.device,pad_mask=pad_mask).bool()
            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + text_embeds.shape[1]+img_embeds.shape[1], device=text_embeds.device
                )
            hidden_states = torch.cat([text_embeds,img_embeds],dim=1)
            

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        position_embeddings = self.rotary_emb(hidden_states, position_ids) #b*l*c, 1*L
        #import pdb
        #pdb.set_trace()
        
        #print(attn_mask[:,0,0])
        # x =kwargs.get("task")
        # print ("k_mask",x)
    
        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    text_embeds,
                    img_embeds,
                    temb,
                    attn_mask,
                    kwargs.get("image_mask",None),
                    kwargs.get("task"),
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    text_embeds,
                    img_embeds,
                    temb,
                    attention_mask=attn_mask,
                    img_mask=kwargs.get("image_mask",None),
                    task=kwargs.get("task"),
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    # **kwargs,
                )

            text_embeds,img_embeds=layer_outputs
        text_embeds = self.norm(text_embeds)
        shift, scale= self.diffusion_ada_ln(temb).chunk(2, dim=1)
        img_embeds = self.diffusion_norm(img_embeds) * (1 + scale)[:, None, :] + shift[:, None, :]

        output = {
            "text_embeds":text_embeds,
            "img_embeds":img_embeds,
            "past_key_values":past_key_values if use_cache else None,}
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'"
                        " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                        " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2Config,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                # if we have sliding window, we should not attend to tokens beyond sliding window length, so we mask them out also
                # the check is needed to verify is current checkpoint was trained with sliding window or not
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...

class MlpProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        mlp_depth = cfg.get("depth", 1)
        modules = [nn.Linear(cfg["input_dim"], cfg["n_embed"])]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg["n_embed"], cfg["n_embed"]))
        modules = nn.Sequential(*modules)
        self.layers = modules

    def forward(
        self, x:  torch.Tensor
    ):
    #8*16*48*48
        return self.layers(x)
from diffusers import AutoencoderKL
from diffusers import FlowMatchEulerDiscreteScheduler     
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.vision_model = AutoencoderKL.from_pretrained(
            config.vae_path, 
            torch_dtype=torch.float16        
        )
        #self.diffusion_encoder=MlpProjector(config.gen_encoder_config["params"])
        #self.diffusion_decoder=MlpProjector(config.gen_decoder_config["params"])
        self.diffusion_projector=UNetModel(**config.gen_projector["params"])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.noise_scheduler=FlowMatchEulerDiscreteScheduler(**{"num_train_timesteps": 1000,"shift": 3.0})
        self.sigmas = self.noise_scheduler.sigmas.to(dtype=torch.float16)
        self.schedule_timesteps = self.noise_scheduler.timesteps
        self.ar_loss=config.ar_loss
        self.diffusion_loss=config.diffusion_loss
    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        pass
    def image_to_latent(self,x):
        #x=B*3*H*W
        z=self.vision_model.encode(x).latent_dist.sample()
        z=self.vision_model.scaling_factor*z+self.vision_model.shift_factor
        return z
    def latent_to_image(self,z):
        z=(z-self.vision_model.shift_factor)/self.vision_model.scaling_factor
        x=self.vision_model.decode(z).sample
        return x
    def get_sigmas(self,timesteps, n_dim=4, dtype=torch.float32):
        step_indices = [(self.schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = self.sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values:torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        timesteps=None,
        is_train=True,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #kwargs["task"]="image_under"#加的
        if kwargs["task"]=="image_under":
            timesteps=torch.zeros([input_ids.shape[0]])
            z=self.image_to_latent(pixel_values)
            img_embeds=self.diffusion_projector.encode(z)
        else:
            if timesteps is None:
                z=self.image_to_latent(pixel_values)
                #b*(h*w)*16
                noise = torch.randn_like(z)
                u = compute_density_for_timestep_sampling(
                        weighting_scheme="logit_normal",
                        batch_size=z.shape[0],
                        logit_mean=0,
                        logit_std=1
                    )
                indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
                timesteps = self.schedule_timesteps[indices]
                sigmas = self.get_sigmas(timesteps, n_dim=z.ndim, dtype=z.dtype).to(device=z.device)
                noisy_latent = (1.0 - sigmas) * z + sigmas * noise
            else:
                noise = torch.randn((1, 16, 32,32), device=self.model.device, dtype= torch.float32 )
                noisy_latent= pixel_values  
            img_embeds=self.diffusion_projector.encode(noisy_latent)
        tmp_shape=img_embeds.shape
        img_input=img_embeds.view(tmp_shape[0],tmp_shape[1],-1).permute(0, 2, 1)
        outputs = self.model(
            input_ids=input_ids,
            img_embeds=img_input,
            timesteps=timesteps,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        
        text_output=outputs["text_embeds"]
        img_output=outputs["img_embeds"]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        #slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        if text_output is not None:
            t_logits = self.lm_head(text_output)
        
        img_output=img_output.permute(0, 2, 1).view(tmp_shape)
        img_output=self.diffusion_projector.decode(img_output)
        img_res=img_output.view(tmp_shape[0],16,-1).permute(0, 2, 1)
        loss = None
        #import pdb 
        #pdb.set_trace()
        if self.ar_loss and kwargs["task"]=="image_under":
            ar_loss = self.loss_function(logits=t_logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)*0.2
        else:
            ar_loss=0
        
        if self.diffusion_loss and is_train and kwargs["task"]=="image_gen":
            b, l,c= img_res.shape #b*l*c

            weighting = compute_loss_weighting_for_sd3(weighting_scheme="logit_normal", sigmas=sigmas)

            # flow matching loss
            target = noise - z
            target=target.view(tmp_shape[0],16,-1).permute(0, 2, 1)

            # Compute regular loss.
            loss_mse = (weighting.float() * (img_res.float() - target.float()) ** 2).reshape(target.shape[0], -1)
            diffusion_loss = loss_mse.mean()
        else:
            diffusion_loss=0
        loss=ar_loss+diffusion_loss
        if kwargs["task"]=="image_gen":
            return {
                "loss":loss,
                "pred":img_output,
                "latent":noise -img_output,
                "noise": noisy_latent,
                "gt":pixel_values
                #"past_key_values":outputs.past_key_values,
            }
        if kwargs["task"]=="image_under":
            return CausalLMOutputWithPast(
                loss=loss,
                logits=text_output,
                # past_key_values=outputs.past_key_values,
                # hidden_states=outputs.hidden_states,
                # attentions=outputs.attentions,
            )
        else:
            return {
                "loss":loss,
                "pred":img_output,
                #"past_key_values":outputs.past_key_values,
            }

