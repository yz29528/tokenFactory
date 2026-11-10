import torch
import torch.nn as nn

from .config import Qwen3_8Config
from .forward_batch import ForwardBatch
from .full_attention import Qwen3_8FullAttention
from .hybrid_cache_pool import Qwen3_8HybridCachePool
from .linear_attention import Qwen3_8LinearAttention
from .mlp import Qwen3_8MLP
from .rms_norm import Qwen3_8RMSNorm


class Qwen3_8DecoderLayer(nn.Module):
    """
    Qwen 3.8 27B 单层解码器模块 (`layers.X`)。
    根据 `config.layer_types[layer_idx]` 自动装配：
    - 当为 `"linear_attention"` (第 0, 1, 2, 4, 5, 6... 层) 时，实例化 `self.linear_attn = Qwen3_8LinearAttention`
    - 当为 `"full_attention"` (第 3, 7, 11, 15...63 层) 时，实例化 `self.self_attn = Qwen3_8FullAttention`
    子模块命名（`input_layernorm`, `post_attention_layernorm`, `linear_attn` / `self_attn`, `mlp`）
    与 `qwen3.8 27B/config.json` 中的权重路径完全一致。
    """

    def __init__(self, config: Qwen3_8Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        self.input_layernorm = Qwen3_8RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        if self.layer_type == "full_attention":
            self.self_attn = Qwen3_8FullAttention(config, layer_idx)
        elif self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_8LinearAttention(config, layer_idx)
        else:
            raise ValueError(f"未知的 layer_type: {self.layer_type}")

        self.post_attention_layernorm = Qwen3_8RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = Qwen3_8MLP(config)

    @torch.no_grad()
    def load_tp_from_full_layer(self, full_layer: "Qwen3_8DecoderLayer") -> None:
        """从单卡完整 `Qwen3_8DecoderLayer` 切片加载当前 `tp_rank` 的权重。"""
        self.input_layernorm.weight.copy_(full_layer.input_layernorm.weight)
        self.post_attention_layernorm.weight.copy_(
            full_layer.post_attention_layernorm.weight
        )
        if self.layer_type == "full_attention":
            self.self_attn.load_tp_from_full_layer(full_layer.self_attn)
        else:
            self.linear_attn.load_tp_from_full_layer(full_layer.linear_attn)
        self.mlp.load_tp_from_full_mlp(full_layer.mlp)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache_pool: Qwen3_8HybridCachePool,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # 1. Attention / Linear Attention 块 (Pre-Norm)
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)

        if self.layer_type == "full_attention":
            attn_out = self.self_attn(normed, positions, cache_pool, forward_batch)
        else:
            attn_out = self.linear_attn(normed, cache_pool, forward_batch)

        hidden_states = residual + attn_out

        # 2. SwiGLU MLP 块 (Pre-Norm)
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed)
        hidden_states = residual + mlp_out

        return hidden_states
