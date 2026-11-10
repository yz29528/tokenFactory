from typing import Callable, Optional
import torch
import torch.nn as nn

from .config import Qwen3_8Config
from .decoder_layer import Qwen3_8DecoderLayer
from .forward_batch import ForwardBatch
from .fp8_linear import RowParallelFP8Linear
from .hybrid_cache_pool import Qwen3_8HybridCachePool
from .mtp import Qwen3_8MTPBlock
from .rms_norm import Qwen3_8RMSNorm


class Qwen3_8TextModel(nn.Module):
    """
    对应 `model.safetensors.index.json` 中的 `model.language_model` 主干：
    - `embed_tokens`: [vocab_size=248320, hidden_size=5120]
    - `layers`: 64 层混合解码器（48 层 `linear_attention` Gated DeltaNet + 16 层 `full_attention` GQA）
    - `norm`: 顶层 `Qwen3_8RMSNorm`
    """

    def __init__(self, config: Qwen3_8Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3_8DecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_8RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3_8ModelWrapper(nn.Module):
    """
    对应 `model.safetensors.index.json` 中的 `model.*` 容器层级（挂载 `language_model`）。
    """

    def __init__(self, config: Qwen3_8Config):
        super().__init__()
        self.language_model = Qwen3_8TextModel(config)


class Qwen3_8ForCausalLM(nn.Module):
    """
    Qwen 3.8 27B 顶层因果语言模型（权重路径与 `qwen3.8 27B/model.safetensors.index.json` 100% 精确对齐）：
    - `model.language_model.embed_tokens`: [vocab_size=248320, hidden_size=5120]
    - `model.language_model.layers.{0..63}`: 64 层混合解码器
    - `model.language_model.norm`: 顶层 `Qwen3_8RMSNorm`
    - `lm_head`: 独立输出头 (`tie_word_embeddings = False`)
    - `mtp`: Multi-Token Prediction 预测模块 (`mtp_num_hidden_layers = 1`)
    """

    def __init__(self, config: Qwen3_8Config, enable_mtp: bool = False):
        super().__init__()
        self.config = config
        self.model = Qwen3_8ModelWrapper(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

        self.mtp = (
            Qwen3_8MTPBlock(config)
            if (enable_mtp and config.mtp_num_hidden_layers > 0)
            else None
        )

    @property
    def language_model(self) -> Qwen3_8TextModel:
        return self.model.language_model

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.model.language_model.embed_tokens

    @property
    def layers(self) -> nn.ModuleList:
        return self.model.language_model.layers

    @property
    def norm(self) -> Qwen3_8RMSNorm:
        return self.model.language_model.norm

    def set_all_reduce_fn(
        self, all_reduce_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]
    ) -> None:
        """为模型内所有 `RowParallelFP8Linear` 注册自定义的 `AllReduce` 回调（用于多进程或单机多 Rank 验证）。"""
        for module in self.modules():
            if isinstance(module, RowParallelFP8Linear):
                module.all_reduce_fn = all_reduce_fn

    @torch.no_grad()
    def load_tp_from_full_model(self, full_model: "Qwen3_8ForCausalLM") -> None:
        """从单卡完整 `Qwen3_8ForCausalLM (tp_size=1)` 将各层权重按“先列并行、后行并行”规则切分加载到当前 `tp_rank`。"""
        self.embed_tokens.weight.copy_(full_model.embed_tokens.weight)
        self.norm.weight.copy_(full_model.norm.weight)
        self.lm_head.weight.copy_(full_model.lm_head.weight)
        for tp_layer, full_layer in zip(self.layers, full_model.layers):
            tp_layer.load_tp_from_full_layer(full_layer)
        if self.mtp is not None and full_model.mtp is not None:
            self.mtp.pre_fc_norm_embedding.weight.copy_(
                full_model.mtp.pre_fc_norm_embedding.weight
            )
            self.mtp.pre_fc_norm_hidden.weight.copy_(
                full_model.mtp.pre_fc_norm_hidden.weight
            )
            self.mtp.fc.weight.copy_(full_model.mtp.fc.weight)
            self.mtp.norm.weight.copy_(full_model.mtp.norm.weight)
            for tp_l, full_l in zip(self.mtp.layers, full_model.mtp.layers):
                tp_l.load_tp_from_full_layer(full_l)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,          # [num_tokens] 拍平的一维输入 Token 序列
        positions: torch.Tensor,          # [num_tokens] 或 [3, num_tokens] (M-RoPE)
        cache_pool: Qwen3_8HybridCachePool,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        lang_model = self.model.language_model

        # 1. 词嵌入查找
        hidden_states = lang_model.embed_tokens(input_ids)

        # 2. 依次通过 64 层混合解码器 (3x Linear Attention + 1x Full Attention 循环)
        for layer in lang_model.layers:
            hidden_states = layer(
                hidden_states, positions, cache_pool, forward_batch
            )

        # 3. 仅提取每个请求最后一个 token 的隐状态计算下一个 token 的 Logits
        last_token_indices = (
            torch.cumsum(
                torch.tensor(forward_batch.extend_lens, device=input_ids.device),
                dim=0,
            )
            - 1
        )
        last_hidden = lang_model.norm(hidden_states[last_token_indices])
        logits = self.lm_head(last_hidden)  # [batch_size, vocab_size]
        return logits
