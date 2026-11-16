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
        self._last_hidden_states: Optional[torch.Tensor] = None
        self._last_token_hidden_states: Optional[torch.Tensor] = None

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
        return_all_logits: bool = False,
    ) -> torch.Tensor:
        lang_model = self.model.language_model

        # 1. 词嵌入查找
        hidden_states = lang_model.embed_tokens(input_ids)

        # 2. 依次通过 64 层混合解码器 (3x Linear Attention + 1x Full Attention 循环)
        for layer in lang_model.layers:
            hidden_states = layer(
                hidden_states, positions, cache_pool, forward_batch
            )

        # 3. 提取每个请求最后一个 token 的隐状态（或全部 token 隐状态用于 MTP 验证）计算 Logits
        ext_lens_t = (
            forward_batch.extend_lens_tensor
            if forward_batch.extend_lens_tensor is not None
            else torch.tensor(
                forward_batch.extend_lens, dtype=torch.long, device=input_ids.device
            )
        )
        last_token_indices = torch.cumsum(ext_lens_t, dim=0) - 1
        self._last_hidden_states = hidden_states
        self._last_token_hidden_states = hidden_states[last_token_indices]
        if return_all_logits:
            return self.lm_head(lang_model.norm(hidden_states))
        last_hidden = lang_model.norm(self._last_token_hidden_states)
        logits = self.lm_head(last_hidden)  # [batch_size, vocab_size]
        return logits

    @torch.inference_mode()
    def forward_mtp(
        self,
        next_token_ids: torch.Tensor,
        positions: torch.Tensor,
        mtp_cache_pool: Qwen3_8HybridCachePool,
        forward_batch: ForwardBatch,
        main_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        使用 `mtp` 预测头融合主模型隐状态与下一 Token 词嵌入，前向预测投机草稿 Token 的 Logits (`[batch_size, vocab_size]`)。
        """
        if self.mtp is None:
            raise RuntimeError(
                "当前 Qwen3_8ForCausalLM 未启用 MTP 模块 (enable_mtp=False)！"
            )
        if main_hidden_states is None:
            if next_token_ids.shape[0] == forward_batch.batch_size:
                main_hidden_states = self._last_token_hidden_states
            else:
                main_hidden_states = self._last_hidden_states
        assert main_hidden_states is not None, "缺少主模型前向隐状态 main_hidden_states！"

        next_token_embeds = self.model.language_model.embed_tokens(next_token_ids)
        mtp_hidden = self.mtp(
            main_hidden_states=main_hidden_states,
            next_token_embeds=next_token_embeds,
            positions=positions,
            mtp_cache_pool=mtp_cache_pool,
            forward_batch=forward_batch,
        )
        if mtp_hidden.shape[0] > forward_batch.batch_size:
            ext_lens_t = (
                forward_batch.extend_lens_tensor
                if forward_batch.extend_lens_tensor is not None
                else torch.tensor(
                    forward_batch.extend_lens,
                    dtype=torch.long,
                    device=next_token_ids.device,
                )
            )
            last_token_indices = torch.cumsum(ext_lens_t, dim=0) - 1
            mtp_hidden = mtp_hidden[last_token_indices]
        return self.lm_head(mtp_hidden)
