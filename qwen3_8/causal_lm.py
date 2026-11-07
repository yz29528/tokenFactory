import torch
import torch.nn as nn

from .config import Qwen3_8Config
from .decoder_layer import Qwen3_8DecoderLayer
from .forward_batch import ForwardBatch
from .hybrid_cache_pool import Qwen3_8HybridCachePool
from .mtp import Qwen3_8MTPBlock
from .rms_norm import Qwen3_8RMSNorm


class Qwen3_8ForCausalLM(nn.Module):
    """
    Qwen 3.8 27B 顶层因果语言模型：
    - `embed_tokens`: [vocab_size=248320, hidden_size=5120]
    - `layers`: 64 层混合解码器（48 层 `linear_attention` Gated DeltaNet + 16 层 `full_attention` GQA）
    - `norm`: 顶层 `Qwen3_8RMSNorm`
    - `lm_head`: 独立输出头 (`tie_word_embeddings = False`)
    - `mtp`: 可选的 Multi-Token Prediction 预测模块 (`mtp_num_hidden_layers = 1`)
    """

    def __init__(self, config: Qwen3_8Config, enable_mtp: bool = False):
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
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.mtp = (
            Qwen3_8MTPBlock(config)
            if (enable_mtp and config.mtp_num_hidden_layers > 0)
            else None
        )

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,          # [num_tokens] 拍平的一维输入 Token 序列
        positions: torch.Tensor,          # [num_tokens] 或 [3, num_tokens] (M-RoPE)
        cache_pool: Qwen3_8HybridCachePool,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # 1. 词嵌入查找
        hidden_states = self.embed_tokens(input_ids)

        # 2. 依次通过 64 层混合解码器 (3x Linear Attention + 1x Full Attention 循环)
        for layer in self.layers:
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
        last_hidden = self.norm(hidden_states[last_token_indices])
        logits = self.lm_head(last_hidden)  # [batch_size, vocab_size]
        return logits
