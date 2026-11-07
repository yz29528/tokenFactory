from dataclasses import replace
import torch
import torch.nn as nn

from .config import Qwen3_8Config
from .decoder_layer import Qwen3_8DecoderLayer
from .forward_batch import ForwardBatch
from .hybrid_cache_pool import Qwen3_8HybridCachePool
from .rms_norm import Qwen3_8RMSNorm


class Qwen3_8MTPBlock(nn.Module):
    """
    Qwen 3.8 27B 的 Multi-Token Prediction (MTP) 预测头模块。
    对应 `qwen3.8 27B/config.json` 中的：
    - `mtp_num_hidden_layers = 1`
    - `mtp_use_dedicated_embeddings = False`
    - 权重路径：`mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`, `mtp.fc`, `mtp.layers.0`, `mtp.norm`
    用于投机采样（Speculative Decoding）中融合主干隐状态与下一 Token 词嵌入并预测后续 Token。
    """

    def __init__(self, config: Qwen3_8Config):
        super().__init__()
        self.config = config
        self.pre_fc_norm_embedding = Qwen3_8RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = Qwen3_8RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        # MTP 层在 config.json 中使用 full_attention (见 mtp.layers.0.self_attn.q_norm / k_norm)
        mtp_cfg = replace(
            config,
            num_hidden_layers=config.mtp_num_hidden_layers,
            layer_types=["full_attention"] * config.mtp_num_hidden_layers,
        )
        self.layers = nn.ModuleList(
            [
                Qwen3_8DecoderLayer(mtp_cfg, i)
                for i in range(config.mtp_num_hidden_layers)
            ]
        )
        self.norm = Qwen3_8RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @torch.inference_mode()
    def forward(
        self,
        main_hidden_states: torch.Tensor,
        next_token_embeds: torch.Tensor,
        positions: torch.Tensor,
        mtp_cache_pool: Qwen3_8HybridCachePool,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        normed_emb = self.pre_fc_norm_embedding(next_token_embeds)
        normed_hid = self.pre_fc_norm_hidden(main_hidden_states)
        hidden_states = self.fc(torch.cat([normed_emb, normed_hid], dim=-1))

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, positions, mtp_cache_pool, forward_batch
            )

        return self.norm(hidden_states)
