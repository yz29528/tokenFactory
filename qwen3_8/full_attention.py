import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3_8Config
from .forward_batch import ForwardBatch
from .fp8_linear import ColumnParallelFP8Linear, RowParallelFP8Linear
from .hybrid_cache_pool import Qwen3_8HybridCachePool
from .rms_norm import Qwen3_8RMSNorm
from .rotary_embedding import PartialRotaryEmbedding


class Qwen3_8FullAttention(nn.Module):
    """
    Qwen 3.8 27B 的标准全局注意力层 (`self_attn`，每 4 层出现 1 次，共 16 层)。
    采用 Megatron-LM 风格的【先列并行 (Column-Parallel) -> 中间本地 SDPA/門控 (0 通信) -> 后行并行 (Row-Parallel) + 单次 AllReduce】：
    1. 前段（列并行）：
       - `q_proj`, `k_proj`, `v_proj` 均采用 `ColumnParallelFP8Linear`，按 Head 维度切分到各 TP Rank。
    2. 中段（各卡本地独立计算，0 跨卡通信）：
       - `q_norm`, `k_norm` (每个 Head 内部 256 维归一化)、`Partial RoPE`、本地 KV 池分片读写、SDPA 注意力及 `SiLU(gate)` 门控。
    3. 后段（行并行 + 1 次 AllReduce）：
       - `o_proj` 采用 `RowParallelFP8Linear`，沿输入 Head 维切分并在末尾执行单次 `AllReduce`。
    """

    def __init__(self, config: Qwen3_8Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.tp_size = config.tp_size
        self.tp_rank = config.tp_rank

        assert config.num_attention_heads % self.tp_size == 0, (
            f"num_attention_heads={config.num_attention_heads} 无法被 tp_size={self.tp_size} 整除！"
        )

        self.hidden_size = config.hidden_size
        self.global_num_heads = config.num_attention_heads
        self.global_num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn_output_gate = config.attn_output_gate

        # 当前 TP Rank 本地分片头数
        self.num_heads = self.global_num_heads // self.tp_size
        if self.global_num_kv_heads >= self.tp_size:
            assert self.global_num_kv_heads % self.tp_size == 0
            self.num_kv_heads = self.global_num_kv_heads // self.tp_size
        else:
            assert self.tp_size % self.global_num_kv_heads == 0
            self.num_kv_heads = 1

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # 1. 前段：列并行 Q, K, V 投影
        global_q_proj_out_dim = (
            self.global_num_heads * self.head_dim * 2
            if self.attn_output_gate
            else self.global_num_heads * self.head_dim
        )
        global_kv_proj_out_dim = self.global_num_kv_heads * self.head_dim

        self.q_proj = ColumnParallelFP8Linear(
            in_features=self.hidden_size,
            global_out_features=global_q_proj_out_dim,
            bias=config.attention_bias,
            config=config,
        )
        self.k_proj = ColumnParallelFP8Linear(
            in_features=self.hidden_size,
            global_out_features=global_kv_proj_out_dim,
            bias=config.attention_bias,
            config=config,
            num_total_heads=self.global_num_kv_heads,
        )
        self.v_proj = ColumnParallelFP8Linear(
            in_features=self.hidden_size,
            global_out_features=global_kv_proj_out_dim,
            bias=config.attention_bias,
            config=config,
            num_total_heads=self.global_num_kv_heads,
        )

        # 2. 中段：Head-wise Q/K RMSNorm 与 Partial RoPE (每卡保存完整 256 维 norm 权重，0 通信)
        self.q_norm = Qwen3_8RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_8RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = PartialRotaryEmbedding(
            head_dim=self.head_dim,
            partial_rotary_factor=config.partial_rotary_factor,
            rope_theta=config.rope_theta,
            mrope_section=config.mrope_section,
            mrope_interleaved=config.mrope_interleaved,
        )

        # 3. 后段：行并行 o_proj + 单次 AllReduce
        self.o_proj = RowParallelFP8Linear(
            global_in_features=self.global_num_heads * self.head_dim,
            out_features=self.hidden_size,
            bias=config.attention_bias,
            config=config,
        )

    @torch.no_grad()
    def load_tp_from_full_layer(self, full_layer: "Qwen3_8FullAttention") -> None:
        """从单卡完整 `Qwen3_8FullAttention` 切片加载当前 `tp_rank` 的权重。"""
        self.q_proj.load_from_full_linear(full_layer.q_proj)
        self.k_proj.load_from_full_linear(full_layer.k_proj)
        self.v_proj.load_from_full_linear(full_layer.v_proj)
        self.o_proj.load_from_full_linear(full_layer.o_proj)
        self.q_norm.weight.copy_(full_layer.q_norm.weight)
        self.k_norm.weight.copy_(full_layer.k_norm.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache_pool: Qwen3_8HybridCachePool,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]

        # 1. 列并行投影得到本卡分片的 Q (含 Gate)、K、V (0 通信)
        if self.attn_output_gate:
            qg = self.q_proj(hidden_states).view(
                num_tokens, self.num_heads, self.head_dim * 2
            )
            q, gate = qg.chunk(2, dim=-1)
            gate = gate.reshape(num_tokens, self.num_heads * self.head_dim)
        else:
            q = self.q_proj(hidden_states).view(
                num_tokens, self.num_heads, self.head_dim
            )
            gate = None

        k = self.k_proj(hidden_states).view(
            num_tokens, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            num_tokens, self.num_kv_heads, self.head_dim
        )

        # 2. 各卡本地 Q/K RMSNorm + Partial RoPE (0 通信)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(q, k, positions)

        # 3. 写入本卡分片的 TokenToKVPool 物理槽位 (out_cache_loc)
        layer_k_buffer, layer_v_buffer = cache_pool.get_full_kv_buffers(self.layer_idx)
        layer_k_buffer[forward_batch.out_cache_loc] = k
        layer_v_buffer[forward_batch.out_cache_loc] = v

        # 4. 本地 Head 分片执行 SDPA 注意力与 Swish 门控 (0 通信)
        attn_output = self._slot_based_attention(
            q, layer_k_buffer, layer_v_buffer, forward_batch
        )
        attn_output = attn_output.reshape(num_tokens, self.num_heads * self.head_dim)

        if gate is not None:
            attn_output = attn_output * F.silu(gate)

        # 5. 行并行 o_proj (内部执行单次 AllReduce)
        return self.o_proj(attn_output)

    def _slot_based_attention(
        self,
        q: torch.Tensor,
        layer_k_buffer: torch.Tensor,
        layer_v_buffer: torch.Tensor,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        outputs = []
        token_offset = 0

        for i in range(batch.batch_size):
            req_idx = int(batch.req_pool_indices[i].item())
            ext_len = batch.extend_lens[i]
            seq_len = int(batch.seq_lens[i].item())

            q_i = q[token_offset : token_offset + ext_len]
            token_offset += ext_len

            kv_slots = batch.req_to_token[req_idx, :seq_len]
            k_i = layer_k_buffer[kv_slots]
            v_i = layer_v_buffer[kv_slots]

            k_i = torch.repeat_interleave(k_i, self.num_queries_per_kv, dim=1)
            v_i = torch.repeat_interleave(v_i, self.num_queries_per_kv, dim=1)

            q_sdpa = q_i.transpose(0, 1).unsqueeze(0)
            k_sdpa = k_i.transpose(0, 1).unsqueeze(0)
            v_sdpa = v_i.transpose(0, 1).unsqueeze(0)

            attn_mask = None
            if ext_len > 1:
                q_pos = torch.arange(
                    seq_len - ext_len, seq_len, device=q.device
                ).unsqueeze(1)
                k_pos = torch.arange(seq_len, device=q.device).unsqueeze(0)
                mask_cond = q_pos >= k_pos
                attn_mask = torch.zeros(
                    (ext_len, seq_len), dtype=q.dtype, device=q.device
                ).masked_fill(~mask_cond, float("-inf"))

            out_i = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, scale=self.scaling
            )
            outputs.append(out_i.squeeze(0).transpose(0, 1))

        return torch.cat(outputs, dim=0)
