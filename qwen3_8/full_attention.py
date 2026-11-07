import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3_8Config
from .forward_batch import ForwardBatch
from .hybrid_cache_pool import Qwen3_8HybridCachePool
from .rms_norm import Qwen3_8RMSNorm
from .rotary_embedding import PartialRotaryEmbedding


class Qwen3_8FullAttention(nn.Module):
    """
    Qwen 3.8 27B 的标准全局注意力层 (`self_attn`，每 4 层出现 1 次，共 16 层)。
    核心特性（严格对应 `qwen3.8 27B/config.json`）：
    1. GQA 架构：`num_attention_heads = 24`, `num_key_value_heads = 4` (6:1 分组查询), `head_dim = 256`
    2. Q/K RMSNorm：包含独立的 `q_norm` 和 `k_norm` (维度 256)
    3. Partial RoPE：`partial_rotary_factor = 0.25`，仅对每个 Head 的前 64 维施加旋转编码
    4. Attention Output Gate (`attn_output_gate = True`, `output_gate_type = "swish"`)：
       `q_proj` 同时投影出 Query 与 Gate 分支，注意力输出经 `SiLU(gate)` 门控后再进入 `o_proj`
    5. 对接 SGLang 两级显存池：通过 `out_cache_loc` 写入 `TokenToKVPool`，通过 `req_to_token` 寻址历史 KV
    """

    def __init__(self, config: Qwen3_8Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn_output_gate = config.attn_output_gate

        # 当启用 attn_output_gate 时，q_proj 输出 2 倍通道（前半为 Q，后半为 Swish Output Gate）
        q_proj_out_dim = (
            self.num_heads * self.head_dim * 2
            if self.attn_output_gate
            else self.num_heads * self.head_dim
        )
        self.q_proj = nn.Linear(
            self.hidden_size, q_proj_out_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        # Q / K Head-wise RMSNorm (对应 config.json 中的 self_attn.q_norm / k_norm)
        self.q_norm = Qwen3_8RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_8RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Partial RoPE (rotary_dim = 64, pass_dim = 192)
        self.rotary_emb = PartialRotaryEmbedding(
            head_dim=self.head_dim,
            partial_rotary_factor=config.partial_rotary_factor,
            rope_theta=config.rope_theta,
            mrope_section=config.mrope_section,
            mrope_interleaved=config.mrope_interleaved,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache_pool: Qwen3_8HybridCachePool,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]

        # 1. Q (含可选 Output Gate)、K、V 线性投影
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

        # 2. Q/K RMSNorm + Partial RoPE (仅旋转前 64 维)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(q, k, positions)

        # 3. 将新计算的 K, V 写入第二层池 TokenToKVPool 的对应物理槽位 (out_cache_loc)
        layer_k_buffer, layer_v_buffer = cache_pool.get_full_kv_buffers(self.layer_idx)
        layer_k_buffer[forward_batch.out_cache_loc] = k
        layer_v_buffer[forward_batch.out_cache_loc] = v

        # 4. 通过第一层池 ReqToTokenPool 收集每个请求的物理槽位并计算 Attention
        attn_output = self._slot_based_attention(
            q, layer_k_buffer, layer_v_buffer, forward_batch
        )
        attn_output = attn_output.reshape(num_tokens, self.num_heads * self.head_dim)

        # 5. 应用 Swish/SiLU Output Gate 并执行输出投影
        if gate is not None:
            attn_output = attn_output * F.silu(gate)

        return self.o_proj(attn_output)

    def _slot_based_attention(
        self,
        q: torch.Tensor,
        layer_k_buffer: torch.Tensor,
        layer_v_buffer: torch.Tensor,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        """
        基于 SGLang 的二级索引寻址：
        先从第一层 `batch.req_to_token[req_idx, :seq_len]` 取出物理槽位列表 `kv_slots`，
        再从第二层 `layer_k_buffer[kv_slots]` 取出实际的 K/V 向量执行 GQA 注意力计算。
        """
        outputs = []
        token_offset = 0

        for i in range(batch.batch_size):
            req_idx = int(batch.req_pool_indices[i].item())
            ext_len = batch.extend_lens[i]
            seq_len = int(batch.seq_lens[i].item())

            q_i = q[token_offset : token_offset + ext_len]  # [ext_len, num_heads, head_dim]
            token_offset += ext_len

            # 第一层寻址：从 ReqToTokenPool 获取该请求所有历史+当前 token 的物理槽位索引
            kv_slots = batch.req_to_token[req_idx, :seq_len]  # [seq_len]

            # 第二层寻址：从 TokenToKVPool 聚集出真实的 K, V 张量
            k_i = layer_k_buffer[kv_slots]  # [seq_len, num_kv_heads, head_dim]
            v_i = layer_v_buffer[kv_slots]

            # GQA 广播: 4 个 KV Head 扩展为 24 个 Head
            k_i = torch.repeat_interleave(k_i, self.num_queries_per_kv, dim=1)
            v_i = torch.repeat_interleave(v_i, self.num_queries_per_kv, dim=1)

            q_sdpa = q_i.transpose(0, 1).unsqueeze(0)  # [1, num_heads, ext_len, head_dim]
            k_sdpa = k_i.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]
            v_sdpa = v_i.transpose(0, 1).unsqueeze(0)

            # Prefill / Extend 阶段构造因果掩码 (兼容 RadixCache 命中前缀的情况)
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
