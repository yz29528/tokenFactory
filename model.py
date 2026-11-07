from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. 配置文件与 SGLang 批次元数据 (Config & ForwardBatch)
# ============================================================================


@dataclass
class Gemma4Config:
    """对齐 google/gemma-4-E4B 的默认配置参数"""

    vocab_size: int = 262144
    hidden_size: int = 2560
    intermediate_size: int = 6144
    num_hidden_layers: int = 35
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 4
    head_dim: int = 256
    global_head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_global_theta: float = 1000000.0
    sliding_window: int = 4096

    # Gemma 4 特有架构开关
    attention_k_eq_v: bool = True  # 全局注意力层复用 K 作为 V
    num_kv_shared_layers: int = 20  # 后 20 层复用前面的 KV Cache (YOCO架构)
    use_double_wide_mlp: bool = True  # KV 共享层使用 2x intermediate_size
    hidden_size_per_layer_input: int = 256  # Per-Layer Embeddings (PLE) 维度
    final_logit_softcapping: Optional[float] = 30.0
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        # 默认每 6 层设置 5 层 sliding_attention + 1 层 full_attention
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]


@dataclass
class ForwardBatch:
    """模仿 SGLang 的 ForwardBatch，描述当前 Step 拍平后的批次信息"""

    batch_size: int
    seq_lens: torch.Tensor  # [batch_size] 每个序列当前总长度
    prefix_lens: torch.Tensor  # [batch_size] 命中的 RadixCache 前缀长度
    extend_lens: List[int]  # 每个序列本次需要前向计算的 token 数 (Decode时全为1)
    out_cache_loc: (
        torch.Tensor
    )  # [num_tokens] 本次计算的每个 token 写入 KV Pool 的物理槽位索引
    block_tables: torch.Tensor  # [batch_size, max_blocks] 每个序列对应的物理块表
    block_size: int = 16
    is_prefill: bool = True


# ============================================================================
# 2. 基础组件：Gemma4RMSNorm 与 RoPE
# ============================================================================


class Gemma4RMSNorm(nn.Module):
    """Gemma 系列标志性的 RMSNorm，支持带缩放权重与无缩放权重 (Scale-free V-Norm)"""

    def __init__(
        self, dim: int, eps: float = 1e-6, with_scale: bool = True
    ):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        if self.with_scale:
            return (x_normed * self.weight.float()).to(orig_dtype)
        return x_normed.to(orig_dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_dim: int,
        max_position: int = 131072,
        base: float = 10000.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        # x: [num_tokens, num_heads, head_dim], positions: [num_tokens]
        freqs = torch.outer(positions.float(), self.inv_freq)  # [num_tokens, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [num_tokens, head_dim]
        cos = emb.cos().unsqueeze(1).to(x.dtype)  # [num_tokens, 1, head_dim]
        sin = emb.sin().unsqueeze(1).to(x.dtype)

        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin


# ============================================================================
# 3. 核心算子：Gemma4Attention (支持 Paged KV Cache 与跨层 KV 共享)
# ============================================================================


class Gemma4Attention(nn.Module):

    def __init__(
        self,
        config: Gemma4Config,
        layer_idx: int,
        kv_source_layer: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.kv_source_layer = kv_source_layer  # 若 != layer_idx，说明该层复用之前层的 KV Cache
        self.is_kv_shared = kv_source_layer != layer_idx

        self.layer_type = config.layer_types[layer_idx]
        self.is_full_attention = self.layer_type == "full_attention"

        # 确定当前层的 head_dim 和 kv_heads
        self.head_dim = (
            config.global_head_dim
            if self.is_full_attention
            else config.head_dim
        )
        self.num_heads = config.num_attention_heads

        # 全局层支持 K=V (无 v_proj)
        self.use_k_eq_v = self.is_full_attention and config.attention_k_eq_v
        if self.use_k_eq_v:
            self.num_kv_heads = config.num_global_key_value_heads
        else:
            self.num_kv_heads = config.num_key_value_heads

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        # Q 投影与归一化 (所有层都有)
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # 只有非 KV 共享层才需要实例化 K/V 投影
        if not self.is_kv_shared:
            self.k_proj = nn.Linear(
                config.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=False,
            )
            self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if not self.use_k_eq_v:
                self.v_proj = nn.Linear(
                    config.hidden_size,
                    self.num_kv_heads * self.head_dim,
                    bias=False,
                )
            # Gemma 4 对 V 采用无缩放权重的 RMSNorm
            self.v_norm = Gemma4RMSNorm(
                self.head_dim, eps=config.rms_norm_eps, with_scale=False
            )

        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        rope_theta = (
            config.rope_global_theta
            if self.is_full_attention
            else config.rope_theta
        )
        self.rotary_emb = RotaryEmbedding(self.head_dim, base=rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [num_tokens, hidden_size]
        positions: torch.Tensor,  # [num_tokens]
        kv_pool: Tuple[torch.Tensor, torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        k_cache, v_cache = kv_pool
        # k_cache 形状约定: [num_layers, total_slots, num_kv_heads, head_dim]
        # 其中 total_slots = total_blocks * block_size

        # 1. 计算 Q 并做 Norm + RoPE
        q = self.q_proj(hidden_states).view(
            num_tokens, self.num_heads, self.head_dim
        )
        q = self.q_norm(q)
        q = self.rotary_emb(q, positions)

        # 2. 如果不是 KV 共享层，计算 K, V 并写入 Paged KV Cache Pool
        if not self.is_kv_shared:
            k = self.k_proj(hidden_states).view(
                num_tokens, self.num_kv_heads, self.head_dim
            )
            v = (
                k.clone()
                if self.use_k_eq_v
                else self.v_proj(hidden_states).view(
                    num_tokens, self.num_kv_heads, self.head_dim
                )
            )

            k = self.k_norm(k)
            v = self.v_norm(v)
            k = self.rotary_emb(k, positions)

            # 写入显存池对应的物理槽位 (out_cache_loc)
            k_cache[self.layer_idx, forward_batch.out_cache_loc, : self.num_kv_heads] = k
            v_cache[self.layer_idx, forward_batch.out_cache_loc, : self.num_kv_heads] = v

        # 3. Paged Attention 计算 (目标层为 self.kv_source_layer)
        # 在生产级 SGLang 中此处调用 FlashInfer；在小型引擎原型中我们按序列还原物理块进行计算
        attn_output = self._paged_attention_reference(
            q,
            k_cache[self.kv_source_layer],
            v_cache[self.kv_source_layer],
            forward_batch,
        )

        attn_output = attn_output.reshape(num_tokens, -1)
        return self.o_proj(attn_output)

    def _paged_attention_reference(
        self,
        q: torch.Tensor,
        layer_k_cache: torch.Tensor,
        layer_v_cache: torch.Tensor,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        """基于 Block Table 读取 KV Cache 并支持局部滑动窗口的参考实现"""
        outputs = []
        token_offset = 0

        for i in range(batch.batch_size):
            ext_len = batch.extend_lens[i]
            seq_len = int(batch.seq_lens[i].item())
            q_i = q[token_offset : token_offset + ext_len]  # [ext_len, num_heads, head_dim]
            token_offset += ext_len

            # 从 block_tables 收集该序列所有已缓存 token 的物理槽位
            num_blocks = (seq_len + batch.block_size - 1) // batch.block_size
            blocks = batch.block_tables[i, :num_blocks]  # [num_blocks]
            slots = (
                blocks.unsqueeze(1) * batch.block_size
                + torch.arange(batch.block_size, device=q.device)
            ).reshape(-1)[:seq_len]

            # 若为局部注意力层且超出 sliding_window，只截取窗口内的 KV
            if not self.is_full_attention and seq_len > self.config.sliding_window:
                # Decode 阶段直接截断最早的KV；Prefill 阶段通过下方的掩码控制
                if ext_len == 1:
                    slots = slots[-self.config.sliding_window :]

            k_i = layer_k_cache[slots, : self.num_kv_heads]  # [kv_len, num_kv_heads, head_dim]
            v_i = layer_v_cache[slots, : self.num_kv_heads]

            # GQA 广播: [kv_len, num_heads, head_dim]
            k_i = torch.repeat_interleave(k_i, self.num_queries_per_kv, dim=1)
            v_i = torch.repeat_interleave(v_i, self.num_queries_per_kv, dim=1)

            # 转换为 SDPA 需要的格式: [1, num_heads, q_len, head_dim]
            q_sdpa = q_i.transpose(0, 1).unsqueeze(0)
            k_sdpa = k_i.transpose(0, 1).unsqueeze(0)
            v_sdpa = v_i.transpose(0, 1).unsqueeze(0)

            # 构造 Attention Mask (处理 Prefix Cache 命中时的因果掩码 + 滑动窗口掩码)
            attn_mask = None
            if ext_len > 1:
                kv_len = k_i.shape[0]
                q_pos = torch.arange(seq_len - ext_len, seq_len, device=q.device).unsqueeze(1)
                k_pos = torch.arange(kv_len, device=q.device).unsqueeze(0)
                mask_cond = q_pos >= k_pos
                if not self.is_full_attention:
                    mask_cond = mask_cond & (
                        (q_pos - k_pos) < self.config.sliding_window
                    )
                attn_mask = torch.zeros(
                    (ext_len, kv_len), dtype=q.dtype, device=q.device
                ).masked_fill(~mask_cond, float("-inf"))

            out_i = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, scale=self.scaling
            )
            outputs.append(out_i.squeeze(0).transpose(0, 1))

        return torch.cat(outputs, dim=0)


# ============================================================================
# 4. MLP 与 DecoderLayer
# ============================================================================


class Gemma4MLP(nn.Module):
    """GeGLU 前馈网络，支持在 KV 共享层自动加倍宽度"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


class Gemma4DecoderLayer(nn.Module):

    def __init__(
        self, config: Gemma4Config, layer_idx: int, kv_source_layer: int
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Gemma4Attention(config, layer_idx, kv_source_layer)

        # 判断是否启用 Double-Wide MLP (Gemma 4 在 KV 共享层将 FFN 宽度翻倍以补偿参数量)
        first_kv_shared_idx = (
            config.num_hidden_layers - config.num_kv_shared_layers
        )
        is_kv_shared = (
            config.num_kv_shared_layers > 0 and layer_idx >= first_kv_shared_idx
        )
        mlp_intermediate = config.intermediate_size * (
            2 if (config.use_double_wide_mlp and is_kv_shared) else 1
        )
        self.mlp = Gemma4MLP(config.hidden_size, mlp_intermediate)

        # Gemma 标志性的 4 个 RMSNorm (Pre/Post Attention + Pre/Post FFN)
        self.input_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Per-Layer Input (PLE) 门控投影 (E2B / E4B 特有)
        self.has_ple = config.hidden_size_per_layer_input > 0
        if self.has_ple:
            self.per_layer_input_gate = nn.Linear(
                config.hidden_size,
                config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_projection = nn.Linear(
                config.hidden_size_per_layer_input,
                config.hidden_size,
                bias=False,
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        # Gemma 4 特有的无条件层输出缩放因子
        self.layer_scalar = nn.Parameter(torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_pool: Tuple[torch.Tensor, torch.Tensor],
        forward_batch: ForwardBatch,
        per_layer_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Self-Attention 块
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(normed, positions, kv_pool, forward_batch)
        hidden_states = residual + self.post_attention_layernorm(attn_out)

        # 2. MLP 块
        residual = hidden_states
        normed = self.pre_feedforward_layernorm(hidden_states)
        mlp_out = self.mlp(normed)
        hidden_states = residual + self.post_feedforward_layernorm(mlp_out)

        # 3. Per-Layer Embedding (PLE) 注入块
        if self.has_ple and per_layer_input is not None:
            residual = hidden_states
            gate = F.gelu(
                self.per_layer_input_gate(hidden_states), approximate="tanh"
            )
            ple_out = self.per_layer_projection(gate * per_layer_input)
            hidden_states = residual + self.post_per_layer_input_norm(ple_out)

        # 4. 乘上层缩放系数
        return hidden_states * self.layer_scalar


# ============================================================================
# 5. 顶层模型：Gemma4ForCausalLM
# ============================================================================


class Gemma4ForCausalLM(nn.Module):

    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.embed_scale = math.sqrt(config.hidden_size)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Per-Layer Embeddings (PLE) 组件初始化
        self.has_ple = config.hidden_size_per_layer_input > 0
        if self.has_ple:
            ple_total_dim = (
                config.num_hidden_layers * config.hidden_size_per_layer_input
            )
            self.embed_tokens_per_layer = nn.Embedding(
                config.vocab_size, ple_total_dim
            )
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size, ple_total_dim, bias=False
            )
            self.per_layer_projection_norm = Gemma4RMSNorm(
                config.hidden_size_per_layer_input, eps=config.rms_norm_eps
            )

        # 预计算每层的 KV 源索引 (YOCO: 后 num_kv_shared_layers 层绑定同类型的最后一个非共享层)
        kv_source_map = self._build_kv_sharing_map(config)

        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(config, i, kv_source_map[i])
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # 权重绑定 (Tie Word Embeddings)
        self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _build_kv_sharing_map(config: Gemma4Config) -> List[int]:
        """为 KV 共享层匹配同类型（sliding 或 full）的最后一个独立 KV 层"""
        first_shared = config.num_hidden_layers - config.num_kv_shared_layers
        last_independent_by_type = {}
        mapping = []
        for i in range(config.num_hidden_layers):
            l_type = config.layer_types[i]
            if i < first_shared or config.num_kv_shared_layers == 0:
                last_independent_by_type[l_type] = i
                mapping.append(i)
            else:
                mapping.append(last_independent_by_type.get(l_type, i))
        return mapping

    def _compute_per_layer_inputs(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        """计算 Gemma 4 E4B 的逐层辅助输入: [num_tokens, num_layers, ple_dim]"""
        num_tokens = input_ids.shape[0]
        L = self.config.num_hidden_layers
        D = self.config.hidden_size_per_layer_input

        tok_ple = self.embed_tokens_per_layer(input_ids) * math.sqrt(D)
        tok_ple = tok_ple.view(num_tokens, L, D)

        proj_ple = self.per_layer_model_projection(inputs_embeds) * (
            self.config.hidden_size**-0.5
        )
        proj_ple = self.per_layer_projection_norm(
            proj_ple.view(num_tokens, L, D)
        )

        return (tok_ple + proj_ple) * (2.0**-0.5)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,  # [num_tokens] 拍平的一维输入
        positions: torch.Tensor,  # [num_tokens] 每个 token 的绝对位置
        kv_pool: Tuple[torch.Tensor, torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # 1. 主干词嵌入并缩放 sqrt(hidden_size)
        hidden_states = self.embed_tokens(input_ids) * self.embed_scale

        # 2. 计算 Per-Layer Embeddings
        per_layer_inputs = (
            self._compute_per_layer_inputs(input_ids, hidden_states)
            if self.has_ple
            else None
        )

        # 3. 逐层前向传播
        for i, layer in enumerate(self.layers):
            ple_i = (
                per_layer_inputs[:, i, :]
                if per_layer_inputs is not None
                else None
            )
            hidden_states = layer(
                hidden_states, positions, kv_pool, forward_batch, ple_i
            )

        # 4. 只提取每个序列最后一个 token 的 hidden_state 来计算 Logits (节省显存与算力)
        last_token_indices = (
            torch.cumsum(
                torch.tensor(
                    forward_batch.extend_lens, device=input_ids.device
                ),
                dim=0,
            )
            - 1
        )
        last_hidden = self.norm(hidden_states[last_token_indices])

        logits = self.lm_head(last_hidden)  # [batch_size, vocab_size]

        # 5. Gemma 标志性的最终 Logit Softcapping
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap

        return logits


# ============================================================================
# 6. 最小化单步运行验证 (Smoke Test)
# ============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # 使用缩小版配置快速验证流水线
    mini_cfg = Gemma4Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_global_key_value_heads=2,
        head_dim=32,
        global_head_dim=32,
        num_kv_shared_layers=2,  # 最后2层共享前4层的KV
        hidden_size_per_layer_input=64,
    )

    model = Gemma4ForCausalLM(mini_cfg).to(device=device, dtype=dtype)

    # 分配 Paged KV Cache Pool: [num_layers, total_blocks * block_size, max_kv_heads, max_head_dim]
    block_size = 16
    total_blocks = 32
    k_pool = torch.zeros(
        (
            mini_cfg.num_hidden_layers,
            total_blocks * block_size,
            mini_cfg.num_key_value_heads,
            mini_cfg.head_dim,
        ),
        device=device,
        dtype=dtype,
    )
    v_pool = torch.zeros_like(k_pool)

    # 模拟 2 个并发请求的 Prefill (长度分别为 5 和 3，总共 8 个 token 拍平)
    input_ids = torch.randint(0, 1000, (8,), device=device)
    positions = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], device=device)

    batch_meta = ForwardBatch(
        batch_size=2,
        seq_lens=torch.tensor([5, 3], device=device),
        prefix_lens=torch.tensor([0, 0], device=device),
        extend_lens=[5, 3],
        # 请求0使用 Block 0 (槽位0~4)，请求1使用 Block 1 (槽位16~18)
        out_cache_loc=torch.tensor([0, 1, 2, 3, 4, 16, 17, 18], device=device),
        block_tables=torch.tensor([[0, 0], [1, 0]], device=device),
        block_size=block_size,
        is_prefill=True,
    )

    logits = model(input_ids, positions, (k_pool, v_pool), batch_meta)
    print(f"Prefill 成功！输出 Logits 形状: {logits.shape}")  # 预期: [2, 1000]