import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3_8Config
from .forward_batch import ForwardBatch
from .fp8_linear import ColumnParallelFP8Linear, RowParallelFP8Linear
from .hybrid_cache_pool import Qwen3_8HybridCachePool
from .rms_norm_gated import Qwen3_8RMSNormGated


class Qwen3_8LinearAttention(nn.Module):
    r"""
    Qwen 3.8 27B 的线性注意力层 (`linear_attn` - Gated DeltaNet，共 48 层)。
    采用 Megatron-LM 风格的【先列并行 (Column-Parallel) -> 中间本地递推 (0 通信) -> 后行并行 (Row-Parallel) + 单次 AllReduce】架构：
    1. 前段（列并行）：
       - `in_proj_qkv`: 分组列并行 (`partition_sizes=[global_key_dim, global_key_dim, global_value_dim]`)，
         每个 TP Rank 切取配对的 `[Q_r, K_r, V_r]`（维持本地 3:1 GVA 比例）。
       - `in_proj_z`: 列并行，切取本卡的 `local_value_dim`。
       - `in_proj_b`, `in_proj_a`: 沿 `num_v_heads` 切取本卡的 `local_num_v_heads`。
    2. 中段（各卡本地独立计算，0 跨卡通信）：
       - `conv1d`: 深度可分离因果一维卷积 (`groups = local_conv_dim`)。
       - `_gated_delta_rule_step`: 各卡独立更新本地 `local_num_v_heads` 个状态矩阵 $S_t \in \mathbb{R}^{128 \times 128}$。
       - `norm`: `Qwen3_8RMSNormGated(head_dim = head_v_dim)`，按每个 Value Head 独立做门控 RMSNorm，无需跨卡同步方差。
    3. 后段（行并行 + 1 次 AllReduce）：
       - `out_proj`: `RowParallelFP8Linear`，沿输入维度 `local_value_dim` 行切分，输出前执行全块唯一一次 `AllReduce`。
    """

    def __init__(self, config: Qwen3_8Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.tp_size = config.tp_size
        self.tp_rank = config.tp_rank

        assert config.linear_num_key_heads % self.tp_size == 0, (
            f"linear_num_key_heads={config.linear_num_key_heads} 无法被 tp_size={self.tp_size} 整除！"
        )
        assert config.linear_num_value_heads % self.tp_size == 0, (
            f"linear_num_value_heads={config.linear_num_value_heads} 无法被 tp_size={self.tp_size} 整除！"
        )

        self.hidden_size = config.hidden_size
        self.global_num_k_heads = config.linear_num_key_heads      # 16
        self.head_k_dim = config.linear_key_head_dim               # 128
        self.global_num_v_heads = config.linear_num_value_heads    # 48
        self.head_v_dim = config.linear_value_head_dim             # 128

        # 当前 TP Rank 本地分片头数与通道维
        self.num_k_heads = self.global_num_k_heads // self.tp_size
        self.num_v_heads = self.global_num_v_heads // self.tp_size
        self.num_v_per_k = self.num_v_heads // self.num_k_heads    # 3

        self.global_key_dim = self.global_num_k_heads * self.head_k_dim      # 2048
        self.global_value_dim = self.global_num_v_heads * self.head_v_dim    # 6144
        self.global_conv_dim = self.global_key_dim * 2 + self.global_value_dim  # 10240

        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim  # 4

        # 1. 前段：列并行输入投影 (in_proj_qkv 按 [Q, K, V] 分组列切分，in_proj_z 连续列切分)
        self.in_proj_qkv = ColumnParallelFP8Linear(
            in_features=self.hidden_size,
            global_out_features=self.global_conv_dim,
            bias=False,
            config=config,
            partition_sizes=[
                self.global_key_dim,
                self.global_key_dim,
                self.global_value_dim,
            ],
        )
        self.in_proj_z = ColumnParallelFP8Linear(
            in_features=self.hidden_size,
            global_out_features=self.global_value_dim,
            bias=False,
            config=config,
        )
        # in_proj_b 与 in_proj_a 保持高精度，按本卡 local_num_v_heads 列切分
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # 2. 中段：各卡本地深度可分离一维因果卷积 + 状态衰减参数 + 按 Head 独立的门控 RMSNorm (0 通信)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.norm = Qwen3_8RMSNormGated(
            self.value_dim, eps=config.rms_norm_eps, head_dim=self.head_v_dim
        )

        # 3. 后段：行并行输出投影 + 单次 AllReduce
        self.out_proj = RowParallelFP8Linear(
            global_in_features=self.global_value_dim,
            out_features=self.hidden_size,
            bias=False,
            config=config,
        )

    @torch.no_grad()
    def load_tp_from_full_layer(self, full_layer: "Qwen3_8LinearAttention") -> None:
        """从单卡完整 `Qwen3_8LinearAttention` 切片加载当前 `tp_rank` 的权重。"""
        self.in_proj_qkv.load_from_full_linear(full_layer.in_proj_qkv)
        self.in_proj_z.load_from_full_linear(full_layer.in_proj_z)
        self.out_proj.load_from_full_linear(full_layer.out_proj)

        # 按 [Q, K, V] 分组切分深度可分离卷积权重 conv1d.weight
        conv_w = self.in_proj_qkv.slice_full_tensor(
            full_layer.conv1d.weight, is_scale=False
        )
        self.conv1d.weight.copy_(conv_w)

        # 按 local_num_v_heads 切分 in_proj_b, in_proj_a, A_log, dt_bias 与 norm.weight
        v_start = self.tp_rank * self.num_v_heads
        v_end = v_start + self.num_v_heads
        self.in_proj_b.weight.copy_(full_layer.in_proj_b.weight[v_start:v_end])
        self.in_proj_a.weight.copy_(full_layer.in_proj_a.weight[v_start:v_end])
        self.A_log.copy_(full_layer.A_log[v_start:v_end])
        self.dt_bias.copy_(full_layer.dt_bias[v_start:v_end])

        dim_start = self.tp_rank * self.value_dim
        dim_end = dim_start + self.value_dim
        self.norm.weight.copy_(full_layer.norm.weight[dim_start:dim_end])

    def _causal_conv1d_step(
        self,
        qkv_seq: torch.Tensor,     # [ext_len, conv_dim]
        req_conv_state: torch.Tensor,  # [conv_dim, kernel_size - 1]
    ) -> torch.Tensor:
        """
        对单个序列执行带状态缓存的因果 Conv1d + SiLU 激活，并原地更新 `req_conv_state`。
        """
        qkv_t = qkv_seq.transpose(0, 1)  # [conv_dim, ext_len]
        padded = torch.cat([req_conv_state, qkv_t], dim=-1)
        req_conv_state.copy_(padded[:, -(self.conv_kernel_size - 1) :])

        conv_out = F.conv1d(
            padded.unsqueeze(0),
            self.conv1d.weight,
            bias=self.conv1d.bias,
            groups=self.conv_dim,
        ).squeeze(0)

        return F.silu(conv_out).transpose(0, 1)

    def _gated_delta_rule_step(
        self,
        q: torch.Tensor,          # [ext_len, num_k_heads, head_k_dim]
        k: torch.Tensor,          # [ext_len, num_k_heads, head_k_dim]
        v: torch.Tensor,          # [ext_len, num_v_heads, head_v_dim]
        b: torch.Tensor,          # [ext_len, num_v_heads]
        a: torch.Tensor,          # [ext_len, num_v_heads]
        req_ssm_state: torch.Tensor,  # [num_v_heads, head_k_dim, head_v_dim] (float32)
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """
        执行 Gated DeltaNet 状态更新（在各卡本地 Head 分片上独立计算，0 通信）：
        - 单步 Decode (`ext_len == 1`)：直接执行融合单步矩阵更新。
        - 多步 Prefill (`ext_len > 1`)：采用基于 WY 表示的分块并行三角求解算法 (Chunk-wise Parallel Gated DeltaNet)，
          消除逐 Token Python 循环，将分块内递推转化为并行矩阵乘与 `solve_triangular`。
        """
        orig_dtype = q.dtype
        q_fp32 = F.normalize(q.float(), p=2.0, dim=-1) * (self.head_k_dim**-0.5)
        k_fp32 = F.normalize(k.float(), p=2.0, dim=-1)

        q_exp = torch.repeat_interleave(q_fp32, self.num_v_per_k, dim=1)
        k_exp = torch.repeat_interleave(k_fp32, self.num_v_per_k, dim=1)
        v_fp32 = v.float()

        decay_rate = -torch.exp(self.A_log.float()) * F.softplus(
            a.float() + self.dt_bias.float()
        )
        beta = torch.sigmoid(b.float())

        seq_len = q.shape[0]
        if seq_len == 1:
            alpha_0 = torch.exp(decay_rate[0]).unsqueeze(-1).unsqueeze(-1)
            beta_0 = beta[0].unsqueeze(-1)
            req_ssm_state.mul_(alpha_0)
            v_pred = torch.einsum("hkd,hk->hd", req_ssm_state, k_exp[0])
            delta = beta_0 * (v_fp32[0] - v_pred)
            req_ssm_state.add_(torch.einsum("hk,hd->hkd", k_exp[0], delta))
            o_0 = torch.einsum("hkd,hk->hd", req_ssm_state, q_exp[0])
            return o_0.unsqueeze(0).to(orig_dtype)

        chunk_outputs = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            c_len = end - start

            q_h = q_exp[start:end].transpose(0, 1)      # [H, C, K]
            k_h = k_exp[start:end].transpose(0, 1)      # [H, C, K]
            v_h = v_fp32[start:end].transpose(0, 1)     # [H, C, V]
            g_h = decay_rate[start:end].transpose(0, 1) # [H, C]
            beta_h = beta[start:end].transpose(0, 1)    # [H, C]

            gamma = torch.cumsum(g_h, dim=-1)           # [H, C]
            diff = gamma.unsqueeze(-1) - gamma.unsqueeze(-2)  # [H, C, C]
            tril_mask = torch.tril(
                torch.ones((c_len, c_len), dtype=torch.bool, device=q.device)
            )
            strict_tril_mask = torch.tril(
                torch.ones((c_len, c_len), dtype=torch.bool, device=q.device),
                diagonal=-1,
            )

            gamma_tril = torch.exp(diff.masked_fill(~tril_mask, float("-inf")))
            gamma_strict_tril = gamma_tril.masked_fill(~strict_tril_mask, 0.0)

            kk = torch.matmul(k_h, k_h.transpose(-1, -2))  # [H, C, C]
            eye = torch.eye(c_len, dtype=torch.float32, device=q.device).unsqueeze(0)
            lhs = eye + beta_h.unsqueeze(-1) * gamma_strict_tril * kk

            exp_gamma = torch.exp(gamma).unsqueeze(-1)     # [H, C, 1]
            k_state = torch.matmul(k_h, req_ssm_state)     # [H, C, V]
            rhs = beta_h.unsqueeze(-1) * (v_h - exp_gamma * k_state)

            u = torch.linalg.solve_triangular(
                lhs, rhs, upper=False, unitriangular=True
            )  # [H, C, V]

            q_state = torch.matmul(q_h, req_ssm_state)     # [H, C, V]
            qk = torch.matmul(q_h, k_h.transpose(-1, -2))  # [H, C, C]
            out_h = exp_gamma * q_state + torch.matmul(gamma_tril * qk, u)
            chunk_outputs.append(out_h.transpose(0, 1))

            total_decay = torch.exp(gamma[:, -1]).view(-1, 1, 1)
            to_end_decay = torch.exp(gamma[:, -1:] - gamma).unsqueeze(-1)
            k_decayed = k_h * to_end_decay
            req_ssm_state.mul_(total_decay).add_(
                torch.matmul(k_decayed.transpose(-1, -2), u)
            )

        return torch.cat(chunk_outputs, dim=0).to(orig_dtype)

    def _batched_decode_step(
        self,
        qkv: torch.Tensor,            # [B, conv_dim]
        b: torch.Tensor,              # [B, num_v_heads]
        a: torch.Tensor,              # [B, num_v_heads]
        layer_conv_state: torch.Tensor,
        layer_ssm_state: torch.Tensor,
        req_indices: torch.Tensor,    # [B]
    ) -> torch.Tensor:
        """
        针对纯 Decode 批次 (所有请求 `ext_len == 1`) 的全批次融合向量化更新：
        消除按 `batch_size` 的 Python 循环，一次性完成全部请求的因果卷积与 Gated DeltaNet 单步递推。
        """
        bsz = req_indices.shape[0]
        conv_states = layer_conv_state[req_indices]  # [B, conv_dim, kernel_size - 1]
        padded = torch.cat([conv_states, qkv.unsqueeze(-1)], dim=-1)
        layer_conv_state[req_indices] = padded[:, :, -(self.conv_kernel_size - 1) :]

        conv_out = F.silu(
            F.conv1d(
                padded,
                self.conv1d.weight,
                bias=self.conv1d.bias,
                groups=self.conv_dim,
            ).squeeze(-1)
        )  # [B, conv_dim]

        q, k, v = conv_out.split(
            [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        q = q.view(bsz, self.num_k_heads, self.head_k_dim)
        k = k.view(bsz, self.num_k_heads, self.head_k_dim)
        v = v.view(bsz, self.num_v_heads, self.head_v_dim)

        orig_dtype = q.dtype
        q_fp32 = F.normalize(q.float(), p=2.0, dim=-1) * (self.head_k_dim**-0.5)
        k_fp32 = F.normalize(k.float(), p=2.0, dim=-1)
        q_exp = torch.repeat_interleave(q_fp32, self.num_v_per_k, dim=1)
        k_exp = torch.repeat_interleave(k_fp32, self.num_v_per_k, dim=1)
        v_fp32 = v.float()

        decay_rate = -torch.exp(self.A_log.float()) * F.softplus(
            a.float() + self.dt_bias.float()
        )
        alpha = torch.exp(decay_rate).unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        beta = torch.sigmoid(b.float()).unsqueeze(-1)              # [B, H, 1]

        states = layer_ssm_state[req_indices] * alpha              # [B, H, K, V]
        v_pred = torch.einsum("bhkd,bhk->bhd", states, k_exp)
        delta = beta * (v_fp32 - v_pred)
        states = states + torch.einsum("bhk,bhd->bhkd", k_exp, delta)
        layer_ssm_state[req_indices] = states

        out = torch.einsum("bhkd,bhk->bhd", states, q_exp)
        return out.reshape(bsz, self.value_dim).to(orig_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_pool: Qwen3_8HybridCachePool,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # 1. 列并行投影得到本卡本地分片的 QKV, Z, b, a (0 通信)
        qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # 2. 从混合池中获取本卡本层的卷积状态与 SSM 状态池分片
        layer_conv_state, layer_ssm_state = cache_pool.get_linear_states(self.layer_idx)

        # 3. 若整个批次均为单步 Decode，走全批次无循环向量化快速路径
        if all(ext_len == 1 for ext_len in forward_batch.extend_lens):
            attn_out = self._batched_decode_step(
                qkv=qkv,
                b=b,
                a=a,
                layer_conv_state=layer_conv_state,
                layer_ssm_state=layer_ssm_state,
                req_indices=forward_batch.req_pool_indices,
            )
            attn_out = self.norm(attn_out, z)
            return self.out_proj(attn_out)

        # 4. 变长 Chunked Prefill / Mixed 批次：使用分块并行 WY 算法更新状态
        seq_outputs = []
        token_offset = 0

        req_indices_list = (
            forward_batch.req_pool_indices_list
            if forward_batch.req_pool_indices_list is not None
            else [int(x) for x in forward_batch.req_pool_indices.tolist()]
        )
        for i in range(forward_batch.batch_size):
            req_idx = req_indices_list[i]
            ext_len = forward_batch.extend_lens[i]

            qkv_i = qkv[token_offset : token_offset + ext_len]
            b_i = b[token_offset : token_offset + ext_len]
            a_i = a[token_offset : token_offset + ext_len]
            token_offset += ext_len

            conv_out_i = self._causal_conv1d_step(qkv_i, layer_conv_state[req_idx])
            q_i, k_i, v_i = conv_out_i.split(
                [self.key_dim, self.key_dim, self.value_dim], dim=-1
            )

            q_i = q_i.view(ext_len, self.num_k_heads, self.head_k_dim)
            k_i = k_i.view(ext_len, self.num_k_heads, self.head_k_dim)
            v_i = v_i.view(ext_len, self.num_v_heads, self.head_v_dim)

            out_i = self._gated_delta_rule_step(
                q_i, k_i, v_i, b_i, a_i, layer_ssm_state[req_idx]
            )
            seq_outputs.append(out_i.reshape(ext_len, self.value_dim))

        attn_out = torch.cat(seq_outputs, dim=0)

        # 5. 按 Head 独立的门控 RMSNorm (0 通信) + 行并行 out_proj (内部执行单次 AllReduce)
        attn_out = self.norm(attn_out, z)
        return self.out_proj(attn_out)
