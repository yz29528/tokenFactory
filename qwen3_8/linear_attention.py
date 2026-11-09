import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3_8Config
from .forward_batch import ForwardBatch
from .hybrid_cache_pool import Qwen3_8HybridCachePool
from .rms_norm_gated import Qwen3_8RMSNormGated


class Qwen3_8LinearAttention(nn.Module):
    r"""
    Qwen 3.8 27B 的线性注意力层 (`linear_attn` - Gated DeltaNet，共 48 层)。
    严格对齐 `qwen3.8 27B/config.json` 中的子模块命名与超参数：
    - `linear_num_key_heads = 16`, `linear_key_head_dim = 128` (Q/K 总维度各 2048)
    - `linear_num_value_heads = 48`, `linear_value_head_dim = 128` (V/Z 总维度各 6144，GVA 比例 3:1)
    - `linear_conv_kernel_dim = 4` (作用于拼接后的 QKV 10240 维的深度可分离因果一维卷积 `conv1d`)
    - `in_proj_qkvz`: 投影出 Q, K, V 与输出门控 Z
    - `in_proj_ba` (或 `in_proj_b`, `in_proj_a`): 投影出写入率控制项 b 与衰减门控项 a
    - `A_log`, `dt_bias`: 控制状态衰减率 $\alpha_t = \exp(-\exp(A\_log) \cdot \text{softplus}(a_t + dt\_bias))$
    - `norm`: `Qwen3_8RMSNormGated`，结合门控 Z 对输出做 RMSNorm
    - `out_proj`: 将 6144 维映射回 `hidden_size = 5120`
    """

    def __init__(self, config: Qwen3_8Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads      # 16
        self.head_k_dim = config.linear_key_head_dim        # 128
        self.num_v_heads = config.linear_num_value_heads    # 48
        self.head_v_dim = config.linear_value_head_dim      # 128
        self.num_v_per_k = self.num_v_heads // self.num_k_heads  # 3

        self.key_dim = self.num_k_heads * self.head_k_dim    # 2048
        self.value_dim = self.num_v_heads * self.head_v_dim  # 6144
        self.conv_dim = self.key_dim * 2 + self.value_dim    # 10240
        self.conv_kernel_size = config.linear_conv_kernel_dim  # 4

        # 输入线性投影 (与 Qwen3.5/3.8 权重结构对齐)
        self.in_proj_qkvz = nn.Linear(
            self.hidden_size, self.conv_dim + self.value_dim, bias=False
        )
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # 深度可分离一维因果卷积 (groups = conv_dim)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )

        # Gated DeltaNet 状态衰减参数
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        # 门控归一化与输出投影
        self.norm = Qwen3_8RMSNormGated(self.value_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def _causal_conv1d_step(
        self,
        qkv_seq: torch.Tensor,     # [ext_len, conv_dim]
        req_conv_state: torch.Tensor,  # [conv_dim, kernel_size - 1]
    ) -> torch.Tensor:
        """
        对单个序列执行带状态缓存的因果 Conv1d + SiLU 激活，并原地更新 `req_conv_state`。
        """
        ext_len = qkv_seq.shape[0]
        qkv_t = qkv_seq.transpose(0, 1)  # [conv_dim, ext_len]

        # 拼接历史卷积窗口状态: [conv_dim, (kernel_size - 1) + ext_len]
        padded = torch.cat([req_conv_state, qkv_t], dim=-1)

        # 更新该请求最新的 kernel_size - 1 个输入状态
        req_conv_state.copy_(padded[:, -(self.conv_kernel_size - 1) :])

        # 使用深度卷积权重计算输出: weight 形状为 [conv_dim, 1, kernel_size]
        conv_out = F.conv1d(
            padded.unsqueeze(0),
            self.conv1d.weight,
            bias=self.conv1d.bias,
            groups=self.conv_dim,
        ).squeeze(0)  # [conv_dim, ext_len]

        return F.silu(conv_out).transpose(0, 1)  # [ext_len, conv_dim]

    def _gated_delta_rule_step(
        self,
        q: torch.Tensor,          # [ext_len, num_k_heads, head_k_dim]
        k: torch.Tensor,          # [ext_len, num_k_heads, head_k_dim]
        v: torch.Tensor,          # [ext_len, num_v_heads, head_v_dim]
        b: torch.Tensor,          # [ext_len, num_v_heads]
        a: torch.Tensor,          # [ext_len, num_v_heads]
        req_ssm_state: torch.Tensor,  # [num_v_heads, head_k_dim, head_v_dim] (float32)
    ) -> torch.Tensor:
        """
        执行 Gated DeltaNet 递归状态更新（严格按照 `mamba_ssm_dtype = "float32"` 在 FP32 下维护状态）：
        S_t = alpha_t * S_{t-1} + k_t * (beta_t * (v_t - S_{t-1}^T k_t))^T
        o_t = S_t^T q_t
        """
        orig_dtype = q.dtype
        # L2 归一化 Q, K 并按 3:1 扩展到与 V 相同的 48 个 Head
        q_fp32 = F.normalize(q.float(), p=2.0, dim=-1) * (self.head_k_dim**-0.5)
        k_fp32 = F.normalize(k.float(), p=2.0, dim=-1)

        q_exp = torch.repeat_interleave(q_fp32, self.num_v_per_k, dim=1)  # [ext_len, 48, 128]
        k_exp = torch.repeat_interleave(k_fp32, self.num_v_per_k, dim=1)  # [ext_len, 48, 128]
        v_fp32 = v.float()

        # 计算衰减因子 alpha_t 与写入门控 beta_t
        decay_rate = -torch.exp(self.A_log.float()) * F.softplus(
            a.float() + self.dt_bias.float()
        )
        alpha = torch.exp(decay_rate)  # [ext_len, 48]
        beta = torch.sigmoid(b.float())  # [ext_len, 48]

        state = req_ssm_state  # [48, 128, 128]
        outputs = []

        for t in range(q.shape[0]):
            q_t = q_exp[t]      # [48, 128]
            k_t = k_exp[t]      # [48, 128]
            v_t = v_fp32[t]     # [48, 128]
            alpha_t = alpha[t].unsqueeze(-1).unsqueeze(-1)  # [48, 1, 1]
            beta_t = beta[t].unsqueeze(-1)                  # [48, 1]

            # 1. 状态指数衰减
            state.mul_(alpha_t)

            # 2. Delta Rule 误差修正: v_pred = S^T @ k_t
            v_pred = torch.einsum("hkd,hk->hd", state, k_t)
            delta = beta_t * (v_t - v_pred)  # [48, 128]

            # 3. 秩-1 状态写入: S <- S + k_t (x) delta
            state.add_(torch.einsum("hk,hd->hkd", k_t, delta))

            # 4. 读取当前步输出: o_t = S^T @ q_t
            o_t = torch.einsum("hkd,hk->hd", state, q_t)
            outputs.append(o_t)

        return torch.stack(outputs, dim=0).to(orig_dtype)  # [ext_len, 48, 128]

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_pool: Qwen3_8HybridCachePool,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]

        # 1. 投影得到 QKV + Z，以及门控项 b, a
        qkvz = self.in_proj_qkvz(hidden_states)
        qkv, z = qkvz.split([self.conv_dim, self.value_dim], dim=-1)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        # 2. 从混合池中获取本层的卷积状态与 SSM 状态池
        layer_conv_state, layer_ssm_state = cache_pool.get_linear_states(self.layer_idx)

        # 3. 按批次中的每个请求更新因果卷积与 Gated DeltaNet 循环状态
        seq_outputs = []
        token_offset = 0

        for i in range(forward_batch.batch_size):
            req_idx = int(forward_batch.req_pool_indices[i].item())
            ext_len = forward_batch.extend_lens[i]

            qkv_i = qkv[token_offset : token_offset + ext_len]
            b_i = b[token_offset : token_offset + ext_len]
            a_i = a[token_offset : token_offset + ext_len]
            token_offset += ext_len

            # 3.1 因果一维卷积 + SiLU
            conv_out_i = self._causal_conv1d_step(qkv_i, layer_conv_state[req_idx])
            q_i, k_i, v_i = conv_out_i.split(
                [self.key_dim, self.key_dim, self.value_dim], dim=-1
            )

            q_i = q_i.view(ext_len, self.num_k_heads, self.head_k_dim)
            k_i = k_i.view(ext_len, self.num_k_heads, self.head_k_dim)
            v_i = v_i.view(ext_len, self.num_v_heads, self.head_v_dim)

            # 3.2 Gated DeltaNet 状态更新与输出计算
            out_i = self._gated_delta_rule_step(
                q_i, k_i, v_i, b_i, a_i, layer_ssm_state[req_idx]
            )
            seq_outputs.append(out_i.reshape(ext_len, self.value_dim))

        attn_out = torch.cat(seq_outputs, dim=0)  # [num_tokens, value_dim]

        # 4. 门控 RMSNorm + 输出投影
        attn_out = self.norm(attn_out, z)
        return self.out_proj(attn_out)
