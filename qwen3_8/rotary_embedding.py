from typing import List, Optional
import torch
import torch.nn as nn


class PartialRotaryEmbedding(nn.Module):
    """
    Qwen 3.8 27B 的 Partial RoPE / Interleaved M-RoPE 实现：
    - `head_dim = 256`, `partial_rotary_factor = 0.25` -> 仅对前 `rotary_dim = 64` 维施加旋转位置编码，
      后 `192` 维保持直通不变。
    - `rope_theta = 10,000,000`
    - 支持一维位置索引 `[num_tokens]` 以及多模态三维位置索引 `[3, num_tokens]` (`mrope_section = [11, 11, 10]`)。
    """

    def __init__(
        self,
        head_dim: int = 256,
        partial_rotary_factor: float = 0.25,
        rope_theta: float = 10000000.0,
        mrope_section: Optional[List[int]] = None,
        mrope_interleaved: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.rope_theta = rope_theta
        self.mrope_section = mrope_section or [11, 11, 10]
        self.mrope_interleaved = mrope_interleaved

        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def _compute_cos_sin(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 若输入为一维文本位置 [num_tokens]
        if positions.ndim == 1:
            freqs = torch.outer(positions.float(), self.inv_freq)  # [num_tokens, rotary_dim // 2]
        else:
            # 三维 M-RoPE 位置 [3, num_tokens]，按 mrope_section [11, 11, 10] (共 32 对频率 = 64 维) 组合
            freqs_3d = torch.einsum("cn,d->cnd", positions.float(), self.inv_freq)
            if self.mrope_interleaved:
                # 交错式 M-RoPE
                freqs = freqs_3d[0].clone()
                for dim_idx in (1, 2):
                    limit = self.mrope_section[dim_idx] * 3
                    freqs[:, dim_idx:limit:3] = freqs_3d[dim_idx, :, dim_idx:limit:3]
            else:
                chunks = [
                    chunk[idx]
                    for idx, chunk in enumerate(
                        freqs_3d.split(self.mrope_section, dim=-1)
                    )
                ]
                freqs = torch.cat(chunks, dim=-1)

        emb = torch.cat((freqs, freqs), dim=-1)  # [num_tokens, rotary_dim]
        cos = emb.cos().unsqueeze(1).to(dtype)   # [num_tokens, 1, rotary_dim]
        sin = emb.sin().unsqueeze(1).to(dtype)
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对 q `[num_tokens, num_q_heads, head_dim]` 和 k `[num_tokens, num_kv_heads, head_dim]`
        的前 `self.rotary_dim` (64) 维执行旋转编码，其余 192 维原样拼接返回。
        """
        cos, sin = self._compute_cos_sin(positions, q.dtype)

        q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
        k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]

        q_rot = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (self._rotate_half(k_rot) * sin)

        q_out = torch.cat((q_rot, q_pass), dim=-1)
        k_out = torch.cat((k_rot, k_pass), dim=-1)
        return q_out, k_out
