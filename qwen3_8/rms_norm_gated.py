from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3_8RMSNormGated(nn.Module):
    """
    Qwen 3.8 27B 线性注意力层 (`linear_attn.norm`) 专用的门控 RMSNorm。
    支持按每个 Value Head (`head_dim = 128`) 独立计算均方根归一化，
    从而在张量并行（先列并行、后行并行）下保证各卡本地 Head 归一化结果与单卡完全等价（中间 0 跨卡通信）。
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        head_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.head_dim = head_dim
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()

        if self.head_dim is not None and x_fp32.shape[-1] % self.head_dim == 0:
            orig_shape = x_fp32.shape
            x_heads = x_fp32.view(*orig_shape[:-1], -1, self.head_dim)
            variance = x_heads.pow(2).mean(dim=-1, keepdim=True)
            x_normed = (x_heads * torch.rsqrt(variance + self.eps)).view(orig_shape)
        else:
            variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_fp32 * torch.rsqrt(variance + self.eps)

        out = self.weight.float() * x_normed * F.silu(gate.float())
        return out.to(orig_dtype)
