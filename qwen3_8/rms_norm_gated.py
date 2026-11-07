import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3_8RMSNormGated(nn.Module):
    """
    Qwen 3.8 27B 线性注意力层 (`linear_attn.norm`) 专用的门控 RMSNorm。
    先利用输出门控分支 `z` (经 SiLU/Swish 激活) 与线性注意力输出相乘或门控归一化。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        # 先在 fp32 下完成 RMSNorm，再乘以 SiLU(gate)
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        out = self.weight.float() * x_normed * F.silu(gate.float())
        return out.to(orig_dtype)
