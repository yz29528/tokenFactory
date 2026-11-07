import torch
import torch.nn as nn


class Qwen3_8RMSNorm(nn.Module):
    """
    Qwen 3.8 27B 标准 RMSNorm 组件。
    用于：
    - 每层的 `input_layernorm` 和 `post_attention_layernorm` (dim=5120)
    - Full Attention 的 `q_norm` 和 `k_norm` (dim=256)
    - 顶层 `model.norm` 及 MTP 模块的归一化层 (`eps=1e-6`)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x_normed).to(orig_dtype)
