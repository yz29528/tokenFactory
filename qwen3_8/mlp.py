import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3_8Config


class Qwen3_8MLP(nn.Module):
    """
    Qwen 3.8 27B 的 SwiGLU 前馈网络 (`mlp`)：
    - `hidden_size = 5120`
    - `intermediate_size = 17408`
    - `hidden_act = "silu"`
    """

    def __init__(self, config: Qwen3_8Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
