import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3_8Config
from .fp8_linear import ColumnParallelFP8Linear, RowParallelFP8Linear


class Qwen3_8MLP(nn.Module):
    """
    Qwen 3.8 27B 的稠密 SwiGLU 前馈网络 (`mlp`)。
    采用 Megatron-LM 风格的【先列并行 (`gate_proj`, `up_proj`) -> 本地逐元素激活 (`SiLU * up`, 0 通信) -> 后行并行 (`down_proj`) + 单次 AllReduce】：
    - `gate_proj`, `up_proj`: `ColumnParallelFP8Linear`（沿 `intermediate_size = 17408` 切分为 `17408 // tp_size`）
    - `down_proj`: `RowParallelFP8Linear`（沿输入维 `17408 // tp_size` 行切分，末尾执行唯一一次 `AllReduce`）
    """

    def __init__(self, config: Qwen3_8Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.global_intermediate_size = config.intermediate_size
        self.tp_size = config.tp_size
        self.tp_rank = config.tp_rank
        self.intermediate_size = self.global_intermediate_size // self.tp_size

        # 1. 前段：列并行 gate_proj 与 up_proj (0 通信)
        self.gate_proj = ColumnParallelFP8Linear(
            in_features=self.hidden_size,
            global_out_features=self.global_intermediate_size,
            bias=False,
            config=config,
        )
        self.up_proj = ColumnParallelFP8Linear(
            in_features=self.hidden_size,
            global_out_features=self.global_intermediate_size,
            bias=False,
            config=config,
        )
        # 2. 后段：行并行 down_proj + 单次 AllReduce
        self.down_proj = RowParallelFP8Linear(
            global_in_features=self.global_intermediate_size,
            out_features=self.hidden_size,
            bias=False,
            config=config,
        )

    @torch.no_grad()
    def load_tp_from_full_mlp(self, full_mlp: "Qwen3_8MLP") -> None:
        """从单卡完整 `Qwen3_8MLP` 切片加载当前 `tp_rank` 的权重。"""
        self.gate_proj.load_from_full_linear(full_mlp.gate_proj)
        self.up_proj.load_from_full_linear(full_mlp.up_proj)
        self.down_proj.load_from_full_linear(full_mlp.down_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
