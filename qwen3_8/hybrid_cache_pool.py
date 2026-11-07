from typing import Dict, Tuple
import torch

from .config import Qwen3_8Config


class Qwen3_8HybridCachePool:
    """
    Qwen 3.8 27B 专用的混合显存池 (Hybrid Cache Pool)：
    1. 针对 16 层 `full_attention`：
       按 SGLang 的 `TokenToKVPool` 管理一维物理槽位缓冲 `[num_full_layers, max_total_tokens, num_kv_heads, head_dim]`。
    2. 针对 48 层 `linear_attention` (Gated DeltaNet)：
       按请求行 (`req_pool_idx`) 管理因果卷积状态 `conv_state` 与循环状态矩阵 `ssm_state` (float32)。
    """

    def __init__(
        self,
        config: Qwen3_8Config,
        max_reqs: int,
        max_total_tokens: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.max_reqs = max_reqs
        self.max_total_tokens = max_total_tokens
        self.device = device
        self.dtype = dtype

        # 建立层号 (0..63) 到各自紧凑存储下标的映射
        self.full_layer_map: Dict[int, int] = {
            layer_idx: i
            for i, layer_idx in enumerate(config.full_attention_layer_indices)
        }
        self.linear_layer_map: Dict[int, int] = {
            layer_idx: i
            for i, layer_idx in enumerate(config.linear_attention_layer_indices)
        }

        num_full_layers = len(self.full_layer_map)
        num_linear_layers = len(self.linear_layer_map)

        # 1. Full Attention 的 TokenToKVPool 物理缓冲区
        self.k_buffer = torch.zeros(
            (
                num_full_layers,
                max_total_tokens,
                config.num_key_value_heads,
                config.head_dim,
            ),
            device=device,
            dtype=dtype,
        )
        self.v_buffer = torch.zeros_like(self.k_buffer)

        # 2. Linear Attention (Gated DeltaNet) 的状态缓冲区
        # QKV 通道总维度: Q(16*128) + K(16*128) + V(48*128) = 2048 + 2048 + 6144 = 10240
        conv_dim = (
            2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
        )
        self.conv_state = torch.zeros(
            (
                num_linear_layers,
                max_reqs,
                conv_dim,
                config.linear_conv_kernel_dim - 1,
            ),
            device=device,
            dtype=dtype,
        )

        # SSM / DeltaNet 递归状态根据配置固定使用 float32 (`mamba_ssm_dtype: "float32"`)
        ssm_dtype = (
            torch.float32 if config.mamba_ssm_dtype == "float32" else dtype
        )
        self.ssm_state = torch.zeros(
            (
                num_linear_layers,
                max_reqs,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
            ),
            device=device,
            dtype=ssm_dtype,
        )

    def get_full_kv_buffers(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定 Full Attention 层的 K/V 物理池张量。"""
        pool_idx = self.full_layer_map[layer_idx]
        return self.k_buffer[pool_idx], self.v_buffer[pool_idx]

    def get_linear_states(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定 Linear Attention 层的 (conv_state, ssm_state) 状态张量。"""
        pool_idx = self.linear_layer_map[layer_idx]
        return self.conv_state[pool_idx], self.ssm_state[pool_idx]

    @torch.inference_mode()
    def reset_req_linear_state(self, req_idx: int) -> None:
        """当请求释放或初始化时，清零该请求对应的 Linear Attention 状态。"""
        self.conv_state[:, req_idx].zero_()
        self.ssm_state[:, req_idx].zero_()
