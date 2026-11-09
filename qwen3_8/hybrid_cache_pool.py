from typing import Dict, Optional, Tuple
import torch

from req_to_state_pool import ReqToStatePool
from .config import Qwen3_8Config


class Qwen3_8HybridCachePool:
    """
    Qwen 3.8 27B 专用的混合显存池 (Hybrid Cache Pool)：
    1. 针对 16 层 `full_attention` (Softmax Attention)：
       按 SGLang 的 `TokenToKVPool` 管理一维物理槽位缓冲 `[num_full_layers, max_total_tokens, num_kv_heads, head_dim]`。
    2. 针对 48 层 `linear_attention` (Gated DeltaNet)：
       集成双区隔离的 `ReqToStatePool`：
       - `[0, max_reqs)` 为活跃请求工作区 (`req_pool_idx`)，支持前向传播原地更新；
       - `[max_reqs, max_reqs + max_snapshots)` 为 `RadixCache` 树节点稀疏状态快照区 (`linear_state_slot`)。
    """

    def __init__(
        self,
        config: Qwen3_8Config,
        max_reqs: int,
        max_total_tokens: int,
        max_snapshots: int = 16,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        state_pool: Optional[ReqToStatePool] = None,
    ):
        self.config = config
        self.max_reqs = max_reqs
        self.max_total_tokens = max_total_tokens
        self.max_snapshots = max_snapshots
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

        # 2. Linear Attention (Gated DeltaNet) 的双区状态池 ReqToStatePool
        self.state_pool: ReqToStatePool = (
            state_pool
            if state_pool is not None
            else ReqToStatePool(
                max_reqs=max_reqs,
                max_snapshots=max_snapshots,
                config=config,
                device=device,
                dtype=dtype,
            )
        )

    @property
    def conv_state(self) -> torch.Tensor:
        assert self.state_pool.conv_state is not None
        return self.state_pool.conv_state

    @property
    def ssm_state(self) -> torch.Tensor:
        assert self.state_pool.ssm_state is not None
        return self.state_pool.ssm_state

    def get_full_kv_buffers(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定 Full Attention 层的 K/V 物理池张量。"""
        pool_idx = self.full_layer_map[layer_idx]
        return self.k_buffer[pool_idx], self.v_buffer[pool_idx]

    def get_linear_states(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定 Linear Attention 层的 (conv_state, ssm_state) 状态张量。"""
        return self.state_pool.get_layer_states(layer_idx)

    @torch.inference_mode()
    def reset_req_linear_state(self, req_idx: int) -> None:
        """当请求释放或初始化时，清零该请求对应的工作区 Linear Attention 状态。"""
        self.state_pool.reset_working_state(req_idx)

    @torch.inference_mode()
    def capture_req_snapshot(
        self, req_idx: int, snapshot_slot: int, debug_tag: Optional[str] = None
    ) -> None:
        """将请求工作区 `req_idx` 的线性状态保存至树节点快照槽位 `snapshot_slot`。"""
        self.state_pool.capture_snapshot(req_idx, snapshot_slot, debug_tag=debug_tag)

    @torch.inference_mode()
    def restore_req_snapshot(self, snapshot_slot: int, req_idx: int) -> None:
        """将树节点快照槽位 `snapshot_slot` 的线性状态恢复 (Fork) 至请求工作区 `req_idx`。"""
        self.state_pool.restore_snapshot(snapshot_slot, req_idx)
