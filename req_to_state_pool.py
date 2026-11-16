from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from qwen3_8.config import Qwen3_8Config


def _inference_mode():
    if torch is not None:
        return torch.inference_mode()
    return lambda fn: fn


class ReqToStatePool:
    """
    面向混合注意力架构 (Qwen3.8-27B 48 层 Gated DeltaNet) 的线性注意力状态池 (`ReqToStatePool`)。
    采用“双区隔离”设计，在同一个连续物理状态张量中管理两类槽位：
    1. 活跃工作区 (Working Zone, 槽位 `[0, max_reqs)`):
       - 与 `ReqToTokenPool` 中的 `req_pool_idx` 一一对应。
       - 供正在执行的请求在 Chunked Prefill 与 Decode 阶段原地更新 (In-place Mutation) `conv_state` 与 `ssm_state`。
    2. 树节点快照区 (Snapshot Zone, 槽位 `[max_reqs, max_reqs + max_snapshots)`):
       - 由 `free_snapshot_slots` 空闲栈管理，专供 `RadixCache` 关键节点挂载只读状态快照 (`linear_state_slot`)。
       - 前缀命中时通过 `restore_snapshot(snapshot_slot, req_idx)` 深拷贝 (Fork) 到请求工作区；
       - 节点归档时通过 `capture_snapshot(req_idx, snapshot_slot)` 从请求工作区保存快照；
       - 快照耗尽时支持由 `RadixCache.evict_linear_state()` 独立按 LRU 回收，无需驱逐 Full Attention KV。
    """

    def __init__(
        self,
        max_reqs: int,
        max_snapshots: int = 16,
        config: Optional[Qwen3_8Config] = None,
        device: str = "cpu",
        dtype: Any = None,
    ):
        self.max_reqs = max_reqs
        self.max_snapshots = max_snapshots
        self.total_slots = max_reqs + max_snapshots
        self.config = config
        self.device = device
        self.dtype = (
            dtype
            if dtype is not None
            else (torch.bfloat16 if torch is not None else None)
        )

        # 快照区槽位号范围: [max_reqs, max_reqs + max_snapshots)
        self.free_snapshot_slots: List[int] = list(
            range(self.total_slots - 1, self.max_reqs - 1, -1)
        )
        # 显式跟踪每个快照槽位是否已被占用（防止 Double-Free）
        self.is_snapshot_occupied: Dict[int, bool] = {
            slot: False for slot in range(self.max_reqs, self.total_slots)
        }

        # 模拟器/调试标签记录：记录每个槽位当前对应的状态摘要
        self.state_debug_tags: List[Optional[str]] = [None] * self.total_slots

        # 若提供了 Qwen3_8Config，则分配真实的 48 层 Gated DeltaNet 物理状态张量
        if config is not None:
            if torch is None:
                raise RuntimeError("分配真实 Qwen3_8 线性状态张量需要安装 PyTorch (torch)！")
            self.linear_layer_map: Dict[int, int] = {
                layer_idx: i
                for i, layer_idx in enumerate(config.linear_attention_layer_indices)
            }
            num_linear_layers = len(self.linear_layer_map)
            tp_size = config.tp_size
            local_num_k_heads = config.linear_num_key_heads // tp_size
            local_num_v_heads = config.linear_num_value_heads // tp_size
            conv_dim = (
                2 * local_num_k_heads * config.linear_key_head_dim
                + local_num_v_heads * config.linear_value_head_dim
            )
            self.conv_state: Optional[torch.Tensor] = torch.zeros(
                (
                    num_linear_layers,
                    self.total_slots,
                    conv_dim,
                    config.linear_conv_kernel_dim - 1,
                ),
                device=device,
                dtype=self.dtype,
            )
            ssm_dtype = (
                torch.float32 if config.mamba_ssm_dtype == "float32" else self.dtype
            )
            self.ssm_state: Optional[torch.Tensor] = torch.zeros(
                (
                    num_linear_layers,
                    self.total_slots,
                    local_num_v_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                ),
                device=device,
                dtype=ssm_dtype,
            )
        else:
            self.linear_layer_map = {}
            self.conv_state = None
            self.ssm_state = None

    def available_snapshot_size(self) -> int:
        """返回当前剩余可用的线性状态快照槽位数量。"""
        return len(self.free_snapshot_slots)

    def get_layer_states(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定 Linear Attention 层的 (conv_state, ssm_state) 状态张量。"""
        assert self.conv_state is not None and self.ssm_state is not None
        pool_idx = self.linear_layer_map[layer_idx]
        return self.conv_state[pool_idx], self.ssm_state[pool_idx]

    @_inference_mode()
    def reset_working_state(self, req_idx: int) -> None:
        """将指定活跃请求工作区 (`req_idx`) 的线性注意力状态清零。"""
        assert 0 <= req_idx < self.max_reqs, f"非法的工作区槽位索引: {req_idx}"
        if self.conv_state is not None and self.ssm_state is not None:
            self.conv_state[:, req_idx].zero_()
            self.ssm_state[:, req_idx].zero_()
        self.state_debug_tags[req_idx] = "ZERO_STATE"

    def alloc_snapshot(self) -> Optional[int]:
        """从快照池申请 1 个空闲快照槽位 (`slot >= max_reqs`)。"""
        if not self.free_snapshot_slots:
            return None
        slot = self.free_snapshot_slots.pop()
        assert not self.is_snapshot_occupied[slot], (
            f"线性快照槽位 {slot} 状态异常：分配前已被标记为占用！"
        )
        self.is_snapshot_occupied[slot] = True
        return slot

    @_inference_mode()
    def free_snapshot(self, slot: int) -> None:
        """释放单个线性注意力快照槽位并归还至空闲栈。"""
        assert self.max_reqs <= slot < self.total_slots, (
            f"槽位 {slot} 不属于快照区范围 [{self.max_reqs}, {self.total_slots})！"
        )
        assert self.is_snapshot_occupied[slot], (
            f"检测到 Double-Free：线性快照槽位 {slot} 当前并未被占用！"
        )
        self.is_snapshot_occupied[slot] = False
        if self.conv_state is not None and self.ssm_state is not None:
            self.conv_state[:, slot].zero_()
            self.ssm_state[:, slot].zero_()
        self.state_debug_tags[slot] = None
        self.free_snapshot_slots.append(slot)

    def free_snapshots(self, slots: List[int]) -> None:
        """批量释放多个线性注意力快照槽位。"""
        for s in slots:
            self.free_snapshot(s)

    @_inference_mode()
    def capture_snapshot(
        self, req_idx: int, snapshot_slot: int, debug_tag: Optional[str] = None
    ) -> None:
        """
        将活跃请求工作区 `req_idx` 当前的 48 层 `conv_state` 与 `ssm_state`
        深拷贝（保存快照）到树节点绑定的只读快照槽位 `snapshot_slot`。
        """
        assert 0 <= req_idx < self.max_reqs
        assert self.max_reqs <= snapshot_slot < self.total_slots
        assert self.is_snapshot_occupied[snapshot_slot], (
            f"快照槽位 {snapshot_slot} 未分配，无法写入快照！"
        )
        if self.conv_state is not None and self.ssm_state is not None:
            self.conv_state[:, snapshot_slot].copy_(self.conv_state[:, req_idx])
            self.ssm_state[:, snapshot_slot].copy_(self.ssm_state[:, req_idx])
        self.state_debug_tags[snapshot_slot] = (
            debug_tag or self.state_debug_tags[req_idx] or f"Snap(from_req_{req_idx})"
        )

    @_inference_mode()
    def restore_snapshot(self, snapshot_slot: int, req_idx: int) -> None:
        """
        当新请求命中 RadixCache 前缀的状态锚点时，将树节点的只读快照 `snapshot_slot`
        深拷贝（Fork）回该请求的活跃工作区 `req_idx`，避免原地更新污染树缓存。
        """
        assert 0 <= req_idx < self.max_reqs
        assert self.max_reqs <= snapshot_slot < self.total_slots
        assert self.is_snapshot_occupied[snapshot_slot], (
            f"尝试从已释放的快照槽位 {snapshot_slot} 恢复线性状态！"
        )
        if self.conv_state is not None and self.ssm_state is not None:
            self.conv_state[:, req_idx].copy_(self.conv_state[:, snapshot_slot])
            self.ssm_state[:, req_idx].copy_(self.ssm_state[:, snapshot_slot])
        self.state_debug_tags[req_idx] = self.state_debug_tags[snapshot_slot]
