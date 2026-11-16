from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class ForwardBatch:
    """
    SGLang 风格的批次元数据 (ForwardBatch)，衔接两级显存池：
    - `req_pool_indices`: [batch_size] 当前批次每个请求在第一层池 (ReqToTokenPool / RecurrentStatePool) 中的行号
    - `seq_lens`: [batch_size] 每个请求截至当前 Step 的总序列长度
    - `prefix_lens`: [batch_size] 命中的 RadixCache 前缀长度
    - `extend_lens`: List[int] 本次 Step 每个请求实际参与前向计算的 token 数量 (Decode 阶段全为 1)
    - `out_cache_loc`: [num_tokens] 本次计算的每个 token 在第二层池 (TokenToKVPool) 中的目标物理槽位索引
    - `req_to_token`: [max_reqs, max_context_len] 第一层池的二维映射表张量 (行: req_idx, 列: token_pos -> 值: kv_slot_idx)
    - `is_prefill`: 当前是否为 Prefill / Extend 阶段
    """

    batch_size: int
    req_pool_indices: torch.Tensor  # [batch_size]
    seq_lens: torch.Tensor          # [batch_size]
    prefix_lens: torch.Tensor       # [batch_size]
    extend_lens: List[int]          # 长度为 batch_size，总和等于本次输入总 token 数 num_tokens
    out_cache_loc: torch.Tensor     # [num_tokens]
    req_to_token: torch.Tensor      # [max_reqs, max_context_len]
    is_prefill: bool = True
    # 以下为 CPU 侧预计算缓存字段，避免前向传播过程中调用 .item() 触发 GPU-CPU 同步
    max_seq_len: int = 0
    max_extend_len: int = 0
    extend_lens_tensor: Optional[torch.Tensor] = None
    req_pool_indices_list: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if self.max_extend_len == 0 and self.extend_lens:
            self.max_extend_len = max(self.extend_lens)
        if self.extend_lens_tensor is None and self.extend_lens:
            self.extend_lens_tensor = torch.tensor(
                self.extend_lens, dtype=torch.long, device=self.out_cache_loc.device
            )
        if self.max_seq_len == 0 and self.seq_lens.numel() > 0:
            self.max_seq_len = int(self.seq_lens.max().item())
        if self.req_pool_indices_list is None and self.req_pool_indices.numel() > 0:
            self.req_pool_indices_list = [
                int(x) for x in self.req_pool_indices.tolist()
            ]
