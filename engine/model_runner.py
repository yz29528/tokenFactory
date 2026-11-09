from typing import List, Optional
import torch

from qwen3_8 import (
    ForwardBatch,
    Qwen3_8Config,
    Qwen3_8ForCausalLM,
    Qwen3_8HybridCachePool,
)
from req_to_state_pool import ReqToStatePool
from .request import Request


class ModelRunner:
    """
    对应 SGLang 中的 `ModelRunner` (位于 Scheduler 进程内，独占管理 GPU 模型权重与物理缓存张量)：
    1. 持有 `Qwen3_8ForCausalLM` 模型实例与 `Qwen3_8HybridCachePool` 混合显存池（内部集成双区 `ReqToStatePool`）。
    2. 支持 Chunked Prefill 与 Mixed Batching：将同一批次内的 Decode 请求 (`extend_len=1`)
       与 Prefill 分块请求 (`extend_len=chunk_size`) 统一拍平为一维无 Padding 张量。
    3. 执行模型前向推理 (`forward`) 与 Token 采样 (`sample`)，返回生成的 `next_token_ids`。
    """

    def __init__(
        self,
        config: Qwen3_8Config,
        max_reqs: int,
        max_context_len: int,
        max_total_tokens: int,
        max_snapshots: int = 16,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        state_pool: Optional[ReqToStatePool] = None,
    ):
        self.config = config
        self.max_reqs = max_reqs
        self.max_context_len = max_context_len
        self.max_total_tokens = max_total_tokens
        self.max_snapshots = max_snapshots
        self.device = device
        self.dtype = dtype

        self.model = Qwen3_8ForCausalLM(config).to(device=device, dtype=dtype)
        self.model.eval()
        self.model.requires_grad_(False)

        self.cache_pool = Qwen3_8HybridCachePool(
            config=config,
            max_reqs=max_reqs,
            max_total_tokens=max_total_tokens,
            max_snapshots=max_snapshots,
            device=device,
            dtype=dtype,
            state_pool=state_pool,
        )

    @property
    def state_pool(self) -> ReqToStatePool:
        """返回底层 48 层线性注意力的 ReqToStatePool（含活跃工作区与 RadixCache 快照区）。"""
        return self.cache_pool.state_pool

    @torch.inference_mode()
    def reset_req_linear_state(self, req_idx: int) -> None:
        """清空已完成或新准入请求在 48 层 Gated DeltaNet 线性注意力中的工作区状态。"""
        self.cache_pool.reset_req_linear_state(req_idx)

    @torch.inference_mode()
    def restore_req_linear_state(self, snapshot_slot: int, req_idx: int) -> None:
        """将 RadixCache 状态锚点的快照 (`snapshot_slot`) 深拷贝恢复至请求工作区 (`req_idx`)。"""
        self.cache_pool.restore_req_snapshot(snapshot_slot, req_idx)

    @torch.inference_mode()
    def execute_batch(
        self,
        batch_reqs: List[Request],
        req_to_token_table: List[List[int]],
    ) -> List[int]:
        """
        将一批任意混合了 Decode 与 Chunked Prefill 的 Request 拍平为一维张量，
        构造 `ForwardBatch` 执行前向计算并采样出每个请求对应位置的输出 Token ID。
        """
        flat_input_ids: List[int] = []
        flat_positions: List[int] = []
        flat_out_cache_loc: List[int] = []
        extend_lens: List[int] = []
        req_pool_indices: List[int] = []
        seq_lens: List[int] = []
        prefix_lens: List[int] = []

        for req in batch_reqs:
            assert req.req_pool_idx is not None
            ext_len = len(req.cur_chunk_tokens)
            assert ext_len > 0 and ext_len == len(req.out_cache_loc)

            req_pool_indices.append(req.req_pool_idx)
            seq_lens.append(req.cur_seq_len)
            prefix_lens.append(req.cur_start_pos)
            extend_lens.append(ext_len)

            flat_input_ids.extend(req.cur_chunk_tokens)
            flat_positions.extend(range(req.cur_start_pos, req.cur_seq_len))
            flat_out_cache_loc.extend(req.out_cache_loc)

        input_ids_tensor = torch.tensor(
            flat_input_ids, dtype=torch.long, device=self.device
        )
        positions_tensor = torch.tensor(
            flat_positions, dtype=torch.long, device=self.device
        )
        req_to_token_tensor = torch.tensor(
            req_to_token_table, dtype=torch.long, device=self.device
        )

        forward_batch = ForwardBatch(
            batch_size=len(batch_reqs),
            req_pool_indices=torch.tensor(
                req_pool_indices, dtype=torch.long, device=self.device
            ),
            seq_lens=torch.tensor(seq_lens, dtype=torch.long, device=self.device),
            prefix_lens=torch.tensor(prefix_lens, dtype=torch.long, device=self.device),
            extend_lens=extend_lens,
            out_cache_loc=torch.tensor(
                flat_out_cache_loc, dtype=torch.long, device=self.device
            ),
            req_to_token=req_to_token_tensor,
            is_prefill=any(not r.is_prefilled for r in batch_reqs),
        )

        # 1. 模型前向计算: [batch_size, vocab_size]
        logits = self.model(
            input_ids=input_ids_tensor,
            positions=positions_tensor,
            cache_pool=self.cache_pool,
            forward_batch=forward_batch,
        )

        # 2. 逐请求采样
        return self._sample(logits, batch_reqs)

    def _sample(self, logits: torch.Tensor, batch_reqs: List[Request]) -> List[int]:
        next_tokens: List[int] = []
        for i, req in enumerate(batch_reqs):
            req_logits = logits[i]
            if req.temperature <= 1e-5:
                token_id = int(torch.argmax(req_logits, dim=-1).item())
            else:
                probs = torch.softmax(req_logits / req.temperature, dim=-1)
                token_id = int(torch.multinomial(probs, num_samples=1).item())
            next_tokens.append(token_id)
        return next_tokens
