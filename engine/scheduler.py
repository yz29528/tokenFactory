import multiprocessing as mp
import queue
import time
from typing import List, Optional
import torch

from qwen3_8 import Qwen3_8Config
from radix_tree import RadixCache
from req_to_state_pool import ReqToStatePool
from req_to_token_pool import ReqToTokenPool
from token_to_kv_pool import TokenToKVPool
from .ipc_messages import BatchTokenIDOut, ShutdownSignal, TokenizedReqInput
from .model_runner import ModelRunner
from .request import Request


class Scheduler:
    """
    进程 2 (Scheduler 进程)：支持混合注意力 (`TokenToKVPool` + `ReqToStatePool`)、
    **Chunked Prefill (分块预填充)** 与 **Mixed Batching (Prefill+Decode 混合批处理)** 的 Continuous Batching 调度器。
    核心机制：
    1. 双池协同管理：
       - `TokenToKVPool`: 管理 16 层 Softmax Full Attention 的逐 Token KV 槽位。
       - `ReqToStatePool`: 管理 48 层 Gated DeltaNet 线性注意力的活跃请求工作区 (`[0, max_reqs)`) 与树节点稀疏快照区 (`[max_reqs, max_reqs + max_snapshots)`)。
    2. 混合前缀匹配与状态锚点回退 (`match_prefix_hybrid`)：
       - 新请求准入时，在 `RadixCache` 中同时获取最长 Softmax KV 前缀 (`prefix_len_at_start`) 与最深有效线性状态锚点 (`state_anchor_len`)。
       - 将状态锚点的 `linear_state_slot` 深拷贝 (Fork) 到请求工作区，并从 `state_anchor_len` 开始推进 `prefilled_len`。
       - 若 `state_anchor_len < prefix_len_at_start`（例如某节点线性快照曾被 LRU 单独回收，但保留了 Softmax KV 槽位），重算该区间时直接复用树上已有 KV 槽位，零重复分配并在到达终点时完成状态自愈！
    3. 稀疏快照与非对称驱逐：
       - 仅在 Prompt 预填充完成点 (`is_prefill_complete`) 与请求结束点 (`_retire_request`) 挂载线性状态快照，中间分块不消耗快照槽位。
       - 快照池满时由 `RadixCache.evict_linear_state()` 按 LRU 仅剥离 `linear_state_slot`（置为 `None`）；Softmax KV 叶子被驱逐时同步销毁其挂载的线性状态。
    """

    def __init__(
        self,
        config: Qwen3_8Config,
        max_reqs: int = 8,
        max_context_len: int = 128,
        max_total_tokens: int = 256,
        max_snapshots: int = 16,
        chunked_prefill_size: int = 4,
        device: str = "cpu",
        enable_mtp: bool = False,
    ):
        self.config = config
        self.max_reqs = max_reqs
        self.max_context_len = max_context_len
        self.max_total_tokens = max_total_tokens
        self.max_snapshots = max_snapshots
        self.chunked_prefill_size = chunked_prefill_size
        self.min_prefill_budget = 1
        self.device = device
        self.enable_mtp = enable_mtp

        # 初始化第一层映射池与 Softmax KV 物理池
        self.req_to_token_pool = ReqToTokenPool(max_reqs, max_context_len)
        self.token_to_kv_pool = TokenToKVPool(
            max_total_tokens, allocate_mock_buffers=False
        )

        # 初始化底层算子执行器 ModelRunner（内含绑定了真实张量的 ReqToStatePool）
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model_runner = ModelRunner(
            config=config,
            max_reqs=max_reqs,
            max_context_len=max_context_len,
            max_total_tokens=max_total_tokens,
            max_snapshots=max_snapshots,
            device=device,
            dtype=dtype,
            enable_mtp=enable_mtp,
        )
        self.req_to_state_pool: ReqToStatePool = self.model_runner.state_pool

        # 初始化同时绑定 TokenToKVPool 与 ReqToStatePool 的混合 RadixCache
        self.radix_cache = RadixCache(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_state_pool=self.req_to_state_pool,
            on_kv_free=self.model_runner.free_mtp_slots,
        )

        # 三个状态队列：
        # 1. waiting_queue: 尚未分配 req_pool_idx 的新请求
        # 2. inflight_prefill_queue: 已分配 req_pool_idx、正在分块执行 Chunked Prefill 的请求
        # 3. running_batch: 已完成全部 Prefill、正在逐 Token Decode 的请求
        self.waiting_queue: List[Request] = []
        self.inflight_prefill_queue: List[Request] = []
        self.running_batch: List[Request] = []
        self.step_counter: int = 0
        self._pending_vocab_updates: dict[int, str] = {}

    def _alloc_kv_slots(self, num_tokens: int) -> Optional[List[int]]:
        """申请物理 Softmax KV 槽位；若不足则触发 RadixCache LRU 驱逐（含同步清理被驱逐叶子的线性快照）。"""
        if self.token_to_kv_pool.available_size() < num_tokens:
            need = num_tokens - self.token_to_kv_pool.available_size()
            evicted = self.radix_cache.evict(need)
            if evicted:
                self.model_runner.free_mtp_slots(evicted)
        return self.token_to_kv_pool.alloc(num_tokens)

    def recv_requests(self, in_queue: mp.Queue) -> bool:
        """非阻塞地从 TokenizerManager 拉取所有新请求放入 `waiting_queue`。"""
        while True:
            try:
                msg = in_queue.get_nowait()
            except queue.Empty:
                break

            if isinstance(msg, ShutdownSignal):
                return False

            if isinstance(msg, TokenizedReqInput):
                if msg.vocab_updates:
                    self._pending_vocab_updates.update(msg.vocab_updates)
                if not msg.prompt_ids:
                    continue
                prompt_ids = msg.prompt_ids
                if len(prompt_ids) >= self.max_context_len:
                    prompt_ids = prompt_ids[: self.max_context_len - 1]
                max_allowed_new = max(1, self.max_context_len - len(prompt_ids))
                clamped_new_tokens = max(1, min(msg.max_new_tokens, max_allowed_new))
                req = Request(
                    req_id=msg.req_id,
                    prompt_text=msg.prompt_text,
                    prompt_ids=prompt_ids,
                    max_new_tokens=clamped_new_tokens,
                    eos_token_id=(
                        msg.eos_token_id
                        if msg.eos_token_id is not None
                        else self.config.eos_token_id
                    ),
                    temperature=msg.temperature,
                )
                self.waiting_queue.append(req)
        return True

    def _estimate_inflight_demands(self, budget_limit: int) -> tuple[int, int]:
        """评估当前已在 `inflight_prefill_queue` 中的请求在本步需要消耗的 Token 预算与新 KV 槽位数。"""
        used_budget = 0
        used_kv = 0
        rem = budget_limit
        for req in self.inflight_prefill_queue:
            if rem <= 0:
                break
            chunk_len = min(req.remaining_prefill_len, rem)
            start_pos = req.prefilled_len
            if start_pos < req.prefix_len_at_start < start_pos + chunk_len:
                chunk_len = req.prefix_len_at_start - start_pos
            end_pos = start_pos + chunk_len
            alloc_start = max(start_pos, req.prefix_len_at_start)
            need_new_kv = max(0, end_pos - alloc_start)
            used_budget += chunk_len
            used_kv += need_new_kv
            rem -= chunk_len
        return used_budget, used_kv

    def _estimate_inflight_budget_demand(self, budget_limit: int) -> int:
        """评估当前已在 `inflight_prefill_queue` 中的请求在本步需要消耗的 Token 预算。"""
        used_budget, _ = self._estimate_inflight_demands(budget_limit)
        return used_budget

    def _admit_waiting_requests(
        self,
        budget_limit: Optional[int] = None,
        reserved_kv: int = 0,
    ) -> None:
        """
        将 `waiting_queue` 中的新请求按当前剩余 Token 预算与可用 KV 槽位准入到 `inflight_prefill_queue`：
        1. 评估当前剩余预算 `budget_limit` 及扣除在途请求需求后的可用物理 KV 槽位，避免过度准入锁死请求槽位或树节点。
        2. 使用 `match_prefix_hybrid(req.prompt_ids[:-1])` 查找最长可复用 Softmax KV 前缀及最深有效线性状态锚点（预留至少最后 1 个 Prompt Token 用于前向产出采样 Logits）。
        3. 锁定 `last_kv_node` 防止整条路径上的 KV 槽位被驱逐，并将 `matched_kv_slots` 写入 `ReqToTokenPool`。
        4. 若存在有效 `linear_state_slot`，将其从快照区深拷贝 (Fork) 到请求工作区 `req_idx`；否则清零工作区。
        5. 将 `req.prefilled_len` 设为 `state_anchor_len`（确保 48 层线性注意力与 16 层全局注意力状态严格对齐）。
        """
        remaining: List[Request] = []
        avail_budget = budget_limit
        reserved_new_kv = reserved_kv
        for req in self.waiting_queue:
            if avail_budget is not None and avail_budget <= 0:
                remaining.append(req)
                continue
            if not self.req_to_token_pool.free_req_slots:
                remaining.append(req)
                continue

            # 仅对前 len(prompt_ids) - 1 个 token 做前缀匹配，确保至少保留最后 1 个 token 参与前向计算并产出首个采样 token
            searchable_prefix = req.prompt_ids[:-1] if len(req.prompt_ids) > 1 else []
            match_res = self.radix_cache.match_prefix_hybrid(searchable_prefix)

            kv_prefix_len = len(match_res.matched_kv_slots)
            state_anchor_len = match_res.state_anchor_len
            self.radix_cache.inc_lock_ref(match_res.last_kv_node)

            if avail_budget is not None:
                rem_prefill = len(req.prompt_ids) - state_anchor_len
                planned_chunk = min(rem_prefill, avail_budget)
                if (
                    state_anchor_len
                    < kv_prefix_len
                    < state_anchor_len + planned_chunk
                ):
                    planned_chunk = kv_prefix_len - state_anchor_len
                end_pos = state_anchor_len + planned_chunk
                alloc_start = max(state_anchor_len, kv_prefix_len)
                need_new_kv = max(0, end_pos - alloc_start)

                max_avail_kv = (
                    self.token_to_kv_pool.available_size()
                    + self.radix_cache.evictable_size_
                    - reserved_new_kv
                )
                if need_new_kv > max_avail_kv:
                    self.radix_cache.dec_lock_ref(match_res.last_kv_node)
                    remaining.append(req)
                    continue
                avail_budget -= planned_chunk
                reserved_new_kv += need_new_kv

            req_idx = self.req_to_token_pool.alloc_req()
            assert req_idx is not None

            if kv_prefix_len > 0:
                self.req_to_token_pool.write(req_idx, 0, match_res.matched_kv_slots)

            # 恢复线性注意力状态锚点快照（或重置为全零初始状态）
            if match_res.linear_state_slot is not None and state_anchor_len > 0:
                self.model_runner.restore_req_linear_state(
                    match_res.linear_state_slot, req_idx
                )
            else:
                self.model_runner.reset_req_linear_state(req_idx)

            req.req_pool_idx = req_idx
            req.prefix_len_at_start = kv_prefix_len
            req.state_anchor_len = state_anchor_len
            req.prefilled_len = state_anchor_len
            req.last_locked_node = match_res.last_kv_node

            self.inflight_prefill_queue.append(req)

        self.waiting_queue = remaining

    def _build_mixed_chunked_batch(self) -> tuple[List[Request], str]:
        """
        核心组批逻辑（Chunked Prefill + Decode 混合批处理）：
        1. 先为 `running_batch` 中每个活跃 Decode 请求分配 1 个 Token 槽位。
        2. 使用剩余预算从 `inflight_prefill_queue` 中切分 Prompt Chunk。
           若某请求处于 `[state_anchor_len, prefix_len_at_start)` 的线性状态追赶区间，
           直接复用 `req_to_token_pool` 中已绑定的树上 KV 槽位，仅对超出 `prefix_len_at_start` 的部分申请新 KV 槽位！
        """
        active_batch: List[Request] = []
        has_decode = False
        has_prefill = False

        # 1. 优先接纳所有正在 Decode 的请求（每个仅占 1 个 Token 预算）
        preempted_this_step = False
        idx = 0
        while idx < len(self.running_batch):
            req = self.running_batch[idx]
            assert req.req_pool_idx is not None
            can_verify_mtp = (
                self.enable_mtp
                and req.pending_draft_token is not None
                and req.seq_len + 1 <= self.max_context_len
                and len(req.output_ids) + 1 < req.max_new_tokens
            )
            new_slot = self._alloc_kv_slots(2) if can_verify_mtp else None
            if can_verify_mtp and new_slot is None:
                can_verify_mtp = False
                req.pending_draft_token = None

            if not can_verify_mtp:
                new_slot = self._alloc_kv_slots(1)

            while new_slot is None:
                if self.inflight_prefill_queue:
                    victim = self.inflight_prefill_queue.pop()
                    self._preempt_request(victim)
                    preempted_this_step = True
                    new_slot = self._alloc_kv_slots(1)
                elif len(self.running_batch) - 1 > idx or (
                    len(self.running_batch) - 1 == idx and active_batch
                ):
                    victim = self.running_batch.pop()
                    self._preempt_request(victim)
                    preempted_this_step = True
                    if victim is req:
                        break
                    new_slot = self._alloc_kv_slots(1)
                else:
                    break

            if new_slot is None:
                if req not in self.running_batch:
                    continue
                raise RuntimeError("KV Cache 物理池耗尽，无法为 Decode 请求分配槽位！")

            start_pos = req.seq_len - 1
            self.req_to_token_pool.write(req.req_pool_idx, start_pos, new_slot)

            if can_verify_mtp:
                assert req.pending_draft_token is not None
                req.cur_chunk_tokens = [req.output_ids[-1], req.pending_draft_token]
                req.cur_start_pos = start_pos
                req.cur_seq_len = req.seq_len + 1
                req.out_cache_loc = new_slot
                req.cur_is_mtp_verify = True
            else:
                req.cur_chunk_tokens = [req.output_ids[-1]]
                req.cur_start_pos = start_pos
                req.cur_seq_len = req.seq_len
                req.out_cache_loc = new_slot
                req.cur_is_mtp_verify = False
            active_batch.append(req)
            has_decode = True
            idx += 1

        if not preempted_this_step:
            remaining_budget = self.chunked_prefill_size - len(active_batch)
            if remaining_budget <= 0 and (
                self.inflight_prefill_queue or self.waiting_queue
            ):
                remaining_budget = self.min_prefill_budget

            # 2. 尝试将 waiting_queue 中的新请求按剩余可用预算准入到 inflight_prefill_queue
            if self.waiting_queue:
                inflight_demand, inflight_kv = self._estimate_inflight_demands(
                    remaining_budget
                )
                admit_budget = max(0, remaining_budget - inflight_demand)
                if admit_budget > 0:
                    self._admit_waiting_requests(
                        budget_limit=admit_budget,
                        reserved_kv=inflight_kv,
                    )

            # 3. 利用剩余 Token 预算对 inflight_prefill_queue 执行分块预填充 (Chunked Prefill)
            for req in list(self.inflight_prefill_queue):
                if remaining_budget <= 0:
                    break
                if req not in self.inflight_prefill_queue:
                    continue

                assert req.req_pool_idx is not None
                chunk_len = min(req.remaining_prefill_len, remaining_budget)
                start_pos = req.prefilled_len
                # 若当前处于 [state_anchor_len, prefix_len_at_start) 的线性状态追赶区间，
                # 在到达分支节点边界 prefix_len_at_start 处对齐切块，以便在该树节点上完成线性状态快照自愈！
                if start_pos < req.prefix_len_at_start < start_pos + chunk_len:
                    chunk_len = req.prefix_len_at_start - start_pos
                end_pos = start_pos + chunk_len

                # 仅对尚未拥有树上 KV 槽位的后缀部分申请新物理槽位
                alloc_start = max(start_pos, req.prefix_len_at_start)
                need_new_kv = max(0, end_pos - alloc_start)
                if need_new_kv > 0:
                    max_avail_kv = (
                        self.token_to_kv_pool.available_size()
                        + self.radix_cache.evictable_size_
                    )
                    if max_avail_kv == 0 and not active_batch:
                        # 当多个 inflight_prefill 请求互相持有 KV 导致无槽位推进时，从队尾抢占低优先级请求解除活锁
                        while (
                            self.token_to_kv_pool.available_size()
                            + self.radix_cache.evictable_size_
                            == 0
                            and len(self.inflight_prefill_queue) > 1
                        ):
                            victim = self.inflight_prefill_queue.pop()
                            if victim is req:
                                self.inflight_prefill_queue.append(victim)
                                break
                            self._preempt_request(victim)
                        max_avail_kv = (
                            self.token_to_kv_pool.available_size()
                            + self.radix_cache.evictable_size_
                        )
                        if max_avail_kv == 0:
                            raise RuntimeError(
                                "KV Cache 物理池容量不足，单个 Prefill 请求无法推进！"
                            )

                    if 0 < max_avail_kv < need_new_kv:
                        need_new_kv = max_avail_kv
                        end_pos = alloc_start + need_new_kv
                        chunk_len = end_pos - start_pos

                    new_slots = self._alloc_kv_slots(need_new_kv)
                    if new_slots is None:
                        break
                    self.req_to_token_pool.write(req.req_pool_idx, alloc_start, new_slots)

                req.cur_chunk_tokens = req.prompt_ids[start_pos:end_pos]
                req.cur_start_pos = start_pos
                req.cur_seq_len = end_pos
                req.out_cache_loc = self.req_to_token_pool.get_req_slots(
                    req.req_pool_idx, end_pos
                )[start_pos:end_pos]
                req.cur_is_mtp_verify = False
                req.prefilled_len = end_pos

                active_batch.append(req)
                remaining_budget -= chunk_len
                has_prefill = True

        if has_prefill and has_decode:
            batch_mode = "MIXED"
        elif has_prefill:
            batch_mode = "CHUNK_PREFILL"
        elif has_decode:
            batch_mode = "DECODE"
        else:
            batch_mode = "IDLE"

        return active_batch, batch_mode

    def step(self) -> Optional[BatchTokenIDOut]:
        """执行单步 Chunked Prefill + Decode 混合调度。"""
        active_batch, batch_mode = self._build_mixed_chunked_batch()
        if not active_batch:
            return None

        self.step_counter += 1

        # 调用 ModelRunner 一次性执行混合批次的前向计算与采样
        if self.enable_mtp:
            accepted_batches = self.model_runner.execute_batch_speculative(
                batch_reqs=active_batch,
                req_to_token_table=self.req_to_token_pool.req_to_token,
                dirty_from=self.req_to_token_pool.dirty_from,
                token_to_kv_pool=self.token_to_kv_pool,
                req_to_token_pool=self.req_to_token_pool,
            )
        else:
            sampled_token_ids = self.model_runner.execute_batch(
                batch_reqs=active_batch,
                req_to_token_table=self.req_to_token_pool.req_to_token,
                dirty_from=self.req_to_token_pool.dirty_from,
            )
            accepted_batches = [[tok] for tok in sampled_token_ids]

        out_msg = BatchTokenIDOut(
            step_id=self.step_counter,
            batch_mode=batch_mode,
            multi_token_ids=[] if self.enable_mtp else None,
        )
        if self._pending_vocab_updates:
            out_msg.vocab_updates = dict(self._pending_vocab_updates)
            self._pending_vocab_updates.clear()

        for req, accepted_toks in zip(active_batch, accepted_batches):
            out_msg.req_ids.append(req.req_id)
            out_msg.cached_prefix_lens.append(req.state_anchor_len)

            if not req.is_prefilled:
                assert req.req_pool_idx is not None and req.last_locked_node is not None

                # 当已算完的进度达到或超过初始 KV 锁定长度时，将新算出的 KV 与当前节点线性状态快照挂入树中
                # （若到达分裂出的公共前缀分支点 req.prefilled_len == req.prefix_len_at_start，自动触发该分支节点的状态自愈！）
                if req.prefilled_len >= req.prefix_len_at_start:
                    should_capture = req.is_prefill_complete or (
                        req.prefilled_len == req.prefix_len_at_start
                        and req.last_locked_node.linear_state_slot is None
                    )
                    new_prefix_len, new_locked_node = (
                        self.radix_cache.cache_unfinished_req(
                            req_idx=req.req_pool_idx,
                            token_ids=req.prompt_ids[: req.prefilled_len],
                            prefix_len_at_start=req.prefix_len_at_start,
                            last_locked_node=req.last_locked_node,
                            capture_linear_state=should_capture,
                        )
                    )
                    req.prefix_len_at_start = new_prefix_len
                    req.last_locked_node = new_locked_node

                progress_str = f"prefill {req.prefilled_len}/{len(req.prompt_ids)}"
                out_msg.chunk_progress.append(progress_str)

                if not req.is_prefill_complete:
                    # 尚有剩余 Prompt Chunk 未算完：不产出新 Token，继续留在 inflight_prefill_queue
                    out_msg.new_token_ids.append(None)
                    if out_msg.multi_token_ids is not None:
                        out_msg.multi_token_ids.append([])
                    out_msg.finished_flags.append(False)
                    continue

                # 最后一个 Prefill Chunk 已完成！采出首个生成 Token，并从 inflight_prefill_queue 移出
                next_tok = accepted_toks[0]
                req.is_prefilled = True
                if req in self.inflight_prefill_queue:
                    self.inflight_prefill_queue.remove(req)
                req.output_ids.append(next_tok)
                out_msg.new_token_ids.append(next_tok)
                if out_msg.multi_token_ids is not None:
                    out_msg.multi_token_ids.append([next_tok])

                finished = req.is_finished()
                out_msg.finished_flags.append(finished)
                if finished:
                    self._retire_request(req)
                else:
                    self.running_batch.append(req)
            else:
                # 当前请求处于 Decode 阶段（若 MTP 验证命中，单步可追加 2 个 Token）
                emitted_toks: List[int] = []
                for idx_tok, tok in enumerate(accepted_toks):
                    req.output_ids.append(tok)
                    emitted_toks.append(tok)
                    if req.is_finished():
                        if idx_tok == 0 and len(accepted_toks) == 2:
                            # 首个 Token 即触发停止（如 EOS），释放第 2 个多余槽位防止泄漏
                            self.token_to_kv_pool.free([req.out_cache_loc[1]])
                            assert req.req_pool_idx is not None
                            self.req_to_token_pool.write(
                                req.req_pool_idx, req.cur_start_pos + 1, [-1]
                            )
                            self.model_runner.invalidate_req_synced_len(
                                req.req_pool_idx, req.cur_start_pos + 1
                            )
                        break

                out_msg.chunk_progress.append(
                    f"decode {len(req.output_ids)}/{req.max_new_tokens}"
                )
                out_msg.new_token_ids.append(emitted_toks[-1])
                if out_msg.multi_token_ids is not None:
                    out_msg.multi_token_ids.append(emitted_toks)

                finished = req.is_finished()
                out_msg.finished_flags.append(finished)
                if finished:
                    if req in self.running_batch:
                        self.running_batch.remove(req)
                    self._retire_request(req)

        return out_msg

    def _preempt_request(self, req: Request) -> None:
        """
        抢占一个正在运行或处于 Chunked Prefill 中的请求并回退至 `waiting_queue` 头部：
        1. 释放尚未入树的私有 KV 槽位、解锁 RadixCache 节点并归还 `ReqToTokenPool` 行。
        2. 清零 `ReqToStatePool` 中的活跃线性状态工作区。
        3. 若请求已处于 Decode 阶段并产出部分 Token，将其合并回 `prompt_ids` 并扣减 `max_new_tokens`，
           使后续重新准入时无需重复生成已产出的 Token 且可继续复用 RadixCache 前缀。
        """
        if req.req_pool_idx is not None and req.last_locked_node is not None:
            req_idx = req.req_pool_idx
            if req.cached_token_count > req.prefix_len_at_start:
                uninserted = [
                    s
                    for s in self.req_to_token_pool.get_req_slots(
                        req_idx, req.cached_token_count
                    )[req.prefix_len_at_start : req.cached_token_count]
                    if s >= 0
                ]
                if uninserted:
                    self.model_runner.free_mtp_slots(uninserted)
            self.radix_cache.abort_req(
                req_idx=req_idx,
                allocated_seq_len=req.cached_token_count,
                prefix_len_at_start=req.prefix_len_at_start,
                last_locked_node=req.last_locked_node,
            )
            self.model_runner.reset_req_linear_state(req_idx)

        if req.is_prefilled and req.output_ids:
            req.max_new_tokens = max(1, req.max_new_tokens - len(req.output_ids))
            req.prompt_ids.extend(req.output_ids)
            req.output_ids.clear()

        req.req_pool_idx = None
        req.prefix_len_at_start = 0
        req.state_anchor_len = 0
        req.last_locked_node = None
        req.prefilled_len = 0
        req.is_prefilled = False
        req.cur_chunk_tokens = []
        req.cur_start_pos = 0
        req.cur_seq_len = 0
        req.out_cache_loc = []
        req.pending_draft_token = None
        req.cur_is_mtp_verify = False

        self.waiting_queue.insert(0, req)

    def _retire_request(self, req: Request) -> None:
        """
        请求完成时：
        1. 将已计算完成全部 KV + 线性状态的序列 `full_token_ids[:-1]` 归档至 RadixCache（含挂载终点线性快照）；
        2. 清零该请求在 `ReqToStatePool` 中的活跃工作区状态并释放请求行号。
        """
        assert req.req_pool_idx is not None and req.last_locked_node is not None
        req_idx = req.req_pool_idx
        computed_tokens = req.full_token_ids[:-1]
        self.radix_cache.cache_finished_req(
            req_idx=req_idx,
            token_ids=computed_tokens,
            prefix_len_at_start=req.prefix_len_at_start,
            last_locked_node=req.last_locked_node,
            capture_linear_state=True,
        )
        self.model_runner.reset_req_linear_state(req_idx)

    def abort_request(self, req_id: str) -> bool:
        """
        中途异常取消/中止指定请求：
        释放其尚未入树的私有物理 KV 槽位、解锁树节点、清零工作区线性状态并归还 ReqToTokenPool 行。
        """
        for i, req in enumerate(self.waiting_queue):
            if req.req_id == req_id:
                self.waiting_queue.pop(i)
                return True

        for queue_list in (self.inflight_prefill_queue, self.running_batch):
            for i, req in enumerate(queue_list):
                if req.req_id == req_id:
                    queue_list.pop(i)
                    if req.req_pool_idx is not None and req.last_locked_node is not None:
                        req_idx = req.req_pool_idx
                        self.radix_cache.abort_req(
                            req_idx=req_idx,
                            allocated_seq_len=req.cached_token_count,
                            prefix_len_at_start=req.prefix_len_at_start,
                            last_locked_node=req.last_locked_node,
                        )
                        self.model_runner.reset_req_linear_state(req_idx)
                    return True
        return False

    @staticmethod
    def run_process_loop(
        config: Qwen3_8Config,
        in_queue: mp.Queue,
        out_queue: mp.Queue,
        max_reqs: int = 8,
        max_context_len: int = 128,
        max_total_tokens: int = 256,
        max_snapshots: int = 16,
        chunked_prefill_size: int = 4,
        device: str = "cpu",
        enable_mtp: bool = False,
    ) -> None:
        """子进程入口函数：启动带混合注意力缓存与 Chunked Prefill 的 Scheduler 事件循环。"""
        scheduler = Scheduler(
            config=config,
            max_reqs=max_reqs,
            max_context_len=max_context_len,
            max_total_tokens=max_total_tokens,
            max_snapshots=max_snapshots,
            chunked_prefill_size=chunked_prefill_size,
            device=device,
            enable_mtp=enable_mtp,
        )

        keep_running = True
        while keep_running:
            keep_running = scheduler.recv_requests(in_queue)
            batch_out = scheduler.step()
            if batch_out is not None:
                out_queue.put(batch_out)
            else:
                time.sleep(0.002)

        out_queue.put(ShutdownSignal())
