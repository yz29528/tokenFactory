import multiprocessing as mp
import queue
import time
from typing import List, Optional
import torch

from qwen3_8 import Qwen3_8Config
from radix_tree import RadixCache
from req_to_token_pool import ReqToTokenPool
from token_to_kv_pool import TokenToKVPool
from .ipc_messages import BatchTokenIDOut, ShutdownSignal, TokenizedReqInput
from .model_runner import ModelRunner
from .request import Request


class Scheduler:
    """
    进程 2 (Scheduler 进程)：支持 **Chunked Prefill (分块预填充)** 与 **Mixed Batching (Prefill+Decode 混合批处理)** 的 Continuous Batching 调度器。
    核心机制：
    1. 每个 Step 设定最大前向 Token 预算 `chunked_prefill_size` (例如 4 或 512)。
    2. **Decode 保底与混合组批**：
       - 首先将 `running_batch` 中所有正在 Decode 的请求加入本步批次（每个请求消耗 1 个 Token 预算），保证正在生成的请求绝不因为长 Prompt 到来而停顿。
       - 若预算仍有剩余 (`remaining_budget > 0`)，继续从正在分块中的 `inflight_prefill_queue` 或新到达的 `waiting_queue` 中切出长度为 `min(remaining_prefill_len, remaining_budget)` 的 Prompt 分块（Chunk），与 Decode 请求**合批在同一个 `ForwardBatch`** 中并行计算！
    3. **分块推进与转正**：
       - 若某个 Prefill 请求在本步仅算完中间一个 Chunk (`not req.is_prefill_complete`)，仅更新其 KV 缓存与线性注意力状态，忽略本步采样值，留待下一 Step 继续算剩余 Chunk。
       - 一旦最后一个 Chunk 算完 (`req.is_prefill_complete`)，正式采出第 1 个新 Token 并转入 `running_batch`。
    """

    def __init__(
        self,
        config: Qwen3_8Config,
        max_reqs: int = 8,
        max_context_len: int = 128,
        max_total_tokens: int = 256,
        chunked_prefill_size: int = 4,
        device: str = "cpu",
    ):
        self.config = config
        self.max_reqs = max_reqs
        self.max_context_len = max_context_len
        self.max_total_tokens = max_total_tokens
        self.chunked_prefill_size = chunked_prefill_size
        self.device = device

        # 初始化两级显存池与 RadixCache
        self.req_to_token_pool = ReqToTokenPool(max_reqs, max_context_len)
        self.token_to_kv_pool = TokenToKVPool(max_total_tokens)
        self.radix_cache = RadixCache(self.req_to_token_pool, self.token_to_kv_pool)

        # 初始化底层算子执行器 ModelRunner
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model_runner = ModelRunner(
            config=config,
            max_reqs=max_reqs,
            max_context_len=max_context_len,
            max_total_tokens=max_total_tokens,
            device=device,
            dtype=dtype,
        )

        # 三个状态队列：
        # 1. waiting_queue: 尚未分配 req_pool_idx 的新请求
        # 2. inflight_prefill_queue: 已分配 req_pool_idx、正在分块执行 Chunked Prefill 的请求
        # 3. running_batch: 已完成全部 Prefill、正在逐 Token Decode 的请求
        self.waiting_queue: List[Request] = []
        self.inflight_prefill_queue: List[Request] = []
        self.running_batch: List[Request] = []
        self.step_counter: int = 0

    def _alloc_kv_slots(self, num_tokens: int) -> Optional[List[int]]:
        """申请物理 KV 槽位；若不足则触发 RadixCache LRU 驱逐。"""
        if self.token_to_kv_pool.available_size() < num_tokens:
            need = num_tokens - self.token_to_kv_pool.available_size()
            self.radix_cache.evict(need)
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
                req = Request(
                    req_id=msg.req_id,
                    prompt_text=msg.prompt_text,
                    prompt_ids=msg.prompt_ids,
                    max_new_tokens=msg.max_new_tokens,
                    eos_token_id=msg.eos_token_id or self.config.eos_token_id,
                    temperature=msg.temperature,
                )
                self.waiting_queue.append(req)
        return True

    def _admit_waiting_requests(self) -> None:
        """
        将 `waiting_queue` 中的新请求准入到 `inflight_prefill_queue`：
        匹配 `RadixCache` 最长前缀并加锁、分配 `ReqToTokenPool` 行，并将 `prefilled_len` 初始化为命中前缀长度。
        """
        remaining: List[Request] = []
        for req in self.waiting_queue:
            if not self.req_to_token_pool.free_req_slots:
                remaining.append(req)
                continue

            req_idx = self.req_to_token_pool.alloc_req()
            assert req_idx is not None
            self.model_runner.reset_req_linear_state(req_idx)

            # 查找 RadixCache 可复用前缀
            matched_slots, last_node = self.radix_cache.match_prefix(req.prompt_ids)
            if len(matched_slots) == len(req.prompt_ids) and len(matched_slots) > 0:
                matched_slots = matched_slots[:-1]

            prefix_len = len(matched_slots)
            self.radix_cache.inc_lock_ref(last_node)

            if prefix_len > 0:
                self.req_to_token_pool.write(req_idx, 0, matched_slots)

            req.req_pool_idx = req_idx
            req.prefix_len_at_start = prefix_len
            req.prefilled_len = prefix_len
            req.last_locked_node = last_node

            self.inflight_prefill_queue.append(req)

        self.waiting_queue = remaining

    def _build_mixed_chunked_batch(self) -> tuple[List[Request], str]:
        """
        核心组批逻辑（Chunked Prefill + Decode 混合批处理）：
        1. 先为 `running_batch` 中每个活跃 Decode 请求分配 1 个 Token 槽位。
        2. 计算剩余 Token 预算 `budget = max(1, chunked_prefill_size - len(decode_reqs))`。
        3. 使用剩余预算从 `inflight_prefill_queue` 中切分 Prompt Chunk（可跨请求或对单个长请求分块）。
        """
        active_batch: List[Request] = []
        has_decode = False
        has_prefill = False

        # 1. 优先接纳所有正在 Decode 的请求（每个仅占 1 个 Token 预算）
        for req in self.running_batch:
            assert req.req_pool_idx is not None
            new_slot = self._alloc_kv_slots(1)
            if new_slot is None:
                raise RuntimeError("KV Cache 物理池耗尽，无法为 Decode 请求分配槽位！")

            start_pos = req.seq_len - 1
            self.req_to_token_pool.write(req.req_pool_idx, start_pos, new_slot)

            req.cur_chunk_tokens = [req.output_ids[-1]]
            req.cur_start_pos = start_pos
            req.cur_seq_len = req.seq_len
            req.out_cache_loc = new_slot
            active_batch.append(req)
            has_decode = True

        # 2. 尝试将 waiting_queue 中的新请求准入到 inflight_prefill_queue
        if self.waiting_queue:
            self._admit_waiting_requests()

        # 3. 利用剩余 Token 预算对 inflight_prefill_queue 执行分块预填充 (Chunked Prefill)
        remaining_budget = self.chunked_prefill_size - len(active_batch)
        # 即使 Decode 已经占满预算，若想保证 Prefill 不饿死，也可按配置给至少 0 或剩余预算；这里严格遵守总预算上限（若无 Decode 则预算为全额 chunked_prefill_size）
        for req in list(self.inflight_prefill_queue):
            if remaining_budget <= 0:
                break

            assert req.req_pool_idx is not None
            chunk_len = min(req.remaining_prefill_len, remaining_budget)
            new_slots = self._alloc_kv_slots(chunk_len)
            if new_slots is None:
                break

            start_pos = req.prefilled_len
            end_pos = start_pos + chunk_len
            self.req_to_token_pool.write(req.req_pool_idx, start_pos, new_slots)

            req.cur_chunk_tokens = req.prompt_ids[start_pos:end_pos]
            req.cur_start_pos = start_pos
            req.cur_seq_len = end_pos
            req.out_cache_loc = new_slots
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
        sampled_token_ids = self.model_runner.execute_batch(
            batch_reqs=active_batch,
            req_to_token_table=self.req_to_token_pool.req_to_token,
        )

        out_msg = BatchTokenIDOut(step_id=self.step_counter, batch_mode=batch_mode)

        for req, next_tok in zip(active_batch, sampled_token_ids):
            out_msg.req_ids.append(req.req_id)
            out_msg.cached_prefix_lens.append(req.prefix_len_at_start)

            if not req.is_prefilled:
                # 当前请求处于 Chunked Prefill 阶段
                progress_str = f"prefill {req.prefilled_len}/{len(req.prompt_ids)}"
                out_msg.chunk_progress.append(progress_str)

                if not req.is_prefill_complete:
                    # 尚有剩余 Prompt Chunk 未算完：不产出新 Token，继续留在 inflight_prefill_queue
                    out_msg.new_token_ids.append(None)
                    out_msg.finished_flags.append(False)
                    continue

                # 最后一个 Prefill Chunk 已完成！采出首个生成 Token，并从 inflight_prefill_queue 移出
                req.is_prefilled = True
                if req in self.inflight_prefill_queue:
                    self.inflight_prefill_queue.remove(req)
                req.output_ids.append(next_tok)
                out_msg.new_token_ids.append(next_tok)

                finished = req.is_finished()
                out_msg.finished_flags.append(finished)
                if finished:
                    self._retire_request(req)
                else:
                    self.running_batch.append(req)
            else:
                # 当前请求处于 Decode 阶段
                req.output_ids.append(next_tok)
                out_msg.chunk_progress.append(
                    f"decode {len(req.output_ids)}/{req.max_new_tokens}"
                )
                out_msg.new_token_ids.append(next_tok)

                finished = req.is_finished()
                out_msg.finished_flags.append(finished)
                if finished:
                    if req in self.running_batch:
                        self.running_batch.remove(req)
                    self._retire_request(req)

        return out_msg

    def _retire_request(self, req: Request) -> None:
        """请求完成时归档已计算 KV 的序列至 RadixCache 并释放第一层池行号。"""
        assert req.req_pool_idx is not None and req.last_locked_node is not None
        computed_tokens = req.full_token_ids[:-1]
        self.radix_cache.cache_finished_req(
            req_idx=req.req_pool_idx,
            token_ids=computed_tokens,
            prefix_len_at_start=req.prefix_len_at_start,
            last_locked_node=req.last_locked_node,
        )
        self.model_runner.reset_req_linear_state(req.req_pool_idx)

    @staticmethod
    def run_process_loop(
        config: Qwen3_8Config,
        in_queue: mp.Queue,
        out_queue: mp.Queue,
        max_reqs: int = 8,
        max_context_len: int = 128,
        max_total_tokens: int = 256,
        chunked_prefill_size: int = 4,
    ) -> None:
        """子进程入口函数：启动带 Chunked Prefill 的 Scheduler 事件循环。"""
        scheduler = Scheduler(
            config=config,
            max_reqs=max_reqs,
            max_context_len=max_context_len,
            max_total_tokens=max_total_tokens,
            chunked_prefill_size=chunked_prefill_size,
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
