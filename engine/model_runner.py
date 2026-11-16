from typing import Dict, List, Optional, Tuple
import torch

from qwen3_8 import (
    ForwardBatch,
    Qwen3_8Config,
    Qwen3_8ForCausalLM,
    Qwen3_8HybridCachePool,
)
from req_to_state_pool import ReqToStatePool
from req_to_token_pool import ReqToTokenPool
from token_to_kv_pool import TokenToKVPool
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
        enable_mtp: bool = False,
    ):
        self.config = config
        self.max_reqs = max_reqs
        self.max_context_len = max_context_len
        self.max_total_tokens = max_total_tokens
        self.max_snapshots = max_snapshots
        self.device = device
        self.dtype = dtype
        self.enable_mtp = enable_mtp

        self.model = Qwen3_8ForCausalLM(config, enable_mtp=enable_mtp).to(
            device=device, dtype=dtype
        )
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
        self.mtp_cache_pool: Optional[Qwen3_8HybridCachePool] = (
            self.cache_pool.get_mtp_cache_pool()
            if (enable_mtp and self.model.mtp is not None)
            else None
        )

        # 常驻设备的 req_to_token 映射张量 [max_reqs, max_context_len]，每步仅增量更新变动切片
        self.req_to_token_tensor = torch.full(
            (max_reqs, max_context_len),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._req_synced_lens: List[int] = [0] * max_reqs
        self.last_mtp_draft_tokens: List[int] = []
        # 记录每个物理 KV 槽位在 MTP 层对应的主模型隐状态及当时使用的 next_token，用于前缀复用边界校准
        self._mtp_slot_hidden: Dict[int, torch.Tensor] = {}
        self._mtp_slot_next_tok: Dict[int, int] = {}

    @property
    def state_pool(self) -> ReqToStatePool:
        """返回底层 48 层线性注意力的 ReqToStatePool（含活跃工作区与 RadixCache 快照区）。"""
        return self.cache_pool.state_pool

    @torch.inference_mode()
    def reset_req_linear_state(self, req_idx: int) -> None:
        """清空已完成或新准入请求在 48 层 Gated DeltaNet 线性注意力中的工作区状态及映射同步游标。"""
        self.cache_pool.reset_req_linear_state(req_idx)
        self._req_synced_lens[req_idx] = 0

    @torch.inference_mode()
    def restore_req_linear_state(self, snapshot_slot: int, req_idx: int) -> None:
        """将 RadixCache 状态锚点的快照 (`snapshot_slot`) 深拷贝恢复至请求工作区 (`req_idx`)。"""
        self.cache_pool.restore_req_snapshot(snapshot_slot, req_idx)
        self._req_synced_lens[req_idx] = 0

    def invalidate_req_synced_len(self, req_idx: int, pos: int = 0) -> None:
        """当 CPU 侧 `req_to_token` 表发生前缀去重覆写时，回退设备端同步游标以强制刷新。"""
        if pos < self._req_synced_lens[req_idx]:
            self._req_synced_lens[req_idx] = pos

    def free_mtp_slots(self, slots: List[int]) -> None:
        """当物理 KV 槽位被释放回收时，同步清理其关联的 MTP 隐状态与边界 Token 缓存。"""
        for s in slots:
            self._mtp_slot_hidden.pop(s, None)
            self._mtp_slot_next_tok.pop(s, None)

    def _prepare_forward_batch(
        self,
        batch_reqs: List[Request],
        req_to_token_table: List[List[int]],
        dirty_from: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, ForwardBatch]:
        """拍平批次请求并增量同步设备端 `req_to_token_tensor`，构造 `ForwardBatch`。"""
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
        out_cache_loc_tensor = torch.tensor(
            flat_out_cache_loc, dtype=torch.long, device=self.device
        )

        # 增量同步常驻设备的 req_to_token_tensor，避免每步全量重建 [max_reqs, max_context_len]
        loc_offset = 0
        for req in batch_reqs:
            assert req.req_pool_idx is not None
            r_idx = req.req_pool_idx
            ext_len = len(req.cur_chunk_tokens)
            synced_len = self._req_synced_lens[r_idx]
            if dirty_from is not None and dirty_from[r_idx] < synced_len:
                synced_len = dirty_from[r_idx]
            if synced_len < req.cur_start_pos:
                prefix_slice = req_to_token_table[r_idx][synced_len : req.cur_start_pos]
                self.req_to_token_tensor[r_idx, synced_len : req.cur_start_pos] = (
                    torch.tensor(prefix_slice, dtype=torch.long, device=self.device)
                )
            self.req_to_token_tensor[r_idx, req.cur_start_pos : req.cur_seq_len] = (
                out_cache_loc_tensor[loc_offset : loc_offset + ext_len]
            )
            self._req_synced_lens[r_idx] = req.cur_seq_len
            if dirty_from is not None:
                dirty_from[r_idx] = req.cur_seq_len
            loc_offset += ext_len

        extend_lens_tensor = torch.tensor(
            extend_lens, dtype=torch.long, device=self.device
        )
        forward_batch = ForwardBatch(
            batch_size=len(batch_reqs),
            req_pool_indices=torch.tensor(
                req_pool_indices, dtype=torch.long, device=self.device
            ),
            seq_lens=torch.tensor(seq_lens, dtype=torch.long, device=self.device),
            prefix_lens=torch.tensor(prefix_lens, dtype=torch.long, device=self.device),
            extend_lens=extend_lens,
            out_cache_loc=out_cache_loc_tensor,
            req_to_token=self.req_to_token_tensor,
            is_prefill=any(not r.is_prefilled for r in batch_reqs),
            max_seq_len=max(seq_lens) if seq_lens else 0,
            max_extend_len=max(extend_lens) if extend_lens else 0,
            extend_lens_tensor=extend_lens_tensor,
            req_pool_indices_list=req_pool_indices,
        )
        return input_ids_tensor, positions_tensor, forward_batch

    @torch.inference_mode()
    def execute_batch(
        self,
        batch_reqs: List[Request],
        req_to_token_table: List[List[int]],
        dirty_from: Optional[List[int]] = None,
    ) -> List[int]:
        """
        将一批任意混合了 Decode 与 Chunked Prefill 的 Request 拍平为一维张量，
        构造 `ForwardBatch` 执行前向计算并采样出每个请求对应位置的输出 Token ID。
        """
        input_ids_tensor, positions_tensor, forward_batch = self._prepare_forward_batch(
            batch_reqs, req_to_token_table, dirty_from
        )

        # 1. 模型前向计算: [batch_size, vocab_size]
        logits = self.model(
            input_ids=input_ids_tensor,
            positions=positions_tensor,
            cache_pool=self.cache_pool,
            forward_batch=forward_batch,
        )

        # 2. 逐请求采样
        sampled_tokens = self._sample(logits, batch_reqs)

        # 3. 若启用了 MTP，仅对已完成 Prefill（真正产出采样 Token）的请求生成投机草稿 Token
        if self.enable_mtp and self.mtp_cache_pool is not None:
            self.last_mtp_draft_tokens = self.propose_mtp_draft_tokens(
                batch_reqs=batch_reqs,
                sampled_token_ids=sampled_tokens,
                forward_batch=forward_batch,
            )

        return sampled_tokens

    @torch.inference_mode()
    def execute_batch_speculative(
        self,
        batch_reqs: List[Request],
        req_to_token_table: List[List[int]],
        dirty_from: Optional[List[int]],
        token_to_kv_pool: TokenToKVPool,
        req_to_token_pool: ReqToTokenPool,
    ) -> List[List[int]]:
        """
        带 MTP 投机验证与状态回滚的混合批次执行：
        - 对带有 `cur_is_mtp_verify=True` 的请求（`[last_tok, draft_tok]`），一次性前向计算 2 个位置并校验草稿是否命中：
          - 命中：单步产出 `[target_1, bonus_2]` 两个 Token，并继续用 MTP 预测下一草稿；
          - 未命中：释放拒绝槽位 `out_cache_loc[1]`，恢复线性注意力状态并重算 `last_tok`（1 Token）保证状态 bit-exact，产出 `[target_1]`。
        - 对 Prefill 分块与普通 Decode 请求，同步填充 `mtp_cache_pool` 并在 Prefill 完成或 Decode 后提出下一个草稿 Token。
        """
        assert self.mtp_cache_pool is not None and self.model.mtp is not None

        # 1. 对待验证的请求保存其线性注意力工作区快照，以备草稿被拒绝时精确回滚
        saved_linear_states: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for req in batch_reqs:
            if req.cur_is_mtp_verify:
                assert req.req_pool_idx is not None
                r_idx = req.req_pool_idx
                saved_linear_states[r_idx] = (
                    self.cache_pool.conv_state[:, r_idx].clone(),
                    self.cache_pool.ssm_state[:, r_idx].clone(),
                )

        input_ids_tensor, positions_tensor, forward_batch = self._prepare_forward_batch(
            batch_reqs, req_to_token_table, dirty_from
        )

        # 2. 模型前向计算全部位置 Logits: [num_tokens, vocab_size]
        all_logits = self.model(
            input_ids=input_ids_tensor,
            positions=positions_tensor,
            cache_pool=self.cache_pool,
            forward_batch=forward_batch,
            return_all_logits=True,
        )
        main_all_hidden = self.model._last_hidden_states
        assert main_all_hidden is not None

        accepted_per_req: List[List[int]] = []
        mtp_drafts: List[int] = []
        offset = 0

        for req in batch_reqs:
            assert req.req_pool_idx is not None
            r_idx = req.req_pool_idx
            ext_len = len(req.cur_chunk_tokens)
            req_logits = all_logits[offset : offset + ext_len]
            req_hidden = main_all_hidden[offset : offset + ext_len]
            offset += ext_len

            if req.cur_is_mtp_verify:
                assert ext_len == 2 and req.pending_draft_token is not None
                draft_tok = req.pending_draft_token
                target_1 = self._sample_one(req_logits[0], req)
                stops_on_first = (
                    req.eos_token_id is not None and target_1 == req.eos_token_id
                ) or (len(req.output_ids) + 1 >= req.max_new_tokens)

                if target_1 == draft_tok and not stops_on_first:
                    bonus_2 = self._sample_one(req_logits[1], req)
                    accepted_per_req.append([target_1, bonus_2])
                    # 两个 Token 均有效：将 [draft_tok, bonus_2] 喂给 MTP 填充缓冲并生成下一草稿
                    next_draft = self._run_mtp_for_req(
                        req_idx=r_idx,
                        req=req,
                        main_hidden=req_hidden,
                        next_tokens=[draft_tok, bonus_2],
                        start_pos=req.cur_start_pos,
                        out_locs=req.out_cache_loc,
                        propose_draft=True,
                    )
                    assert next_draft is not None
                    req.pending_draft_token = next_draft
                    mtp_drafts.append(next_draft)
                else:
                    # 草稿被拒绝，或第 1 个 Token 即触发停止条件（如 EOS）：
                    # 释放第 2 个 KV 槽位并精确回滚线性状态至单步，防止多算的第 2 步状态污染 RadixCache 快照
                    accepted_per_req.append([target_1])
                    rejected_slot = req.out_cache_loc[1]
                    token_to_kv_pool.free([rejected_slot])
                    self.free_mtp_slots([rejected_slot])
                    req_to_token_pool.write(r_idx, req.cur_start_pos + 1, [-1])
                    self.invalidate_req_synced_len(r_idx, req.cur_start_pos + 1)
                    self.req_to_token_tensor[r_idx, req.cur_start_pos + 1] = -1

                    saved_conv, saved_ssm = saved_linear_states[r_idx]
                    self.cache_pool.conv_state[:, r_idx].copy_(saved_conv)
                    self.cache_pool.ssm_state[:, r_idx].copy_(saved_ssm)

                    # 用真实有效的第 1 个 Token 重算 1 步以恢复准确的线性状态与隐状态
                    single_batch = ForwardBatch(
                        batch_size=1,
                        req_pool_indices=torch.tensor(
                            [r_idx], dtype=torch.long, device=self.device
                        ),
                        seq_lens=torch.tensor(
                            [req.cur_start_pos + 1], dtype=torch.long, device=self.device
                        ),
                        prefix_lens=torch.tensor(
                            [req.cur_start_pos], dtype=torch.long, device=self.device
                        ),
                        extend_lens=[1],
                        out_cache_loc=torch.tensor(
                            [req.out_cache_loc[0]], dtype=torch.long, device=self.device
                        ),
                        req_to_token=self.req_to_token_tensor,
                        is_prefill=False,
                        max_seq_len=req.cur_start_pos + 1,
                        max_extend_len=1,
                        extend_lens_tensor=torch.ones(
                            1, dtype=torch.long, device=self.device
                        ),
                        req_pool_indices_list=[r_idx],
                    )
                    self.model(
                        input_ids=torch.tensor(
                            [req.cur_chunk_tokens[0]],
                            dtype=torch.long,
                            device=self.device,
                        ),
                        positions=torch.tensor(
                            [req.cur_start_pos], dtype=torch.long, device=self.device
                        ),
                        cache_pool=self.cache_pool,
                        forward_batch=single_batch,
                    )
                    assert self.model._last_token_hidden_states is not None
                    if stops_on_first:
                        req.pending_draft_token = None
                    else:
                        next_draft = self._run_mtp_for_req(
                            req_idx=r_idx,
                            req=req,
                            main_hidden=self.model._last_token_hidden_states,
                            next_tokens=[target_1],
                            start_pos=req.cur_start_pos,
                            out_locs=[req.out_cache_loc[0]],
                            propose_draft=True,
                        )
                        assert next_draft is not None
                        req.pending_draft_token = next_draft
                        mtp_drafts.append(next_draft)
            else:
                tok_0 = self._sample_one(req_logits[-1], req)
                accepted_per_req.append([tok_0])
                if not req.is_prefilled and not req.is_prefill_complete:
                    # 中间 Prefill 分块：使用已知的下一分块 Prompt Token 预填充 mtp_cache_pool
                    shifted_next = req.prompt_ids[
                        req.cur_start_pos + 1 : req.cur_seq_len + 1
                    ]
                    self._run_mtp_for_req(
                        req_idx=r_idx,
                        req=req,
                        main_hidden=req_hidden,
                        next_tokens=shifted_next,
                        start_pos=req.cur_start_pos,
                        out_locs=req.out_cache_loc,
                        propose_draft=False,
                    )
                    req.pending_draft_token = None
                else:
                    if not req.is_prefilled:
                        shifted_next = (
                            req.prompt_ids[req.cur_start_pos + 1 : req.cur_seq_len]
                            + [tok_0]
                        )
                    else:
                        shifted_next = [tok_0]
                    next_draft = self._run_mtp_for_req(
                        req_idx=r_idx,
                        req=req,
                        main_hidden=req_hidden,
                        next_tokens=shifted_next,
                        start_pos=req.cur_start_pos,
                        out_locs=req.out_cache_loc,
                        propose_draft=True,
                    )
                    assert next_draft is not None
                    req.pending_draft_token = next_draft
                    mtp_drafts.append(next_draft)

        self.last_mtp_draft_tokens = mtp_drafts
        return accepted_per_req

    def _run_mtp_for_req(
        self,
        req_idx: int,
        req: Request,
        main_hidden: torch.Tensor,
        next_tokens: List[int],
        start_pos: int,
        out_locs: List[int],
        propose_draft: bool,
    ) -> Optional[int]:
        """在专属 `mtp_cache_pool` 中为单个请求前向填充位置 KV 并可选采样下一草稿 Token。"""
        assert self.mtp_cache_pool is not None
        # 若当前请求存在前置前缀 (start_pos > 0)，扫描前缀中的所有位置 [0, start_pos)，
        # 检查是否有任何分支分叉点槽位在 mtp_cache_pool 中被其它并发分支覆写成了不同的 next_token；
        # 若不一致则按位置顺序刷新该槽位的 MTP KV，保证并发共享前缀下所有请求的 MTP 上下文精确一致
        if start_pos > 0 and req.cur_chunk_tokens:
            full_seq = req.full_token_ids
            prefix_slots = self.req_to_token_tensor[req_idx, :start_pos].tolist()
            for prev_pos, prev_slot in enumerate(prefix_slots):
                actual_next = full_seq[prev_pos + 1]
                if (
                    prev_slot > 0
                    and prev_slot in self._mtp_slot_hidden
                    and self._mtp_slot_next_tok.get(prev_slot) != actual_next
                ):
                    fix_batch = ForwardBatch(
                        batch_size=1,
                        req_pool_indices=torch.tensor(
                            [req_idx], dtype=torch.long, device=self.device
                        ),
                        seq_lens=torch.tensor(
                            [prev_pos + 1], dtype=torch.long, device=self.device
                        ),
                        prefix_lens=torch.tensor(
                            [prev_pos], dtype=torch.long, device=self.device
                        ),
                        extend_lens=[1],
                        out_cache_loc=torch.tensor(
                            [prev_slot], dtype=torch.long, device=self.device
                        ),
                        req_to_token=self.req_to_token_tensor,
                        is_prefill=False,
                        max_seq_len=prev_pos + 1,
                        max_extend_len=1,
                        extend_lens_tensor=torch.ones(
                            1, dtype=torch.long, device=self.device
                        ),
                        req_pool_indices_list=[req_idx],
                    )
                    self.model.forward_mtp(
                        next_token_ids=torch.tensor(
                            [actual_next], dtype=torch.long, device=self.device
                        ),
                        positions=torch.tensor(
                            [prev_pos], dtype=torch.long, device=self.device
                        ),
                        mtp_cache_pool=self.mtp_cache_pool,
                        forward_batch=fix_batch,
                        main_hidden_states=self._mtp_slot_hidden[prev_slot],
                    )
                    self._mtp_slot_next_tok[prev_slot] = actual_next

        ext_len = len(next_tokens)
        for i, (slot, n_tok) in enumerate(zip(out_locs, next_tokens)):
            self._mtp_slot_hidden[slot] = main_hidden[i : i + 1].detach().clone()
            self._mtp_slot_next_tok[slot] = n_tok

        end_pos = start_pos + ext_len
        mtp_batch = ForwardBatch(
            batch_size=1,
            req_pool_indices=torch.tensor(
                [req_idx], dtype=torch.long, device=self.device
            ),
            seq_lens=torch.tensor([end_pos], dtype=torch.long, device=self.device),
            prefix_lens=torch.tensor(
                [start_pos], dtype=torch.long, device=self.device
            ),
            extend_lens=[ext_len],
            out_cache_loc=torch.tensor(out_locs, dtype=torch.long, device=self.device),
            req_to_token=self.req_to_token_tensor,
            is_prefill=ext_len > 1,
            max_seq_len=end_pos,
            max_extend_len=ext_len,
            extend_lens_tensor=torch.tensor(
                [ext_len], dtype=torch.long, device=self.device
            ),
            req_pool_indices_list=[req_idx],
        )
        mtp_logits = self.model.forward_mtp(
            next_token_ids=torch.tensor(
                next_tokens, dtype=torch.long, device=self.device
            ),
            positions=torch.arange(
                start_pos, end_pos, dtype=torch.long, device=self.device
            ),
            mtp_cache_pool=self.mtp_cache_pool,
            forward_batch=mtp_batch,
            main_hidden_states=main_hidden,
        )
        if not propose_draft:
            return None
        return self._sample_one(mtp_logits[0], req)

    @torch.inference_mode()
    def propose_mtp_draft_tokens(
        self,
        batch_reqs: List[Request],
        sampled_token_ids: List[int],
        forward_batch: ForwardBatch,
    ) -> List[int]:
        """使用 MTP 预测头与专属 `mtp_cache_pool` 填充每个请求当前片段的 MTP KV，并对已完成 Prefill 的请求预测投机草稿 Token。"""
        assert self.mtp_cache_pool is not None
        assert self.model._last_hidden_states is not None
        main_all_hidden = self.model._last_hidden_states

        drafts: List[int] = []
        offset = 0
        for req, tok_0 in zip(batch_reqs, sampled_token_ids):
            assert req.req_pool_idx is not None
            r_idx = req.req_pool_idx
            ext_len = len(req.cur_chunk_tokens)
            req_hidden = main_all_hidden[offset : offset + ext_len]
            offset += ext_len

            if not req.is_prefilled and not req.is_prefill_complete:
                shifted_next = req.prompt_ids[
                    req.cur_start_pos + 1 : req.cur_seq_len + 1
                ]
                self._run_mtp_for_req(
                    req_idx=r_idx,
                    req=req,
                    main_hidden=req_hidden,
                    next_tokens=shifted_next,
                    start_pos=req.cur_start_pos,
                    out_locs=req.out_cache_loc,
                    propose_draft=False,
                )
            else:
                if not req.is_prefilled:
                    shifted_next = (
                        req.prompt_ids[req.cur_start_pos + 1 : req.cur_seq_len]
                        + [tok_0]
                    )
                else:
                    shifted_next = [tok_0]
                next_draft = self._run_mtp_for_req(
                    req_idx=r_idx,
                    req=req,
                    main_hidden=req_hidden,
                    next_tokens=shifted_next,
                    start_pos=req.cur_start_pos,
                    out_locs=req.out_cache_loc,
                    propose_draft=True,
                )
                assert next_draft is not None
                drafts.append(next_draft)
        return drafts

    def _sample_one(self, req_logits: torch.Tensor, req: Request) -> int:
        if req.temperature <= 1e-5:
            return int(torch.argmax(req_logits, dim=-1).item())
        probs = torch.softmax(req_logits / req.temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def _sample(self, logits: torch.Tensor, batch_reqs: List[Request]) -> List[int]:
        return [self._sample_one(logits[i], req) for i, req in enumerate(batch_reqs)]
