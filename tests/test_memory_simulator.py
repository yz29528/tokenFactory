from pathlib import Path
import sys
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radix_tree import RadixCache, TreeNode
from req_to_state_pool import ReqToStatePool
from req_to_token_pool import ReqToTokenPool
from token_to_kv_pool import TokenToKVPool


class SGLangMemorySimulator:
    """
    模拟 SGLang Scheduler 中对混合注意力请求生命周期与双物理显存池
    (`TokenToKVPool` + `ReqToStatePool`) 的协同调度与非对称驱逐。
    """

    def __init__(
        self,
        max_reqs: int = 4,
        max_context_len: int = 16,
        max_kv_tokens: int = 12,
        max_state_snapshots: int = 2,
    ):
        self.req_to_token_pool = ReqToTokenPool(max_reqs, max_context_len)
        self.token_to_kv_pool = TokenToKVPool(max_kv_tokens)
        self.req_to_state_pool = ReqToStatePool(
            max_reqs=max_reqs, max_snapshots=max_state_snapshots
        )
        self.radix_cache = RadixCache(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_state_pool=self.req_to_state_pool,
        )

    def _alloc_kv_slots_with_evict(self, num_tokens: int) -> List[int]:
        """向 TokenToKVPool 申请物理槽位；如果余量不足，自动触发 RadixCache LRU 驱逐（含同步驱逐挂载的线性状态）。"""
        if self.token_to_kv_pool.available_size() < num_tokens:
            need_to_evict = num_tokens - self.token_to_kv_pool.available_size()
            print(
                f"  [Softmax KV 显存不足] 需申请 {num_tokens} 个槽位，当前仅剩 {self.token_to_kv_pool.available_size()} 个，"
                f"触发 RadixCache 驱逐 {need_to_evict} 个 token..."
            )
            evicted = self.radix_cache.evict(need_to_evict)
            print(f"  [Softmax KV 回收] RadixCache 已驱逐并回收物理 KV 槽位: {evicted}")

        slots = self.token_to_kv_pool.alloc(num_tokens)
        if slots is None:
            raise RuntimeError("Out of KV Cache Memory! 即使驱逐后物理槽位仍不足。")
        return slots

    def start_request_prefill(
        self, prompt_tokens: List[int]
    ) -> Tuple[int, List[int], int, TreeNode]:
        """
        阶段 1 (Prefill)：
        - 在 RadixCache 中执行混合前缀匹配 (`match_prefix_hybrid`)，同时获取 KV 前缀与最深线性状态锚点
        - 若命中线性状态快照，Fork 恢复至请求工作区；否则从最近状态锚点（或 0）推进
        - 对未拥有 KV 槽位的后缀向 `TokenToKVPool` 申请新槽位，并在 Prompt 结尾挂载线性状态快照
        """
        req_idx = self.req_to_token_pool.alloc_req()
        assert req_idx is not None, "ReqToTokenPool 行已满！"

        # 1. 混合前缀匹配与加锁
        match_res = self.radix_cache.match_prefix_hybrid(prompt_tokens)
        matched_kv_slots = match_res.matched_kv_slots
        last_node = match_res.last_kv_node
        self.radix_cache.inc_lock_ref(last_node)

        kv_prefix_len = len(matched_kv_slots)
        state_anchor_len = match_res.state_anchor_len

        # 2. 将命中的 Softmax KV 物理槽位直接写入 ReqToTokenPool 的前半行
        if kv_prefix_len > 0:
            self.req_to_token_pool.write(req_idx, 0, matched_kv_slots)

        # 3. 恢复线性注意力状态锚点快照（或重置为 ZERO_STATE）
        if match_res.linear_state_slot is not None and state_anchor_len > 0:
            self.req_to_state_pool.restore_snapshot(match_res.linear_state_slot, req_idx)
        else:
            self.req_to_state_pool.reset_working_state(req_idx)

        # 4. 为尚未拥有 KV 槽位的后缀申请新物理槽位
        suffix_kv_tokens = prompt_tokens[kv_prefix_len:]
        if suffix_kv_tokens:
            new_slots = self._alloc_kv_slots_with_evict(len(suffix_kv_tokens))
            self.req_to_token_pool.write(req_idx, kv_prefix_len, new_slots)
            self.token_to_kv_pool.set_kv_data(new_slots, suffix_kv_tokens)

        # 5. 更新工作区线性状态推进至 Prompt 末尾，并在 Prompt 边界节点保存快照
        self.req_to_state_pool.state_debug_tags[req_idx] = (
            f"State({prompt_tokens})"
        )
        kv_prefix_len, last_node = self.radix_cache.cache_unfinished_req(
            req_idx=req_idx,
            token_ids=prompt_tokens,
            prefix_len_at_start=kv_prefix_len,
            last_locked_node=last_node,
            capture_linear_state=True,
        )

        print(
            f"[Prefill] Req {req_idx} | Prompt={prompt_tokens} | "
            f"KV命中={len(matched_kv_slots)}, 线性状态锚点={state_anchor_len} (复用快照={match_res.linear_state_slot}) | "
            f"Prompt末尾节点挂载快照={last_node.linear_state_slot} (剩余空闲快照={self.req_to_state_pool.available_snapshot_size()})"
        )
        return req_idx, list(prompt_tokens), kv_prefix_len, last_node

    def step_decode(self, req_idx: int, seq_tokens: List[int], new_token: int) -> None:
        """
        阶段 2 (Decode)：
        - 每生成 1 个新 token，向 TokenToKVPool 申请 1 个物理槽位
        - 原地更新工作区 `req_idx` 的线性状态（Decode 期间不消耗新的快照槽位）
        """
        new_slot = self._alloc_kv_slots_with_evict(1)
        pos = len(seq_tokens)
        self.req_to_token_pool.write(req_idx, pos, new_slot)
        self.token_to_kv_pool.set_kv_data(new_slot, [new_token])
        seq_tokens.append(new_token)
        self.req_to_state_pool.state_debug_tags[req_idx] = f"State({seq_tokens})"
        print(
            f"[Decode]  Req {req_idx} | 生成新 Token={new_token} -> 分配物理 KV 槽位 {new_slot[0]} | "
            f"工作区状态={self.req_to_state_pool.state_debug_tags[req_idx]}"
        )

    def finish_request(
        self, req_idx: int, seq_tokens: List[int], prefix_len_at_start: int, last_node: TreeNode
    ) -> None:
        """
        阶段 3 (Finish)：
        - 将该请求整行的 (seq_tokens -> kv_slots) 存入 RadixCache，并在终点节点挂载线性注意力快照
        - 解锁树节点，释放 ReqToTokenPool 的行，清零工作区状态
        """
        self.radix_cache.cache_finished_req(
            req_idx, seq_tokens, prefix_len_at_start, last_node, capture_linear_state=True
        )
        self.req_to_state_pool.reset_working_state(req_idx)
        print(
            f"[Finish]  Req {req_idx} 完成！请求行已释放，空闲快照槽位数={self.req_to_state_pool.available_snapshot_size()}。"
        )


if __name__ == "__main__":
    # 创建一个总物理 KV 槽位 11 个 (有效 10 个)、线性状态快照槽位仅 2 个的混合显存模拟器
    sim = SGLangMemorySimulator(
        max_reqs=4, max_context_len=16, max_kv_tokens=11, max_state_snapshots=2
    )

    print("=== 步骤 1: 运行请求 A (Prompt [10, 20, 30, 40] + Decode [50]) ===")
    req_a, tokens_a, plen_a, node_a = sim.start_request_prefill([10, 20, 30, 40])
    sim.step_decode(req_a, tokens_a, new_token=50)
    sim.finish_request(req_a, tokens_a, plen_a, node_a)
    sim.radix_cache.pretty_print()

    print("\n=== 步骤 2: 运行请求 B (共享前缀 [10, 20, 30] + 新问题 [60, 70]) ===")
    print("    (注意：快照池仅 2 个槽位，当请求 B 挂载新快照时，会自动按 LRU 仅剥离最旧节点的 linear_state_slot，而不驱逐其 kv_slots！)")
    req_b, tokens_b, plen_b, node_b = sim.start_request_prefill([10, 20, 30, 60, 70])
    sim.step_decode(req_b, tokens_b, new_token=80)
    sim.radix_cache.pretty_print()

    print("\n=== 步骤 3: 运行请求 C (无共享前缀 [91, 92, 93, 94]，触发 Softmax KV 驱逐 + 同步回收挂载的线性快照) ===")
    req_c, tokens_c, plen_c, node_c = sim.start_request_prefill([91, 92, 93, 94])
    sim.radix_cache.pretty_print()

    print("\n=== 步骤 4: 请求 B 和请求 C 相继完成 ===")
    sim.finish_request(req_b, tokens_b, plen_b, node_b)
    sim.finish_request(req_c, tokens_c, plen_c, node_c)
    sim.radix_cache.pretty_print()
