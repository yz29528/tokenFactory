from pathlib import Path
import sys
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radix_tree import RadixCache, TreeNode
from req_to_token_pool import ReqToTokenPool
from token_to_kv_pool import TokenToKVPool


class SGLangMemorySimulator:
    """模拟 SGLang Scheduler 中对请求生命周期与两级显存池的调度。"""

    def __init__(self, max_reqs: int = 4, max_context_len: int = 16, max_kv_tokens: int = 12):
        self.req_to_token_pool = ReqToTokenPool(max_reqs, max_context_len)
        self.token_to_kv_pool = TokenToKVPool(max_kv_tokens)
        self.radix_cache = RadixCache(self.req_to_token_pool, self.token_to_kv_pool)

    def _alloc_kv_slots_with_evict(self, num_tokens: int) -> List[int]:
        """向 TokenToKVPool 申请物理槽位；如果余量不足，自动触发 RadixCache LRU 驱逐。"""
        if self.token_to_kv_pool.available_size() < num_tokens:
            need_to_evict = num_tokens - self.token_to_kv_pool.available_size()
            print(
                f"  [显存不足] 需申请 {num_tokens} 个槽位，当前仅剩 {self.token_to_kv_pool.available_size()} 个，"
                f"触发 RadixCache 驱逐 {need_to_evict} 个 token..."
            )
            evicted = self.radix_cache.evict(need_to_evict)
            print(f"  [显存回收] RadixCache 已驱逐并回收物理槽位: {evicted}")

        slots = self.token_to_kv_pool.alloc(num_tokens)
        if slots is None:
            raise RuntimeError("Out of KV Cache Memory! 即使驱逐后物理槽位仍不足。")
        return slots

    def start_request_prefill(self, prompt_tokens: List[int]) -> Tuple[int, List[int], int, TreeNode]:
        """
        阶段 1 (Prefill)：
        - 在 RadixCache 中匹配最长公共前缀并加锁
        - 在第一层 ReqToTokenPool 分配一行，直接拷入命中的物理槽位索引（零拷贝复用）
        - 仅对未命中的后缀 Token 向第二层 TokenToKVPool 申请新物理槽位并计算 KV
        """
        req_idx = self.req_to_token_pool.alloc_req()
        assert req_idx is not None, "ReqToTokenPool 行已满！"

        # 1. 前缀匹配与加锁
        matched_kv_slots, last_node = self.radix_cache.match_prefix(prompt_tokens)
        self.radix_cache.inc_lock_ref(last_node)
        prefix_len = len(matched_kv_slots)

        # 2. 将命中的物理槽位直接写入 ReqToTokenPool 的前半行
        if prefix_len > 0:
            self.req_to_token_pool.write(req_idx, 0, matched_kv_slots)

        # 3. 为未命中的后缀申请新物理槽位，写入 ReqToTokenPool 并写入真实 KV 向量
        suffix_tokens = prompt_tokens[prefix_len:]
        if suffix_tokens:
            new_slots = self._alloc_kv_slots_with_evict(len(suffix_tokens))
            self.req_to_token_pool.write(req_idx, prefix_len, new_slots)
            self.token_to_kv_pool.set_kv_data(new_slots, suffix_tokens)

        print(
            f"[Prefill] Req {req_idx} | Prompt={prompt_tokens} | "
            f"命中前缀长度={prefix_len} (复用槽位={matched_kv_slots}) | "
            f"整行映射 ReqToToken[{req_idx}]={self.req_to_token_pool.get_req_slots(req_idx, len(prompt_tokens))}"
        )
        return req_idx, list(prompt_tokens), prefix_len, last_node

    def step_decode(self, req_idx: int, seq_tokens: List[int], new_token: int) -> None:
        """
        阶段 2 (Decode)：
        - 每生成 1 个新 token，向 TokenToKVPool 申请 1 个物理槽位
        - 将该物理槽位追加写入 ReqToTokenPool[req_idx, current_len]
        """
        new_slot = self._alloc_kv_slots_with_evict(1)
        pos = len(seq_tokens)
        self.req_to_token_pool.write(req_idx, pos, new_slot)
        self.token_to_kv_pool.set_kv_data(new_slot, [new_token])
        seq_tokens.append(new_token)
        print(
            f"[Decode]  Req {req_idx} | 生成新 Token={new_token} -> 分配物理槽位 {new_slot[0]} | "
            f"当前整行映射 ReqToToken[{req_idx}]={self.req_to_token_pool.get_req_slots(req_idx, len(seq_tokens))}"
        )

    def finish_request(
        self, req_idx: int, seq_tokens: List[int], prefix_len_at_start: int, last_node: TreeNode
    ) -> None:
        """
        阶段 3 (Finish)：
        - 将该请求整行的 (seq_tokens -> kv_slots) 存入 RadixCache
        - 解锁树节点，释放 ReqToTokenPool 的行，但保留 TokenToKVPool 中的 KV 数据供未来复用
        """
        self.radix_cache.cache_finished_req(req_idx, seq_tokens, prefix_len_at_start, last_node)
        print(f"[Finish]  Req {req_idx} 完成！请求行已释放，KV 槽位已移交 RadixCache 管理。")


if __name__ == "__main__":
    # 创建一个总物理槽位只有 11 个 (有效槽位 1..10，共 10 个可用) 的模拟器
    sim = SGLangMemorySimulator(max_reqs=4, max_context_len=16, max_kv_tokens=11)

    print("=== 步骤 1: 运行请求 A (System Prompt [10, 20, 30] + 用户问题 [40]) ===")
    req_a, tokens_a, plen_a, node_a = sim.start_request_prefill([10, 20, 30, 40])
    sim.step_decode(req_a, tokens_a, new_token=50)
    sim.finish_request(req_a, tokens_a, plen_a, node_a)
    sim.radix_cache.pretty_print()

    print("\n=== 步骤 2: 运行请求 B (共享前缀 [10, 20, 30] + 新问题 [60, 70]) ===")
    req_b, tokens_b, plen_b, node_b = sim.start_request_prefill([10, 20, 30, 60, 70])
    sim.step_decode(req_b, tokens_b, new_token=80)
    # 此时不结束请求 B（保持运行状态，其前缀 [10, 20, 30] 处于 lock_ref=1 锁定状态）
    sim.radix_cache.pretty_print()

    print("\n=== 步骤 3: 运行请求 C (无共享前缀 [91, 92, 93, 94]，触发 LRU 驱逐) ===")
    req_c, tokens_c, plen_c, node_c = sim.start_request_prefill([91, 92, 93, 94])
    sim.radix_cache.pretty_print()

    print("\n=== 步骤 4: 请求 B 和请求 C 相继完成 ===")
    sim.finish_request(req_b, tokens_b, plen_b, node_b)
    sim.finish_request(req_c, tokens_c, plen_c, node_c)
    sim.radix_cache.pretty_print()
