from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple

from req_to_state_pool import ReqToStatePool
from req_to_token_pool import ReqToTokenPool
from token_to_kv_pool import TokenToKVPool


class TreeNode:
    """
    SGLang 风格的混合注意力 Radix Tree 节点：
    1. `value`: 边上每个 Token 对应的 Softmax Full Attention 物理 KV 槽位列表 (`TokenToKVPool`)。
    2. `linear_state_slot`: 到达该节点末尾时的 48 层 Gated DeltaNet 线性注意力状态快照槽位 (`ReqToStatePool`)。
       由于线性状态快照体积大（单份约 147MB）且不可切片，并非所有节点都保存快照（稀疏快照，无快照时为 `None`）。
    """

    def __init__(
        self,
        key: Optional[List[int]] = None,
        value: Optional[List[int]] = None,
        linear_state_slot: Optional[int] = None,
    ):
        # children 映射：边上的第一个 token_id -> 子节点 TreeNode
        self.children: Dict[int, "TreeNode"] = {}
        self.parent: Optional["TreeNode"] = None

        # 边上存储的 token 序列（压缩前缀）
        self.key: List[int] = key if key is not None else []
        # 对应的 TokenToKVPool 物理槽位索引列表 (slot indices)
        self.value: List[int] = value if value is not None else []
        # 到达该节点末尾时的 ReqToStatePool 快照槽位索引 (稀疏挂载，可为 None)
        self.linear_state_slot: Optional[int] = linear_state_slot

        # 引用计数：当前有多少正在运行的请求锁定了该节点（>0 时其 KV 槽位不可被驱逐）
        self.lock_ref: int = 0
        # 最近访问时间戳，用于 LRU 驱逐策略
        self.last_access_time: float = time.monotonic()

    def __len__(self) -> int:
        return len(self.key)


@dataclass
class HybridPrefixMatchResult:
    """
    混合注意力前缀匹配结果：
    - `matched_kv_slots`: 命中的全部 Softmax Full Attention KV 槽位列表 (长度 `kv_prefix_len`)
    - `last_kv_node`: KV 匹配停靠的最深节点 (用于 `inc_lock_ref` 保护整条路径的 KV 槽位)
    - `state_anchor_len`: 匹配路径上拥有有效 `linear_state_slot` 的最深祖先锚点对应的累计 Token 长度
    - `state_anchor_node`: 对应的状态锚点节点 (若路径上均无快照则为 `root`，此时 `state_anchor_len = 0`)
    - `linear_state_slot`: 状态锚点节点的 `linear_state_slot` (若为 `root` 则为 `None`)
    """

    matched_kv_slots: List[int]
    last_kv_node: TreeNode
    state_anchor_len: int
    state_anchor_node: TreeNode
    linear_state_slot: Optional[int]


class RadixCache:
    """
    面向 Qwen3.8-27B 混合注意力架构 (Softmax Full Attention + Gated DeltaNet Linear Attention) 的 RadixCache：
    1. 协同管理两级物理池：`TokenToKVPool` (逐 Token KV 缓存) 与 `ReqToStatePool` (请求工作区 + 树节点稀疏状态快照)。
    2. 稀疏快照与节点分裂保护：`_split_node` 在边中间切分时，中间新节点 `linear_state_slot=None`，原末尾快照保留在子节点上零丢失。
    3. 非对称驱逐法则：
       - `evict_linear_state()`: 当快照槽位耗尽时，按 LRU 仅将冷门节点的 `linear_state_slot` 置回 `None`，保留树结构与 Full Attention `kv_slots`。
       - `evict()`: 当 Softmax KV 槽位耗尽摘除叶子节点时，若该叶子挂有 `linear_state_slot`，必须同步回收其线性状态快照，防止孤儿状态泄漏。
    """

    def __init__(
        self,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool: Optional[TokenToKVPool] = None,
        req_to_state_pool: Optional[ReqToStatePool] = None,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.req_to_state_pool = req_to_state_pool

        self.root = TreeNode()
        self.root.lock_ref = 1  # 根节点永远不被驱逐
        self.evictable_size_ = 0  # 可驱逐的 Softmax KV token 总数

    @staticmethod
    def _common_prefix_len(seq1: List[int], seq2: List[int]) -> int:
        """计算两个 token 序列的最长公共前缀长度。"""
        i = 0
        min_len = min(len(seq1), len(seq2))
        while i < min_len and seq1[i] == seq2[i]:
            i += 1
        return i

    def _split_node(self, child: TreeNode, split_len: int) -> TreeNode:
        """
        将节点 `child` 在 `split_len` 处分裂为两个节点：
        原 parent -> new_node (前 split_len 个 token) -> child (剩余 token)
        注意：
        - `new_node.linear_state_slot = None`（中间分裂点没有对应的历史线性状态快照）
        - `child.linear_state_slot` 保持原值不变（原快照依然精确对应 `child` 末尾位置！）
        """
        new_node = TreeNode(
            key=child.key[:split_len],
            value=child.value[:split_len],
            linear_state_slot=None,
        )
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.last_access_time = child.last_access_time

        # 更新原 child 节点的剩余部分（保留其末尾挂载的 linear_state_slot）
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        child.parent = new_node

        # 建立连接
        new_node.children[child.key[0]] = child
        if new_node.parent is not None:
            new_node.parent.children[new_node.key[0]] = new_node

        return new_node

    def match_prefix_hybrid(self, key: List[int]) -> HybridPrefixMatchResult:
        """
        混合注意力感知的前缀匹配：
        1. 查找输入 token 序列 `key` 在 Radix Tree 中的最长 Softmax KV 匹配前缀（必要时分裂边）。
        2. 同时记录匹配路径上拥有有效 `linear_state_slot` 的最深状态锚点 (`state_anchor_node`, `state_anchor_len`)。
        """
        now = time.monotonic()
        node = self.root
        node.last_access_time = now

        matched_values: List[int] = []
        state_anchor_len: int = 0
        state_anchor_node: TreeNode = self.root
        idx = 0

        while idx < len(key):
            first_token = key[idx]
            if first_token not in node.children:
                break

            child = node.children[first_token]
            child.last_access_time = now

            prefix_len = self._common_prefix_len(child.key, key[idx:])
            if prefix_len < len(child.key):
                # 匹配在边的中间断开，分裂节点（分裂出的 child.linear_state_slot 为 None）
                child = self._split_node(child, prefix_len)

            matched_values.extend(child.value)
            node = child
            idx += prefix_len

            if node.linear_state_slot is not None:
                state_anchor_len = idx
                state_anchor_node = node

        return HybridPrefixMatchResult(
            matched_kv_slots=matched_values,
            last_kv_node=node,
            state_anchor_len=state_anchor_len,
            state_anchor_node=state_anchor_node,
            linear_state_slot=state_anchor_node.linear_state_slot,
        )

    def match_prefix(self, key: List[int]) -> Tuple[List[int], TreeNode]:
        """
        查找输入 token 序列 `key` 在 Radix Tree 中的最长匹配前缀（兼容标准接口）。
        返回:
            (matched_values, last_matched_node): 匹配到的 KV 槽位列表，以及停靠的最后一个节点
        """
        res = self.match_prefix_hybrid(key)
        return res.matched_kv_slots, res.last_kv_node

    def insert(self, key: List[int], value: Optional[List[int]] = None) -> int:
        """
        将一段 token 序列 `key` 及其对应的 KV Cache 槽位 `value` 插入树中。
        返回树中原本已存在的匹配前缀长度。
        """
        if value is None:
            value = list(key)
        assert len(key) == len(value), "key 和 value 的长度必须一致"

        if not key:
            return 0

        now = time.monotonic()
        node = self.root
        node.last_access_time = now
        idx = 0

        while idx < len(key):
            first_token = key[idx]
            if first_token not in node.children:
                break

            child = node.children[first_token]
            child.last_access_time = now

            prefix_len = self._common_prefix_len(child.key, key[idx:])
            if prefix_len == len(child.key):
                node = child
                idx += prefix_len
            else:
                # 部分匹配，分裂节点后跳出循环，插入新分支
                node = self._split_node(child, prefix_len)
                idx += prefix_len
                break

        already_matched_len = idx

        # 如果还有未匹配的后缀，新建叶子节点挂载
        if idx < len(key):
            new_leaf = TreeNode(key=key[idx:], value=value[idx:])
            new_leaf.parent = node
            node.children[key[idx]] = new_leaf
            self.evictable_size_ += len(new_leaf)

        return already_matched_len

    def evict_linear_state(self, num_slots: int = 1) -> List[int]:
        """
        当 `ReqToStatePool` 的快照槽位耗尽时，按 LRU 策略仅回收树节点上的 `linear_state_slot`（置回 `None`），
        不驱逐节点的 Full Attention `kv_slots`，也不改变树的拓扑结构。
        """
        if num_slots <= 0:
            return []

        stateful_nodes: List[TreeNode] = []
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            if node.linear_state_slot is not None:
                stateful_nodes.append(node)
            stack.extend(node.children.values())

        # 优先淘汰无锁节点 (lock_ref == 0)，其次按最近访问时间从旧到新 (LRU)
        stateful_nodes.sort(
            key=lambda n: (n.lock_ref > 0, n.last_access_time),
            reverse=True,
        )

        evicted_state_slots: List[int] = []
        while len(evicted_state_slots) < num_slots and stateful_nodes:
            victim = stateful_nodes.pop()
            slot = victim.linear_state_slot
            assert slot is not None
            victim.linear_state_slot = None
            evicted_state_slots.append(slot)

        if self.req_to_state_pool is not None and evicted_state_slots:
            self.req_to_state_pool.free_snapshots(evicted_state_slots)

        return evicted_state_slots

    def attach_node_linear_state(
        self,
        node: TreeNode,
        req_idx: int,
        debug_tag: Optional[str] = None,
    ) -> Optional[int]:
        """
        将请求 `req_idx` 工作区当前的线性注意力状态保存为树节点 `node` 的只读快照。
        若 `ReqToStatePool` 快照槽位不足，自动调用 `evict_linear_state(1)` 按 LRU 回收旧快照槽位。
        """
        if self.req_to_state_pool is None or node == self.root:
            return None

        if node.linear_state_slot is None:
            if self.req_to_state_pool.available_snapshot_size() < 1:
                self.evict_linear_state(1)
            slot = self.req_to_state_pool.alloc_snapshot()
            if slot is None:
                return None
            node.linear_state_slot = slot

        self.req_to_state_pool.capture_snapshot(
            req_idx=req_idx,
            snapshot_slot=node.linear_state_slot,
            debug_tag=debug_tag,
        )
        node.last_access_time = time.monotonic()
        return node.linear_state_slot

    def cache_unfinished_req(
        self,
        req_idx: int,
        token_ids: List[int],
        prefix_len_at_start: int,
        last_locked_node: TreeNode,
        capture_linear_state: bool = True,
    ) -> Tuple[int, TreeNode]:
        """
        当一个请求刚跑完一个 Prefill Chunk 时调用：
        1. 将当前已算完的 `token_ids` 及其物理 KV 槽位插入 RadixCache。
        2. 若并发期间已有其它请求抢先插入了相同片段，回收重复分配的物理 KV 槽位并指向树中标准槽位。
        3. 稀疏状态快照：默认中间分块 (`capture_linear_state=False`) 不保存线性状态；
           当整条 Prompt 的最后一个 Prefill Chunk 完成 (`capture_linear_state=True`) 时，在 Prompt 边界节点保存线性状态快照。
        """
        assert self.req_to_token_pool is not None and self.token_to_kv_pool is not None

        kv_indices = self.req_to_token_pool.get_req_slots(req_idx, len(token_ids))
        already_in_tree_len = self.insert(token_ids, kv_indices)

        # 如果并发期间已有别的请求插入了重叠区间，释放当前请求多余分配的私有槽位
        if already_in_tree_len > prefix_len_at_start:
            duplicate_slots = kv_indices[prefix_len_at_start:already_in_tree_len]
            self.token_to_kv_pool.free(duplicate_slots)

        # 重新在树上匹配当前前缀，获取树中标准槽位与最新停靠节点
        tree_slots, new_last_node = self.match_prefix(token_ids)
        if already_in_tree_len > prefix_len_at_start:
            self.req_to_token_pool.write(
                req_idx,
                prefix_len_at_start,
                tree_slots[prefix_len_at_start:already_in_tree_len],
            )

        # 若到达关键边界（如 Prompt Prefill 完成点），挂载/自愈该节点的线性注意力状态快照
        if capture_linear_state and self.req_to_state_pool is not None:
            self.attach_node_linear_state(
                new_last_node,
                req_idx,
                debug_tag=f"LinearState(len={len(token_ids)})",
            )

        # 先锁新路径、再解旧路径，保证已缓存槽位全程受保护
        self.inc_lock_ref(new_last_node)
        self.dec_lock_ref(last_locked_node)

        return len(token_ids), new_last_node

    def cache_finished_req(
        self,
        req_idx: int,
        token_ids: List[int],
        prefix_len_at_start: int,
        last_locked_node: TreeNode,
        capture_linear_state: bool = True,
    ) -> None:
        """
        当一个请求完成生成时调用：
        1. 从 ReqToTokenPool 取出该请求整行的物理槽位索引插入 RadixCache
        2. 释放并发重复分配的物理 KV 槽位
        3. 若启用了 `capture_linear_state` 且绑定了 `ReqToStatePool`，在终点节点保存线性注意力状态快照
        4. 解锁该请求在树上持有的引用计数，并释放 ReqToTokenPool 中的请求行
        """
        assert self.req_to_token_pool is not None and self.token_to_kv_pool is not None

        kv_indices = self.req_to_token_pool.get_req_slots(req_idx, len(token_ids))
        already_in_tree_len = self.insert(token_ids, kv_indices)

        if already_in_tree_len > prefix_len_at_start:
            duplicate_slots = kv_indices[prefix_len_at_start:already_in_tree_len]
            self.token_to_kv_pool.free(duplicate_slots)

        if capture_linear_state and self.req_to_state_pool is not None and token_ids:
            _, end_node = self.match_prefix(token_ids)
            self.attach_node_linear_state(
                end_node,
                req_idx,
                debug_tag=f"LinearState(len={len(token_ids)})",
            )

        self.dec_lock_ref(last_locked_node)
        self.req_to_token_pool.free_req(req_idx)

    def abort_req(
        self,
        req_idx: int,
        allocated_seq_len: int,
        prefix_len_at_start: int,
        last_locked_node: TreeNode,
    ) -> None:
        """
        当一个正在运行的请求被异常取消/中止（Abort）且不希望将其未完成部分挂入树中时调用：
        1. 将该请求在 `[prefix_len_at_start : allocated_seq_len]` 区间内尚未入树的私有物理槽位直接归还 `TokenToKVPool`。
        2. 解锁其在 `RadixCache` 上锁定的前缀节点 `last_locked_node`。
        3. 释放其在 `ReqToTokenPool` 中占用的行。
        """
        assert self.req_to_token_pool is not None and self.token_to_kv_pool is not None

        if allocated_seq_len > prefix_len_at_start:
            all_slots = self.req_to_token_pool.get_req_slots(req_idx, allocated_seq_len)
            uninserted_slots = [
                s for s in all_slots[prefix_len_at_start:allocated_seq_len] if s >= 0
            ]
            if uninserted_slots:
                self.token_to_kv_pool.free(uninserted_slots)

        self.dec_lock_ref(last_locked_node)
        self.req_to_token_pool.free_req(req_idx)

    def inc_lock_ref(self, node: TreeNode) -> None:
        """从该节点向上遍历到根节点增加引用计数，防止正在使用的 KV 槽位被驱逐。"""
        curr = node
        while curr is not None and curr != self.root:
            if curr.lock_ref == 0:
                self.evictable_size_ -= len(curr)
            curr.lock_ref += 1
            curr = curr.parent

    def dec_lock_ref(self, node: TreeNode) -> None:
        """请求完成或释放时，从该节点向上递减引用计数。"""
        curr = node
        while curr is not None and curr != self.root:
            curr.lock_ref -= 1
            if curr.lock_ref == 0:
                self.evictable_size_ += len(curr)
            curr = curr.parent

    def _collect_evictable_leaves(self) -> List[TreeNode]:
        """收集所有可被驱逐的叶子节点（无子节点且 lock_ref == 0）。"""
        leaves = []
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            if not node.children:
                if node.lock_ref == 0:
                    leaves.append(node)
            else:
                stack.extend(node.children.values())
        return leaves

    def evict(self, num_tokens: int) -> List[int]:
        """
        按 LRU 策略从叶子节点开始驱逐 Softmax KV Cache，直到释放至少 `num_tokens` 个 token。
        【非对称驱逐联动】：当某个叶子节点的 Softmax KV Cache 被驱逐时，
        若其上挂载了 `linear_state_slot`，必须同步回收该线性状态快照槽位，防止产生孤儿状态泄漏！
        """
        evicted_values: List[int] = []
        leaves = self._collect_evictable_leaves()
        leaves.sort(key=lambda n: n.last_access_time, reverse=True)

        while len(evicted_values) < num_tokens and leaves:
            leaf = leaves.pop()
            evicted_values.extend(leaf.value)
            self.evictable_size_ -= len(leaf)

            # 同步回收该叶子节点上挂载的线性注意力快照槽位
            if leaf.linear_state_slot is not None:
                if self.req_to_state_pool is not None:
                    self.req_to_state_pool.free_snapshot(leaf.linear_state_slot)
                leaf.linear_state_slot = None

            parent = leaf.parent
            if parent is not None:
                del parent.children[leaf.key[0]]
                if not parent.children and parent != self.root and parent.lock_ref == 0:
                    leaves.append(parent)
                    leaves.sort(key=lambda n: n.last_access_time, reverse=True)

        if self.token_to_kv_pool is not None and evicted_values:
            self.token_to_kv_pool.free(evicted_values)

        return evicted_values

    def pretty_print(self, node: Optional[TreeNode] = None, indent: int = 0) -> None:
        """打印树结构（含 Softmax KV 槽位与 Linear State 快照槽位），用于调试展示。"""
        if node is None:
            node = self.root
            print("RadixCache Tree:")
        else:
            prefix = "  " * indent
            print(
                f"{prefix}├── tokens={node.key}, kv_slots={node.value}, "
                f"linear_state={node.linear_state_slot}, lock={node.lock_ref}"
            )
        for child in node.children.values():
            self.pretty_print(child, indent + 1)
