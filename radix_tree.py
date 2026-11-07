import time
from typing import Dict, List, Optional, Tuple

from req_to_token_pool import ReqToTokenPool
from token_to_kv_pool import TokenToKVPool


class TreeNode:
    """SGLang 风格的 Radix Tree 节点，用于管理 LLM 的 KV Cache 前缀复用。"""

    def __init__(self, key: Optional[List[int]] = None, value: Optional[List[int]] = None):
        # children 映射：边上的第一个 token_id -> 子节点 TreeNode
        self.children: Dict[int, "TreeNode"] = {}
        self.parent: Optional["TreeNode"] = None

        # 边上存储的 token 序列（压缩前缀）
        self.key: List[int] = key if key is not None else []
        # 对应的 TokenToKVPool 物理槽位索引列表 (slot indices)
        self.value: List[int] = value if value is not None else []

        # 引用计数：当前有多少正在运行的请求锁定了该节点（>0 时不可被驱逐）
        self.lock_ref: int = 0
        # 最近访问时间戳，用于 LRU 驱逐策略
        self.last_access_time: float = time.monotonic()

    def __len__(self) -> int:
        return len(self.key)


class RadixCache:
    """
    SGLang RadixAttention 中的核心数据结构：RadixCache (基数树)。
    管理 Token 序列到 TokenToKVPool 物理槽位索引的映射与 LRU 回收。
    """

    def __init__(
        self,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool: Optional[TokenToKVPool] = None,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool

        self.root = TreeNode()
        self.root.lock_ref = 1  # 根节点永远不被驱逐
        self.evictable_size_ = 0  # 可驱逐的 token 总数

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
        """
        new_node = TreeNode(
            key=child.key[:split_len],
            value=child.value[:split_len],
        )
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.last_access_time = child.last_access_time

        # 更新原 child 节点的剩余部分
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        child.parent = new_node

        # 建立连接
        new_node.children[child.key[0]] = child
        if new_node.parent is not None:
            new_node.parent.children[new_node.key[0]] = new_node

        return new_node

    def match_prefix(self, key: List[int]) -> Tuple[List[int], TreeNode]:
        """
        查找输入 token 序列 `key` 在 Radix Tree 中的最长匹配前缀。
        如果匹配停在某条边的中间，会自动分裂该节点以便细粒度复用和加锁。

        返回:
            (matched_values, last_matched_node): 匹配到的 KV 槽位列表，以及停靠的最后一个节点
        """
        if not key:
            return [], self.root

        node = self.root
        node.last_access_time = time.monotonic()
        matched_values: List[int] = []
        idx = 0

        while idx < len(key):
            first_token = key[idx]
            if first_token not in node.children:
                break

            child = node.children[first_token]
            child.last_access_time = time.monotonic()

            prefix_len = self._common_prefix_len(child.key, key[idx:])
            if prefix_len < len(child.key):
                # 匹配在边的中间断开，分裂节点
                child = self._split_node(child, prefix_len)

            matched_values.extend(child.value)
            node = child
            idx += prefix_len

        return matched_values, node

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

        node = self.root
        node.last_access_time = time.monotonic()
        idx = 0

        while idx < len(key):
            first_token = key[idx]
            if first_token not in node.children:
                break

            child = node.children[first_token]
            child.last_access_time = time.monotonic()

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

    def cache_finished_req(
        self,
        req_idx: int,
        token_ids: List[int],
        prefix_len_at_start: int,
        last_locked_node: TreeNode,
    ) -> None:
        """
        当一个请求完成生成时调用（对应 SGLang 的 cache_finished_req）：
        1. 从 ReqToTokenPool 取出该请求整行的物理槽位索引
        2. 插入到 RadixCache 中供后续请求复用
        3. 若并发期间已有其它请求插入了相同前缀，释放多余分配的物理槽位
        4. 解锁该请求在树上持有的引用计数，并释放 ReqToTokenPool 中的请求行
        """
        assert self.req_to_token_pool is not None and self.token_to_kv_pool is not None

        kv_indices = self.req_to_token_pool.get_req_slots(req_idx, len(token_ids))
        already_in_tree_len = self.insert(token_ids, kv_indices)

        # 如果树上已存在的长度超过了该请求启动时复用的前缀长度，说明并发运行期间有别的请求
        # 已经把相同的前缀插入了树中，当前请求为这部分重复计算的物理槽位需要立即回收
        if already_in_tree_len > prefix_len_at_start:
            duplicate_slots = kv_indices[prefix_len_at_start:already_in_tree_len]
            self.token_to_kv_pool.free(duplicate_slots)

        # 解锁启动时锁定的前缀节点，并归还第一层池的 Request 行
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
        按 LRU 策略从叶子节点开始驱逐，直到释放至少 `num_tokens` 个 token。
        如果绑定了 TokenToKVPool，会自动通知物理池回收这些槽位。
        """
        evicted_values: List[int] = []
        leaves = self._collect_evictable_leaves()
        leaves.sort(key=lambda n: n.last_access_time, reverse=True)

        while len(evicted_values) < num_tokens and leaves:
            leaf = leaves.pop()
            evicted_values.extend(leaf.value)
            self.evictable_size_ -= len(leaf)

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
        """打印树结构，用于调试展示。"""
        if node is None:
            node = self.root
            print("RadixCache Tree:")
        else:
            prefix = "  " * indent
            print(f"{prefix}├── tokens={node.key}, kv_slots={node.value}, lock={node.lock_ref}")
        for child in node.children.values():
            self.pretty_print(child, indent + 1)
