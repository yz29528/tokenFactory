from typing import Dict, List, Optional, Tuple


class RadixNode:

    def __init__(self):
        self.children: Dict[int, "RadixNode"] = {}  # token_id -> child node
        self.block_ids: List[int] = []  # 该节点对应的物理块
        self.token_ids: List[int] = []  # 压缩的一段 token 序列
        self.ref_count: int = 0  # 引用计数（活跃请求数）
        self.last_access_time: float = 0.0


class RadixCache:

    def __init__(self, memory_pool: KVCachePool):
        self.root = RadixNode()
        self.pool = memory_pool

    def match_prefix(self, token_ids: List[int]) -> Tuple[int, List[int]]:
        """查找已有前缀，返回匹配长度和复用的 block_ids"""
        curr = self.root
        matched_tokens = 0
        reused_blocks = []

        idx = 0
        while idx < len(token_ids):
            first_tok = token_ids[idx]
            if first_tok not in curr.children:
                break
            child = curr.children[first_tok]

            # 比对压缩路径上的 token
            seq_len = len(child.token_ids)
            if token_ids[idx : idx + seq_len] == child.token_ids:
                matched_tokens += seq_len
                reused_blocks.extend(child.block_ids)
                idx += seq_len
                curr = child
            else:
                # 前缀部分匹配分支（更高级的实现可在这里做分裂 Split）
                break

        return matched_tokens, reused_blocks

    def insert(self, token_ids: List[int], block_ids: List[int]):
        """简化的插入逻辑：将新生成的 tokens 和 blocks 挂到树上"""
        curr = self.root
        # 实际工程中需按 block_size 边界分段建树
        new_node = RadixNode()
        new_node.token_ids = token_ids
        new_node.block_ids = block_ids
        curr.children[token_ids[0]] = new_node