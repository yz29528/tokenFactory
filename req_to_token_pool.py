from typing import List, Optional


class ReqToTokenPool:
    """
    第一层池：维护 [max_reqs, max_context_len] 的二维映射表。
    每一行代表一个正在运行的 Request，列元素保存 TokenToKVPool 中的物理槽位索引 (slot_idx)。
    """

    def __init__(self, max_reqs: int, max_context_len: int):
        self.max_reqs = max_reqs
        self.max_context_len = max_context_len
        # 使用 -1 表示未分配的空槽位
        self.req_to_token: List[List[int]] = [
            [-1] * max_context_len for _ in range(max_reqs)
        ]
        # 记录空闲的 request 行号
        self.free_req_slots: List[int] = list(range(max_reqs - 1, -1, -1))

    def alloc_req(self) -> Optional[int]:
        """为一个新请求分配一行 (req_pool_idx)。"""
        if not self.free_req_slots:
            return None
        return self.free_req_slots.pop()

    def write(self, req_idx: int, start_pos: int, kv_slots: List[int]) -> None:
        """将分配或复用得到的物理 KV 槽位索引写入该请求对应的行中。"""
        for offset, slot in enumerate(kv_slots):
            self.req_to_token[req_idx][start_pos + offset] = slot

    def get_req_slots(self, req_idx: int, seq_len: int) -> List[int]:
        """获取该请求前 seq_len 个 token 对应的所有物理 KV 槽位索引。"""
        return self.req_to_token[req_idx][:seq_len]

    def free_req(self, req_idx: int) -> None:
        """
        释放请求占用的行（注意：只回收行号，不直接释放底层的 KV 槽位，
        因为底层 KV 槽位会移交给 RadixCache 进行缓存复用）。
        """
        self.req_to_token[req_idx] = [-1] * self.max_context_len
        self.free_req_slots.append(req_idx)
