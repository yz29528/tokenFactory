from typing import List, Optional


class TokenToKVPool:
    """
    第二层池：维护实际的物理 KV Cache 缓冲区。
    一维槽位索引 (slot_idx) 对应真实的 Key / Value 向量数据。
    同时维护 `free_slots` 空闲栈 (O(1) 分配) 与 `is_occupied` 布尔占用位图 (防止 Double-Free)。
    """

    def __init__(self, max_total_num_tokens: int):
        self.max_total_num_tokens = max_total_num_tokens
        # 模拟 GPU 上的 K/V 张量缓冲池，这里用字符串模拟每个物理槽位里存放的真实 KV 向量
        self.k_buffer: List[Optional[str]] = [None] * max_total_num_tokens
        self.v_buffer: List[Optional[str]] = [None] * max_total_num_tokens
        # 空闲物理槽位栈（slot 0 在 SGLang 中通常保留作 padding，这里从 1 开始）
        self.free_slots: List[int] = list(range(max_total_num_tokens - 1, 0, -1))
        # 显式标记每个物理槽位当前是否被占用（0 号槽位恒为保留状态）
        self.is_occupied: List[bool] = [True] + [False] * (max_total_num_tokens - 1)

    def available_size(self) -> int:
        return len(self.free_slots)

    def alloc(self, num_tokens: int) -> Optional[List[int]]:
        """申请 num_tokens 个物理 KV 槽位，并将其标记为已占用。"""
        if len(self.free_slots) < num_tokens:
            return None
        allocated: List[int] = []
        for _ in range(num_tokens):
            slot = self.free_slots.pop()
            assert not self.is_occupied[slot], f"物理槽位 {slot} 状态异常：分配前已被标记为占用！"
            self.is_occupied[slot] = True
            allocated.append(slot)
        return allocated

    def free(self, slots: List[int]) -> None:
        """当 RadixCache 驱逐节点、发现重复计算的槽位或请求异常中止时，回收物理 KV 槽位。"""
        for s in slots:
            assert self.is_occupied[s], f"检测到 Double-Free：物理槽位 {s} 当前并未被占用！"
            self.is_occupied[s] = False
            self.k_buffer[s] = None
            self.v_buffer[s] = None
            self.free_slots.append(s)

    def set_kv_data(self, slots: List[int], tokens: List[int]) -> None:
        """模拟 Attention 算子将计算出的 K/V 向量写入物理显存槽位。"""
        for slot, token_id in zip(slots, tokens):
            self.k_buffer[slot] = f"K(tok_{token_id})"
            self.v_buffer[slot] = f"V(tok_{token_id})"
