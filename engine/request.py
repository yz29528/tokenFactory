from typing import List, Optional

from radix_tree import TreeNode


class Request:
    """
    Scheduler 内部维护的单个推理请求状态机 (对应 SGLang 的 `Req` 类)。
    支持混合注意力下的 Chunked Prefill (分块预填充)、Mixed Batching 与状态锚点回退恢复。
    """

    def __init__(
        self,
        req_id: str,
        prompt_text: str,
        prompt_ids: List[int],
        max_new_tokens: int = 16,
        eos_token_id: Optional[int] = None,
        temperature: float = 0.0,
    ):
        self.req_id = req_id
        self.prompt_text = prompt_text
        self.prompt_ids: List[int] = list(prompt_ids)
        self.output_ids: List[int] = []

        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self.temperature = temperature

        # 显存池与 RadixCache 关联状态
        self.req_pool_idx: Optional[int] = None
        # Softmax KV 命中的前缀长度（这些槽位在准入时已写入 req_to_token 并由 last_locked_node 锁定）
        self.prefix_len_at_start: int = 0
        # 线性注意力状态锚点长度（拥有有效 linear_state_slot 的最深祖先位置，prefilled_len 从此处开始推进）
        self.state_anchor_len: int = 0
        self.last_locked_node: Optional[TreeNode] = None

        # Chunked Prefill 进度游标：记录当前 Prompt 已完成全部层（含线性层与全局层）计算的 Token 数量
        self.prefilled_len: int = 0
        self.is_prefilled: bool = False

        # 当前 Step 调度器为该请求切分出的执行片段信息 (供 ModelRunner 拍平组批)
        self.cur_chunk_tokens: List[int] = []
        self.cur_start_pos: int = 0
        self.cur_seq_len: int = 0
        self.out_cache_loc: List[int] = []

    @property
    def remaining_prefill_len(self) -> int:
        """Prompt 中尚未完成 Prefill 计算的剩余 Token 数量。"""
        return len(self.prompt_ids) - self.prefilled_len

    @property
    def is_prefill_complete(self) -> bool:
        """判断整条 Prompt 的所有分块是否均已完成 Prefill。"""
        return self.prefilled_len >= len(self.prompt_ids)

    @property
    def full_token_ids(self) -> List[int]:
        """返回该请求当前全部 Token 序列 (Prompt + 已生成 Output)。"""
        return self.prompt_ids + self.output_ids

    @property
    def seq_len(self) -> int:
        """当前完整序列总长度 (仅在 Prefill 全部完成后用于 Decode 阶段)。"""
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def cached_token_count(self) -> int:
        """当前已在 ReqToTokenPool 中拥有实际 KV 缓存槽位的 Token 数量。"""
        if not self.is_prefilled:
            return max(self.prefix_len_at_start, self.prefilled_len)
        # Decode 阶段最新采样出的最后一个 output token 尚未经过下一轮前向计算 KV
        return len(self.prompt_ids) + max(0, len(self.output_ids) - 1)

    def is_finished(self) -> bool:
        """判断是否满足停止条件（达到 max_new_tokens 或命中 eos_token_id）。"""
        if not self.is_prefilled:
            return False
        if len(self.output_ids) >= self.max_new_tokens:
            return True
        if (
            self.eos_token_id is not None
            and self.output_ids
            and self.output_ids[-1] == self.eos_token_id
        ):
            return True
        return False
