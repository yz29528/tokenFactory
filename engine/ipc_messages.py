from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TokenizedReqInput:
    """
    IPC 消息 1 (TokenizerManager -> Scheduler):
    携带分词后的 Token ID 序列与采样控制参数。
    """

    req_id: str
    prompt_text: str
    prompt_ids: List[int]
    max_new_tokens: int = 16
    eos_token_id: Optional[int] = None
    temperature: float = 0.0  # 0.0 表示 Greedy 贪心解码


@dataclass
class BatchTokenIDOut:
    """
    IPC 消息 2 (Scheduler -> DetokenizerManager):
    每个 Step 结束后，将本批次中各请求新生成的 Token ID、Chunked Prefill 进度及完成状态发往后处理进程。
    """

    step_id: int
    batch_mode: str  # "PREFILL", "DECODE", 或 "MIXED (CHUNK_PREFILL + DECODE)"
    req_ids: List[str] = field(default_factory=list)
    new_token_ids: List[Optional[int]] = field(
        default_factory=list
    )  # 若某请求本步仅完成中间 Prefill Chunk 尚未产出 Token，则为 None
    finished_flags: List[bool] = field(default_factory=list)
    cached_prefix_lens: List[int] = field(default_factory=list)
    chunk_progress: List[str] = field(default_factory=list)  # 例如 "chunk 4/8" 或 "decode"


@dataclass
class BatchStrOut:
    """
    IPC 消息 3 (DetokenizerManager -> TokenizerManager):
    携带增量反分词后的可读文本片段、Chunk 进度与整句累积输出。
    """

    step_id: int
    batch_mode: str
    req_ids: List[str] = field(default_factory=list)
    delta_texts: List[str] = field(default_factory=list)
    full_texts: List[str] = field(default_factory=list)
    finished_flags: List[bool] = field(default_factory=list)
    cached_prefix_lens: List[int] = field(default_factory=list)
    chunk_progress: List[str] = field(default_factory=list)


@dataclass
class ShutdownSignal:
    """用于优雅关闭子进程的控制信号。"""

    reason: str = "normal_exit"
