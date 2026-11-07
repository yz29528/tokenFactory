import multiprocessing as mp
from typing import Dict, List, Optional

from qwen3_8 import Qwen3_8Config
from .detokenizer_manager import DetokenizerManager
from .ipc_messages import BatchStrOut, ShutdownSignal, TokenizedReqInput
from .scheduler import Scheduler


class TokenizerManager:
    """
    进程 1 (主进程 TokenizerManager)：
    负责分词、启动 `Scheduler` (带 `chunked_prefill_size` 预算控制) 与 `DetokenizerManager` 子进程，
    并流式接收每个 Step 的混合批次结果。
    """

    def __init__(
        self,
        config: Qwen3_8Config,
        max_reqs: int = 8,
        max_context_len: int = 128,
        max_total_tokens: int = 256,
        chunked_prefill_size: int = 4,
    ):
        self.config = config
        self.max_reqs = max_reqs
        self.max_context_len = max_context_len
        self.max_total_tokens = max_total_tokens
        self.chunked_prefill_size = chunked_prefill_size

        self.piece_to_id: Dict[str, int] = {}
        self.id_to_piece: Dict[int, str] = {}
        self._next_vocab_id: int = 100

        self.ctx = mp.get_context("spawn")
        self.queue_to_scheduler: mp.Queue = self.ctx.Queue()
        self.queue_to_detokenizer: mp.Queue = self.ctx.Queue()
        self.queue_from_detokenizer: mp.Queue = self.ctx.Queue()

        self.scheduler_proc: Optional[mp.Process] = None
        self.detokenizer_proc: Optional[mp.Process] = None
        self.active_req_ids: set[str] = set()

    def encode(self, text: str) -> List[int]:
        tokens: List[int] = []
        for word in text.strip().split():
            if word not in self.piece_to_id:
                tok_id = self._next_vocab_id
                self._next_vocab_id = (self._next_vocab_id + 1) % self.config.vocab_size
                self.piece_to_id[word] = tok_id
                self.id_to_piece[tok_id] = word + " "
            tokens.append(self.piece_to_id[word])
        return tokens

    def start_engine_processes(self) -> None:
        self.scheduler_proc = self.ctx.Process(
            target=Scheduler.run_process_loop,
            kwargs={
                "config": self.config,
                "in_queue": self.queue_to_scheduler,
                "out_queue": self.queue_to_detokenizer,
                "max_reqs": self.max_reqs,
                "max_context_len": self.max_context_len,
                "max_total_tokens": self.max_total_tokens,
                "chunked_prefill_size": self.chunked_prefill_size,
            },
            name="SGLang-Scheduler-Process",
        )
        self.detokenizer_proc = self.ctx.Process(
            target=DetokenizerManager.run_process_loop,
            kwargs={
                "in_queue": self.queue_to_detokenizer,
                "out_queue": self.queue_from_detokenizer,
                "id_to_piece": self.id_to_piece,
            },
            name="SGLang-Detokenizer-Process",
        )
        self.scheduler_proc.start()
        self.detokenizer_proc.start()

    def submit_request(
        self,
        req_id: str,
        prompt: str,
        max_new_tokens: int = 8,
        temperature: float = 0.0,
    ) -> TokenizedReqInput:
        prompt_ids = self.encode(prompt)
        req_msg = TokenizedReqInput(
            req_id=req_id,
            prompt_text=prompt,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.config.eos_token_id,
            temperature=temperature,
        )
        self.active_req_ids.add(req_id)
        self.queue_to_scheduler.put(req_msg)
        return req_msg

    def recv_one_step_output(self) -> Optional[BatchStrOut]:
        if not self.active_req_ids:
            return None

        msg = self.queue_from_detokenizer.get()
        if isinstance(msg, ShutdownSignal):
            return None

        if isinstance(msg, BatchStrOut):
            for req_id, finished in zip(msg.req_ids, msg.finished_flags):
                if finished and req_id in self.active_req_ids:
                    self.active_req_ids.remove(req_id)
            return msg
        return None

    def shutdown(self) -> None:
        self.queue_to_scheduler.put(ShutdownSignal())
        if self.scheduler_proc is not None and self.scheduler_proc.is_alive():
            self.scheduler_proc.join(timeout=5.0)
        if self.detokenizer_proc is not None and self.detokenizer_proc.is_alive():
            self.detokenizer_proc.join(timeout=5.0)
