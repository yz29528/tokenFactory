import multiprocessing as mp
import queue
from typing import Any, Dict, List, Optional

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
        max_snapshots: int = 16,
        chunked_prefill_size: int = 4,
        tokenizer_path: Optional[str] = None,
        device: str = "cpu",
        enable_mtp: bool = False,
    ):
        if max_context_len < 2:
            raise ValueError(
                f"max_context_len 必须至少为 2（至少容纳 1 个 prompt token + 1 个生成 token），当前为 {max_context_len}"
            )
        self.config = config
        self.max_reqs = max_reqs
        self.max_context_len = max_context_len
        self.max_total_tokens = max_total_tokens
        self.max_snapshots = max_snapshots
        self.chunked_prefill_size = chunked_prefill_size
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.enable_mtp = enable_mtp

        self._hf_tokenizer: Optional[Any] = None
        if tokenizer_path is not None:
            try:
                from tokenizers import Tokenizer

                self._hf_tokenizer = Tokenizer.from_file(tokenizer_path)
            except ImportError:
                self._hf_tokenizer = None

        self.piece_to_id: Dict[str, int] = {}
        self.id_to_piece: Dict[int, str] = {}
        self._pending_vocab_updates: Dict[int, str] = {}
        self._next_vocab_id: int = 100

        self.ctx = mp.get_context("spawn")
        self.queue_to_scheduler: mp.Queue = self.ctx.Queue()
        self.queue_to_detokenizer: mp.Queue = self.ctx.Queue()
        self.queue_from_detokenizer: mp.Queue = self.ctx.Queue()

        self.scheduler_proc: Optional[mp.Process] = None
        self.detokenizer_proc: Optional[mp.Process] = None
        self.active_req_ids: set[str] = set()

    def encode(self, text: str) -> List[int]:
        if self._hf_tokenizer is not None:
            return list(self._hf_tokenizer.encode(text).ids)
        tokens: List[int] = []
        reserved_ids = {self.config.bos_token_id, self.config.eos_token_id}
        if self.config.pad_token_id is not None:
            reserved_ids.add(self.config.pad_token_id)
        for word in text.strip().split():
            if word not in self.piece_to_id:
                attempts = 0
                while (
                    self._next_vocab_id in self.id_to_piece
                    or self._next_vocab_id in reserved_ids
                ) and attempts < self.config.vocab_size:
                    self._next_vocab_id = (
                        self._next_vocab_id + 1
                    ) % self.config.vocab_size
                    attempts += 1
                tok_id = self._next_vocab_id
                self._next_vocab_id = (self._next_vocab_id + 1) % self.config.vocab_size
                piece = word + " "
                self.piece_to_id[word] = tok_id
                self.id_to_piece[tok_id] = piece
                self._pending_vocab_updates[tok_id] = piece
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
                "max_snapshots": self.max_snapshots,
                "chunked_prefill_size": self.chunked_prefill_size,
                "device": self.device,
                "enable_mtp": self.enable_mtp,
            },
            name="SGLang-Scheduler-Process",
        )
        self.detokenizer_proc = self.ctx.Process(
            target=DetokenizerManager.run_process_loop,
            kwargs={
                "in_queue": self.queue_to_detokenizer,
                "out_queue": self.queue_from_detokenizer,
                "id_to_piece": dict(self.id_to_piece),
                "tokenizer_path": self.tokenizer_path,
            },
            name="SGLang-Detokenizer-Process",
        )
        self._pending_vocab_updates.clear()
        self.scheduler_proc.start()
        self.detokenizer_proc.start()

    def submit_request(
        self,
        req_id: str,
        prompt: str,
        max_new_tokens: int = 8,
        temperature: float = 0.0,
    ) -> TokenizedReqInput:
        if req_id in self.active_req_ids:
            raise ValueError(f"请求 ID '{req_id}' 当前仍在运行中，请勿重复提交相同的 req_id！")
        prompt_ids = self.encode(prompt)
        if not prompt_ids:
            raise ValueError(f"请求 {req_id} 分词后为空，无法提交推理！")
        if len(prompt_ids) >= self.max_context_len:
            prompt_ids = prompt_ids[: self.max_context_len - 1]
        if not prompt_ids:
            raise ValueError(f"请求 {req_id} 截断后为空，无法提交推理！")
        max_allowed_new = max(1, self.max_context_len - len(prompt_ids))
        clamped_new_tokens = max(1, min(max_new_tokens, max_allowed_new))

        vocab_updates = dict(self._pending_vocab_updates)
        self._pending_vocab_updates.clear()
        req_msg = TokenizedReqInput(
            req_id=req_id,
            prompt_text=prompt,
            prompt_ids=prompt_ids,
            max_new_tokens=clamped_new_tokens,
            eos_token_id=self.config.eos_token_id,
            temperature=temperature,
            vocab_updates=vocab_updates,
        )
        self.active_req_ids.add(req_id)
        self.queue_to_scheduler.put(req_msg)
        return req_msg

    def recv_one_step_output(self, timeout: float = 30.0) -> Optional[BatchStrOut]:
        if not self.active_req_ids:
            return None

        elapsed = 0.0
        poll_interval = 0.1
        msg = None
        while elapsed < timeout:
            try:
                msg = self.queue_from_detokenizer.get(timeout=poll_interval)
                break
            except queue.Empty:
                elapsed += poll_interval
                if (
                    self.scheduler_proc is not None
                    and not self.scheduler_proc.is_alive()
                    and self.queue_from_detokenizer.empty()
                ):
                    raise RuntimeError("Scheduler 子进程已异常退出！")
                if (
                    self.detokenizer_proc is not None
                    and not self.detokenizer_proc.is_alive()
                    and self.queue_from_detokenizer.empty()
                ):
                    raise RuntimeError("DetokenizerManager 子进程已异常退出！")

        if msg is None or isinstance(msg, ShutdownSignal):
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
            if self.scheduler_proc.is_alive():
                self.scheduler_proc.terminate()
                self.scheduler_proc.join(timeout=1.0)
        if self.detokenizer_proc is not None and self.detokenizer_proc.is_alive():
            # 若 Scheduler 异常退出未能转发 ShutdownSignal，直接向 Detokenizer 队列补发关闭信号并兜底终止
            self.queue_to_detokenizer.put(ShutdownSignal())
            self.detokenizer_proc.join(timeout=5.0)
            if self.detokenizer_proc.is_alive():
                self.detokenizer_proc.terminate()
                self.detokenizer_proc.join(timeout=1.0)
