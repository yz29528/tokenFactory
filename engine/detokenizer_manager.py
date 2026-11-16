import multiprocessing as mp
from typing import Any, Dict, List, Optional

from .ipc_messages import BatchStrOut, BatchTokenIDOut, ShutdownSignal


class DetokenizerManager:
    """
    进程 3 (DetokenizerManager 进程)：
    职责：
    1. 从 `Scheduler` 进程接收 `BatchTokenIDOut`。
    2. 对于产出新 Token 的请求（Decode 步或最后一个 Prefill Chunk），解码并累积输出文本；
       对于仅完成中间 Prefill Chunk 的请求（`tok_id is None`），标记为 `<prefill_chunk_only>`。
    3. 将 `BatchStrOut` 发送回 `TokenizerManager`。
    """

    def __init__(
        self,
        id_to_piece: Dict[int, str] | None = None,
        tokenizer_path: Optional[str] = None,
    ):
        self.id_to_piece: Dict[int, str] = id_to_piece or {}
        self.req_decoded_history: Dict[str, List[str]] = {}
        self.req_token_ids_history: Dict[str, List[int]] = {}
        self.req_emitted_text: Dict[str, str] = {}
        self._hf_tokenizer: Optional[Any] = None
        if tokenizer_path is not None:
            try:
                from tokenizers import Tokenizer

                self._hf_tokenizer = Tokenizer.from_file(tokenizer_path)
            except ImportError:
                self._hf_tokenizer = None

    def _decode_token(self, token_id: int) -> str:
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.decode([token_id], skip_special_tokens=False)
        if token_id in self.id_to_piece:
            return self.id_to_piece[token_id]
        return f"<tok_{token_id}>"

    def handle_batch_token_out(self, msg: BatchTokenIDOut) -> BatchStrOut:
        if msg.vocab_updates:
            self.id_to_piece.update(msg.vocab_updates)

        str_out = BatchStrOut(
            step_id=msg.step_id,
            batch_mode=msg.batch_mode,
            req_ids=list(msg.req_ids),
            finished_flags=list(msg.finished_flags),
            cached_prefix_lens=list(msg.cached_prefix_lens),
            chunk_progress=list(msg.chunk_progress),
        )

        multi_tokens = msg.multi_token_ids or [None] * len(msg.req_ids)
        for req_id, tok_id, extra_toks, finished in zip(
            msg.req_ids, msg.new_token_ids, multi_tokens, msg.finished_flags
        ):
            if self._hf_tokenizer is not None:
                tok_hist = self.req_token_ids_history.setdefault(req_id, [])
                prev_emitted = self.req_emitted_text.get(req_id, "")
                if extra_toks:
                    tok_hist.extend(extra_toks)
                elif tok_id is not None:
                    tok_hist.append(tok_id)

                if extra_toks or tok_id is not None:
                    decoded = self._hf_tokenizer.decode(
                        tok_hist, skip_special_tokens=False
                    )
                    stable_text = (
                        decoded.rstrip("\ufffd")
                        if (not finished and decoded.endswith("\ufffd"))
                        else decoded
                    )
                    delta = (
                        stable_text[len(prev_emitted) :]
                        if stable_text.startswith(prev_emitted)
                        else stable_text
                    )
                    self.req_emitted_text[req_id] = stable_text
                    str_out.delta_texts.append(delta)
                    str_out.full_texts.append(stable_text)
                else:
                    str_out.delta_texts.append("[chunk_only]")
                    str_out.full_texts.append(prev_emitted)

                if finished:
                    self.req_token_ids_history.pop(req_id, None)
                    self.req_emitted_text.pop(req_id, None)
                continue

            history = self.req_decoded_history.setdefault(req_id, [])

            if extra_toks:
                pieces = [self._decode_token(t) for t in extra_toks]
                history.extend(pieces)
                str_out.delta_texts.append("".join(pieces))
                str_out.full_texts.append("".join(history))
            elif tok_id is None:
                # 处于中间 Chunked Prefill 阶段，尚未产出生成 Token
                str_out.delta_texts.append("[chunk_only]")
                str_out.full_texts.append("".join(history))
            else:
                piece = self._decode_token(tok_id)
                history.append(piece)
                str_out.delta_texts.append(piece)
                str_out.full_texts.append("".join(history))

            if finished:
                self.req_decoded_history.pop(req_id, None)

        return str_out

    @staticmethod
    def run_process_loop(
        in_queue: mp.Queue,
        out_queue: mp.Queue,
        id_to_piece: Dict[int, str] | None = None,
        tokenizer_path: Optional[str] = None,
    ) -> None:
        detokenizer = DetokenizerManager(
            id_to_piece=id_to_piece, tokenizer_path=tokenizer_path
        )

        while True:
            msg = in_queue.get()
            if isinstance(msg, ShutdownSignal):
                out_queue.put(msg)
                break

            if isinstance(msg, BatchTokenIDOut):
                str_out = detokenizer.handle_batch_token_out(msg)
                out_queue.put(str_out)
