import multiprocessing as mp
from typing import Dict, List

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

    def __init__(self, id_to_piece: Dict[int, str] | None = None):
        self.id_to_piece: Dict[int, str] = id_to_piece or {}
        self.req_decoded_history: Dict[str, List[str]] = {}

    def _decode_token(self, token_id: int) -> str:
        if token_id in self.id_to_piece:
            return self.id_to_piece[token_id]
        return f"<tok_{token_id}>"

    def handle_batch_token_out(self, msg: BatchTokenIDOut) -> BatchStrOut:
        str_out = BatchStrOut(
            step_id=msg.step_id,
            batch_mode=msg.batch_mode,
            req_ids=list(msg.req_ids),
            finished_flags=list(msg.finished_flags),
            cached_prefix_lens=list(msg.cached_prefix_lens),
            chunk_progress=list(msg.chunk_progress),
        )

        for req_id, tok_id, finished in zip(
            msg.req_ids, msg.new_token_ids, msg.finished_flags
        ):
            history = self.req_decoded_history.setdefault(req_id, [])

            if tok_id is None:
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
    ) -> None:
        detokenizer = DetokenizerManager(id_to_piece=id_to_piece)

        while True:
            msg = in_queue.get()
            if isinstance(msg, ShutdownSignal):
                out_queue.put(msg)
                break

            if isinstance(msg, BatchTokenIDOut):
                str_out = detokenizer.handle_batch_token_out(msg)
                out_queue.put(str_out)
