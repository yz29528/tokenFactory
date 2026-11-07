from dataclasses import replace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine import TokenizerManager
from qwen3_8 import Qwen3_8Config


def main() -> None:
    # 1. 读取 `qwen3.8 27B/config.json` 并构造轻量验证配置（保留 3:1 混合注意力架构）
    config_path = PROJECT_ROOT / "qwen3.8 27B" / "config.json"
    full_cfg = Qwen3_8Config.from_json_file(config_path)

    mini_cfg = replace(
        full_cfg,
        vocab_size=2000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        num_attention_heads=6,
        num_key_value_heads=1,
        head_dim=64,
        linear_num_key_heads=4,
        linear_key_head_dim=32,
        linear_num_value_heads=12,
        linear_value_head_dim=32,
    )

    # 2. 初始化主进程 TokenizerManager，设置 chunked_prefill_size=4（每步最多处理 4 个 Token）
    tok_manager = TokenizerManager(
        config=mini_cfg,
        max_reqs=8,
        max_context_len=64,
        max_total_tokens=128,
        chunked_prefill_size=4,
    )
    tok_manager.start_engine_processes()

    try:
        print("=== 阶段 1: 并发提交 Req-A (8 个词, max_new_tokens=2) 与 Req-B (9 个词, max_new_tokens=5) ===")
        print("    (因 chunked_prefill_size=4，长 Prompt 会自动拆分为多个 Chunk 分步预填充)")
        tok_manager.submit_request(
            req_id="Req-A",
            prompt="system you are a helpful assistant user hello",
            max_new_tokens=2,
        )
        tok_manager.submit_request(
            req_id="Req-B",
            prompt="system you are a helpful assistant user explain radix",
            max_new_tokens=5,
        )

        # 持续运行直到短请求 Req-A 完成并退出（此时 Req-A 归档至 RadixCache，而 Req-B 仍在运行）
        while "Req-A" in tok_manager.active_req_ids:
            step_out = tok_manager.recv_one_step_output()
            if step_out is not None:
                print(
                    f"[Step {step_out.step_id:02d} | {step_out.batch_mode:13s}] "
                    f"请求={step_out.req_ids} | 进度={step_out.chunk_progress} | "
                    f"命中前缀={step_out.cached_prefix_lens} | 输出={step_out.delta_texts} | 完成={step_out.finished_flags}"
                )

        print("\n=== 阶段 2: Req-A 已退出并写入 RadixCache，中途动态插入长请求 Req-C (12 个词，命中前 6 个公共前缀词) ===")
        print("    (观察 MIXED 批次：Req-B 每步消耗 1 个预算做 Decode，剩余 3 个预算同时给 Req-C 做 Chunked Prefill！)")
        tok_manager.submit_request(
            req_id="Req-C",
            prompt="system you are a helpful assistant user what is chunked prefill scheduling",
            max_new_tokens=3,
        )

        while tok_manager.active_req_ids:
            step_out = tok_manager.recv_one_step_output()
            if step_out is None:
                break
            print(
                f"[Step {step_out.step_id:02d} | {step_out.batch_mode:13s}] "
                f"请求={step_out.req_ids} | 进度={step_out.chunk_progress} | "
                f"命中前缀={step_out.cached_prefix_lens} | 输出={step_out.delta_texts} | 完成={step_out.finished_flags}"
            )

    finally:
        tok_manager.shutdown()
        print("\n=== 三进程引擎已优雅关闭 ===")


if __name__ == "__main__":
    main()
