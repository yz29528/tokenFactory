from dataclasses import replace
from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwen3_8 import (
    ForwardBatch,
    Qwen3_8Config,
    Qwen3_8ForCausalLM,
    Qwen3_8HybridCachePool,
)


def main() -> None:
    # 1. 直接读取项目根目录下的 `qwen3.8 27B/config.json`
    config_path = PROJECT_ROOT / "qwen3.8 27B" / "config.json"
    full_cfg = Qwen3_8Config.from_json_file(config_path)

    print("=== 成功读取 `qwen3.8 27B/config.json` ===")
    print(f"模型类型 (model_type)       : {full_cfg.model_type}")
    print(f"词表大小 (vocab_size)       : {full_cfg.vocab_size}")
    print(f"隐藏层维度 (hidden_size)     : {full_cfg.hidden_size}")
    print(f"FFN 维度 (intermediate_size) : {full_cfg.intermediate_size}")
    print(f"总层数 (num_hidden_layers)   : {full_cfg.num_hidden_layers}")
    print(
        f"混合注意力层分布             : "
        f"{len(full_cfg.linear_attention_layer_indices)} 层 linear_attention + "
        f"{len(full_cfg.full_attention_layer_indices)} 层 full_attention"
    )
    print(
        f"Full Attention 参数          : Q_heads={full_cfg.num_attention_heads}, "
        f"KV_heads={full_cfg.num_key_value_heads}, head_dim={full_cfg.head_dim}, "
        f"rotary_dim={full_cfg.rotary_dim}, output_gate={full_cfg.attn_output_gate}"
    )
    print(
        f"Linear Attention 参数        : K_heads={full_cfg.linear_num_key_heads}x{full_cfg.linear_key_head_dim}, "
        f"V_heads={full_cfg.linear_num_value_heads}x{full_cfg.linear_value_head_dim}, "
        f"conv_kernel={full_cfg.linear_conv_kernel_dim}, ssm_dtype={full_cfg.mamba_ssm_dtype}"
    )

    # 2. 使用等比例缩小的配置在本地进行快速前向推理验证（保留 3:1 的混合层比例）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

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

    model = Qwen3_8ForCausalLM(mini_cfg, enable_mtp=True).to(device=device, dtype=dtype)
    cache_pool = Qwen3_8HybridCachePool(
        mini_cfg, max_reqs=4, max_total_tokens=64, device=device, dtype=dtype
    )

    # 模拟 2 个并发请求的 Prefill (长度分别为 5 和 3，总共 8 个 token)
    input_ids = torch.randint(0, mini_cfg.vocab_size, (8,), device=device)
    positions = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], device=device)

    # 构造第一层池 req_to_token 映射表
    req_to_token = torch.full((4, 16), -1, dtype=torch.long, device=device)
    req_to_token[0, :5] = torch.tensor([1, 2, 3, 4, 5], device=device)
    req_to_token[1, :3] = torch.tensor([6, 7, 8], device=device)

    prefill_batch = ForwardBatch(
        batch_size=2,
        req_pool_indices=torch.tensor([0, 1], device=device),
        seq_lens=torch.tensor([5, 3], device=device),
        prefix_lens=torch.tensor([0, 0], device=device),
        extend_lens=[5, 3],
        out_cache_loc=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=device),
        req_to_token=req_to_token,
        is_prefill=True,
    )

    logits = model(input_ids, positions, cache_pool, prefill_batch)
    print(f"\n[Prefill 验证成功] 输出 Logits 形状: {logits.shape}")

    # 模拟 1 步 Decode (每个请求各生成 1 个新 token，共 2 个 token)
    decode_ids = torch.argmax(logits, dim=-1)
    decode_positions = torch.tensor([5, 3], device=device)
    req_to_token[0, 5] = 9
    req_to_token[1, 3] = 10

    decode_batch = ForwardBatch(
        batch_size=2,
        req_pool_indices=torch.tensor([0, 1], device=device),
        seq_lens=torch.tensor([6, 4], device=device),
        prefix_lens=torch.tensor([5, 3], device=device),
        extend_lens=[1, 1],
        out_cache_loc=torch.tensor([9, 10], device=device),
        req_to_token=req_to_token,
        is_prefill=False,
    )

    decode_logits = model(decode_ids, decode_positions, cache_pool, decode_batch)
    print(f"[Decode  验证成功] 输出 Logits 形状: {decode_logits.shape}")


if __name__ == "__main__":
    main()
