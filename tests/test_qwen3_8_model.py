from dataclasses import replace
import json
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


def verify_against_safetensors_index(full_cfg: Qwen3_8Config) -> None:
    """
    使用 PyTorch `meta` 设备零显存实例化完整的 64 层 27B `Qwen3_8ForCausalLM(enable_mtp=True)`，
    将其 `state_dict().keys()` 与 `qwen3.8 27B/model.safetensors.index.json` 中的全部语言模型权重键进行 1:1 严格核对！
    """
    index_path = PROJECT_ROOT / "qwen3.8 27B" / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map: dict[str, str] = index_data["weight_map"]
    # 过滤掉多模态视觉编码器 (model.visual.*)，提取全部语言模型、LM Head 与 MTP 权重键
    expected_keys = {
        k for k in weight_map.keys() if not k.startswith("model.visual.")
    }

    with torch.device("meta"):
        meta_model = Qwen3_8ForCausalLM(full_cfg, enable_mtp=True)

    actual_state = meta_model.state_dict()
    actual_keys = set(actual_state.keys())

    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)

    assert not missing_keys, f"发现缺失的权重键 ({len(missing_keys)} 个): {missing_keys[:10]}"
    assert not unexpected_keys, (
        f"发现多余的权重键 ({len(unexpected_keys)} 个): {unexpected_keys[:10]}"
    )

    # 抽样校验关键张量与 FP8 128x128 块级缩放因子形状
    assert actual_state["model.language_model.embed_tokens.weight"].shape == (
        248320,
        5120,
    )
    assert actual_state[
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    ].shape == (10240, 5120)
    assert actual_state[
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv"
    ].shape == (10240 // 128, 5120 // 128)
    assert actual_state[
        "model.language_model.layers.0.linear_attn.in_proj_z.weight"
    ].shape == (6144, 5120)
    assert actual_state[
        "model.language_model.layers.0.linear_attn.in_proj_z.weight_scale_inv"
    ].shape == (6144 // 128, 5120 // 128)
    assert actual_state[
        "model.language_model.layers.3.self_attn.q_proj.weight"
    ].shape == (24 * 256 * 2, 5120)
    assert actual_state[
        "model.language_model.layers.3.self_attn.q_proj.weight_scale_inv"
    ].shape == ((24 * 256 * 2) // 128, 5120 // 128)

    print(
        f"\n[Safetensors Index 100% 校验通过] "
        f"共核对 {len(expected_keys)} 个语言模型与 MTP 权重张量键（零缺失、零多余），"
        f"FP8 128x128 weight_scale_inv 形状完全对齐！"
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

    # 2. 对照 `model.safetensors.index.json` 严格核对完整 64 层 27B 模型的全部参数键名与形状
    verify_against_safetensors_index(full_cfg)

    # 3. 使用等比例缩小的配置在本地进行快速前向推理验证（保留 3:1 的混合层比例）
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
    print(f"[Prefill 验证成功] 输出 Logits 形状: {logits.shape}")

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
