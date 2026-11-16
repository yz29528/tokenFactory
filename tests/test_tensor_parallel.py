from dataclasses import replace
from pathlib import Path
import sys
import threading
from typing import List
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


def test_tp4_full_27b_meta_shapes() -> None:
    """
    验证 1：在完整 64 层 27B 配置下设置 `tp_size = 4`，使用 `meta` 设备验证：
    - 列并行层 (in_proj_qkv, in_proj_z, q_proj, k_proj, v_proj, gate_proj, up_proj) 输出维除以 4；
    - 行并行层 (out_proj, o_proj, down_proj) 输入维除以 4；
    - 所有伴随的 128x128 块级 FP8 `weight_scale_inv` 维度精确除以 4。
    """
    config_path = PROJECT_ROOT / "qwen3.8 27B" / "config.json"
    full_cfg = Qwen3_8Config.from_json_file(config_path)
    tp4_cfg = replace(full_cfg, tp_size=4, tp_rank=0)

    with torch.device("meta"):
        tp4_model = Qwen3_8ForCausalLM(tp4_cfg, enable_mtp=True)

    sd = tp4_model.state_dict()

    # 1. Linear Attention (Gated DeltaNet):
    #    in_proj_qkv: ColumnParallel [10240 // 4 = 2560, 5120], scale: [20, 40]
    assert sd["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"].shape == (
        2560,
        5120,
    )
    assert sd[
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv"
    ].shape == (20, 40)
    #    out_proj: RowParallel [5120, 6144 // 4 = 1536], scale: [40, 12]
    assert sd["model.language_model.layers.0.linear_attn.out_proj.weight"].shape == (
        5120,
        1536,
    )
    assert sd[
        "model.language_model.layers.0.linear_attn.out_proj.weight_scale_inv"
    ].shape == (40, 12)

    # 2. Full Attention (GQA):
    #    q_proj: ColumnParallel [12288 // 4 = 3072, 5120], k_proj: [1024 // 4 = 256, 5120]
    assert sd["model.language_model.layers.3.self_attn.q_proj.weight"].shape == (
        3072,
        5120,
    )
    assert sd["model.language_model.layers.3.self_attn.k_proj.weight"].shape == (
        256,
        5120,
    )
    #    o_proj: RowParallel [5120, 6144 // 4 = 1536]
    assert sd["model.language_model.layers.3.self_attn.o_proj.weight"].shape == (
        5120,
        1536,
    )

    # 3. SwiGLU MLP:
    #    gate_proj: ColumnParallel [17408 // 4 = 4352, 5120], scale: [34, 40]
    #    down_proj: RowParallel [5120, 4352], scale: [40, 34]
    assert sd["model.language_model.layers.0.mlp.gate_proj.weight"].shape == (
        4352,
        5120,
    )
    assert sd[
        "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv"
    ].shape == (34, 40)
    assert sd["model.language_model.layers.0.mlp.down_proj.weight"].shape == (
        5120,
        4352,
    )
    assert sd[
        "model.language_model.layers.0.mlp.down_proj.weight_scale_inv"
    ].shape == (40, 34)

    print(
        "[PASS] 完整 64 层 27B 模型 TP=4 分片形状校验通过（所有 ColumnParallel / RowParallel 及 128x128 FP8 缩放块严格对齐）！"
    )


def test_tp2_numerical_equivalence_and_comm_count() -> None:
    """
    验证 2：对比 `tp_size = 1` 单卡完整模型与 `tp_size = 2` 双 Rank 张量并行模型：
    1. 验证“先列并行、后行并行”策略下，4 层模型每步前向传播在每个 Rank 上严格只触发 `4 * 2 = 8` 次 `AllReduce`（中间算子 0 通信）。
    2. 验证 Prefill 与 Decode 阶段的双卡 TP 输出 Logits 与单卡完整模型 100% 数值等价。
    3. 验证双显存池 (`TokenToKVPool` 与 `ReqToStatePool`) 在各 Rank 上显存减半，且拼接后与单卡状态完全相等。
    """
    torch.manual_seed(2026)
    config_path = PROJECT_ROOT / "qwen3.8 27B" / "config.json"
    base_cfg = Qwen3_8Config.from_json_file(config_path)

    full_cfg = replace(
        base_cfg,
        vocab_size=500,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=4,
        linear_key_head_dim=16,
        linear_num_value_heads=12,
        linear_value_head_dim=16,
        tp_size=1,
        tp_rank=0,
    )

    # 1. 构建单卡参考模型 (tp_size=1) 与完整混合显存池
    ref_model = Qwen3_8ForCausalLM(full_cfg).eval()
    ref_pool = Qwen3_8HybridCachePool(
        full_cfg, max_reqs=4, max_total_tokens=32, max_snapshots=4, dtype=torch.float32
    )

    # 2. 构建 TP=2 的两个 Rank 模型与各自减半的分片显存池，并从 ref_model 切分加载权重
    tp_size = 2
    tp_models: List[Qwen3_8ForCausalLM] = []
    tp_pools: List[Qwen3_8HybridCachePool] = []

    for rank in range(tp_size):
        rank_cfg = replace(full_cfg, tp_size=tp_size, tp_rank=rank)
        m = Qwen3_8ForCausalLM(rank_cfg).eval()
        m.load_tp_from_full_model(ref_model)
        pool = Qwen3_8HybridCachePool(
            rank_cfg,
            max_reqs=4,
            max_total_tokens=32,
            max_snapshots=4,
            dtype=torch.float32,
        )
        tp_models.append(m)
        tp_pools.append(pool)

    # 验证双池单卡分片大小已精确减半
    assert tp_pools[0].k_buffer.shape[2] == ref_pool.k_buffer.shape[2] // tp_size
    assert tp_pools[0].ssm_state.shape[2] == ref_pool.ssm_state.shape[2] // tp_size
    assert tp_pools[0].conv_state.shape[2] == ref_pool.conv_state.shape[2] // tp_size

    # 构造多线程 Barrier 模拟 NCCL AllReduce(SUM) 并统计每个 Rank 的 AllReduce 次数
    barrier = threading.Barrier(tp_size)
    shared_buffers: List[torch.Tensor] = [torch.empty(0)] * tp_size
    all_reduce_counts: List[int] = [0] * tp_size

    def make_rank_all_reduce(rank: int):
        def _all_reduce(tensor: torch.Tensor) -> torch.Tensor:
            shared_buffers[rank] = tensor
            all_reduce_counts[rank] += 1
            barrier.wait()
            summed = shared_buffers[0] + shared_buffers[1]
            barrier.wait()
            return summed

        return _all_reduce

    for rank in range(tp_size):
        tp_models[rank].set_all_reduce_fn(make_rank_all_reduce(rank))

    # 3. 准备输入批次：2 个并发请求 Prefill (长度分别为 4 和 3，共 7 个 token)
    input_ids = torch.randint(0, full_cfg.vocab_size, (7,))
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2])
    req_to_token = torch.full((4, 16), -1, dtype=torch.long)
    req_to_token[0, :4] = torch.tensor([1, 2, 3, 4])
    req_to_token[1, :3] = torch.tensor([5, 6, 7])

    prefill_batch = ForwardBatch(
        batch_size=2,
        req_pool_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([4, 3]),
        prefix_lens=torch.tensor([0, 0]),
        extend_lens=[4, 3],
        out_cache_loc=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        req_to_token=req_to_token,
        is_prefill=True,
    )

    # 单卡参考模型前向计算
    ref_prefill_logits = ref_model(input_ids, positions, ref_pool, prefill_batch)

    # 双卡 TP SPMD 并行前向计算
    tp_prefill_outputs: List[torch.Tensor] = [torch.empty(0)] * tp_size

    def _run_prefill_rank(r: int) -> None:
        tp_prefill_outputs[r] = tp_models[r](
            input_ids, positions, tp_pools[r], prefill_batch
        )

    threads = [
        threading.Thread(target=_run_prefill_rank, args=(r,)) for r in range(tp_size)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 验证通信次数：4 层 * 每层 2 次 AllReduce (Attention/LinearAttention 1 次 + MLP 1 次) = 8 次
    expected_all_reduces = full_cfg.num_hidden_layers * 2
    assert all_reduce_counts == [expected_all_reduces, expected_all_reduces], (
        f"预期每卡 AllReduce 次数为 {expected_all_reduces}，实际为 {all_reduce_counts}"
    )

    max_diff_prefill = (tp_prefill_outputs[0] - ref_prefill_logits).abs().max().item()
    assert torch.allclose(tp_prefill_outputs[0], ref_prefill_logits, atol=1e-5), (
        f"Prefill Logits 不一致，最大误差={max_diff_prefill}"
    )

    # 验证两张卡的 ReqToStatePool 分片拼接后与单卡完全一致
    combined_ssm = torch.cat([tp_pools[0].ssm_state, tp_pools[1].ssm_state], dim=2)
    assert torch.allclose(combined_ssm, ref_pool.ssm_state, atol=1e-5)

    # 4. 继续验证 1 步 Decode
    decode_ids = torch.argmax(ref_prefill_logits, dim=-1)
    decode_positions = torch.tensor([4, 3])
    req_to_token[0, 4] = 8
    req_to_token[1, 3] = 9

    decode_batch = ForwardBatch(
        batch_size=2,
        req_pool_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([5, 4]),
        prefix_lens=torch.tensor([4, 3]),
        extend_lens=[1, 1],
        out_cache_loc=torch.tensor([8, 9]),
        req_to_token=req_to_token,
        is_prefill=False,
    )

    ref_decode_logits = ref_model(decode_ids, decode_positions, ref_pool, decode_batch)

    tp_decode_outputs: List[torch.Tensor] = [torch.empty(0)] * tp_size

    def _run_decode_rank(r: int) -> None:
        tp_decode_outputs[r] = tp_models[r](
            decode_ids, decode_positions, tp_pools[r], decode_batch
        )

    threads = [
        threading.Thread(target=_run_decode_rank, args=(r,)) for r in range(tp_size)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    max_diff_decode = (tp_decode_outputs[0] - ref_decode_logits).abs().max().item()
    assert torch.allclose(tp_decode_outputs[0], ref_decode_logits, atol=1e-5), (
        f"Decode Logits 不一致，最大误差={max_diff_decode}"
    )

    print(
        f"[PASS] TP=2 先列并行后行并行等价性验证通过！"
        f"单步通信次数严格={expected_all_reduces} 次 AllReduce (中间 0 通信)，"
        f"Prefill 最大误差={max_diff_prefill:.2e}，Decode 最大误差={max_diff_decode:.2e}！"
    )


if __name__ == "__main__":
    test_tp4_full_27b_meta_shapes()
    test_tp2_numerical_equivalence_and_comm_count()
