from dataclasses import replace
from pathlib import Path
import sys

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radix_tree import RadixCache
from req_to_state_pool import ReqToStatePool
from req_to_token_pool import ReqToTokenPool
from token_to_kv_pool import TokenToKVPool


def test_asymmetric_eviction_and_state_healing() -> None:
    """
    验证：
    1. 当 `ReqToStatePool` 快照槽位耗尽时，`evict_linear_state` 仅按 LRU 将最旧节点的 `linear_state_slot` 置为 `None`，
       而不驱逐其 Softmax `kv_slots`，也不破坏树结构。
    2. 当后续请求命中 `linear_state_slot=None` 的节点时，`match_prefix_hybrid` 自动回退到最近的有状态祖先锚点，
       同时保留整条路径的 `matched_kv_slots` 供免分配复用。
    3. 当 `TokenToKVPool` 槽位耗尽触发 `evict()` 删除叶子节点时，若该叶子挂有 `linear_state_slot`，必须同步回收至 `ReqToStatePool`。
    """
    req_pool = ReqToTokenPool(max_reqs=4, max_context_len=16)
    kv_pool = TokenToKVPool(max_total_num_tokens=16)
    # 仅设置 2 个快照槽位 (槽位号 4, 5)
    state_pool = ReqToStatePool(max_reqs=4, max_snapshots=2)
    cache = RadixCache(req_pool, kv_pool, state_pool)

    # 1. 插入前缀节点 A: [1, 2, 3] 并挂载快照 #1
    req_a = req_pool.alloc_req()
    assert req_a is not None
    slots_a = kv_pool.alloc(3)
    assert slots_a is not None
    req_pool.write(req_a, 0, slots_a)
    state_pool.state_debug_tags[req_a] = "State_A([1,2,3])"
    _, node_a = cache.cache_unfinished_req(
        req_idx=req_a,
        token_ids=[1, 2, 3],
        prefix_len_at_start=0,
        last_locked_node=cache.root,
        capture_linear_state=True,
    )
    assert node_a.linear_state_slot is not None
    snap_a = node_a.linear_state_slot

    # 延伸请求 A 到叶子 B: [1, 2, 3, 4, 5] 并完成归档 (挂载快照 #2)
    slots_b = kv_pool.alloc(2)
    assert slots_b is not None
    req_pool.write(req_a, 3, slots_b)
    state_pool.state_debug_tags[req_a] = "State_B([1,2,3,4,5])"
    cache.cache_finished_req(
        req_idx=req_a,
        token_ids=[1, 2, 3, 4, 5],
        prefix_len_at_start=3,
        last_locked_node=node_a,
        capture_linear_state=True,
    )
    _, node_b = cache.match_prefix([1, 2, 3, 4, 5])
    assert node_b.linear_state_slot is not None
    assert state_pool.available_snapshot_size() == 0  # 2 个快照槽位已满！

    # 访问一次 node_a 使 node_a 比 node_b 更新，从而让 node_b 成为最旧的 LRU 快照节点
    cache.match_prefix([1, 2, 3])

    # 2. 新请求 C 插入分支 [1, 2, 3, 6, 7] 并请求挂载第 3 个快照 -> 自动触发 evict_linear_state(1)
    req_c = req_pool.alloc_req()
    assert req_c is not None
    slots_c = kv_pool.alloc(2)
    assert slots_c is not None
    req_pool.write(req_c, 0, slots_a)
    req_pool.write(req_c, 3, slots_c)
    cache.inc_lock_ref(node_a)
    state_pool.state_debug_tags[req_c] = "State_C([1,2,3,6,7])"
    cache.cache_finished_req(
        req_idx=req_c,
        token_ids=[1, 2, 3, 6, 7],
        prefix_len_at_start=3,
        last_locked_node=node_a,
        capture_linear_state=True,
    )

    # 验证：node_b 的 linear_state_slot 已被 LRU 剥离置为 None，但 node_b 的 kv_slots 仍然完整保留在树上！
    assert node_b.linear_state_slot is None, "node_b 的线性快照应已被 LRU 回收置为 None"
    assert node_b.value == slots_b, "node_b 的 Softmax KV 槽位不应被驱逐"
    assert node_a.linear_state_slot == snap_a, "热点公共前缀 node_a 的线性快照应保留"

    # 3. 此时若有新请求匹配 [1, 2, 3, 4, 5]：
    #    KV 匹配长度为 5 (node_b)，但状态锚点自动回退到长度 3 (node_a)！
    hybrid_res = cache.match_prefix_hybrid([1, 2, 3, 4, 5])
    assert len(hybrid_res.matched_kv_slots) == 5
    assert hybrid_res.state_anchor_len == 3
    assert hybrid_res.linear_state_slot == snap_a

    # 4. 验证 Softmax KV 驱逐时同步释放挂载的线性状态快照
    _, node_c = cache.match_prefix([1, 2, 3, 6, 7])
    assert node_c.linear_state_slot is not None
    # 触发驱逐整棵树的所有可驱逐 KV (共 7 个 token)
    cache.evict(7)
    assert state_pool.available_snapshot_size() == 2, (
        "所有 KV 节点被驱逐后，其上挂载的线性快照槽位必须全部同步归还至 ReqToStatePool！"
    )
    print("[PASS] 双池非对称驱逐与状态锚点回退验证通过！")


def test_hybrid_scheduler_prefix_reuse_exact_equivalence() -> None:
    """
    验证：在 Qwen3.8 混合注意力模型 (3x Linear Attention + 1x Full Attention) 下，
    命中 RadixCache 前缀的请求 Req-2（恢复线性状态快照并跳过前缀计算）与
    完全不经过缓存从头计算整条 Prompt 的参考模型输出 **100% 数值一致**！
    """
    if torch is None:
        print("[SKIP] 当前 Python 环境未安装 torch，跳过算子级张量等价性测试。")
        return

    from engine.scheduler import Scheduler
    from qwen3_8 import Qwen3_8Config

    torch.manual_seed(42)
    config_path = PROJECT_ROOT / "qwen3.8 27B" / "config.json"
    full_cfg = Qwen3_8Config.from_json_file(config_path)
    mini_cfg = replace(
        full_cfg,
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
        num_key_value_heads=1,
        head_dim=32,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_num_value_heads=6,
        linear_value_head_dim=16,
    )

    scheduler = Scheduler(
        config=mini_cfg,
        max_reqs=4,
        max_context_len=32,
        max_total_tokens=64,
        max_snapshots=4,
        chunked_prefill_size=4,
        device="cpu",
    )

    # 1. 先运行请求 1: prompt = [10, 20, 30, 40, 50, 60], max_new_tokens = 2
    from engine.request import Request

    req1 = Request(
        req_id="req-1",
        prompt_text="p1",
        prompt_ids=[10, 20, 30, 40, 50, 60],
        max_new_tokens=2,
        temperature=0.0,
    )
    scheduler.waiting_queue.append(req1)
    while scheduler.waiting_queue or scheduler.inflight_prefill_queue or scheduler.running_batch:
        scheduler.step()

    # 此时树中在 [10, 20, 30, 40, 50, 60] 节点处已有保存好的线性注意力状态快照！
    match_check = scheduler.radix_cache.match_prefix_hybrid([10, 20, 30, 40, 50, 60])
    assert match_check.state_anchor_len == 6
    assert match_check.linear_state_slot is not None

    # 2. 提交共享前 6 个 Token 的请求 2: prompt = [10, 20, 30, 40, 50, 60, 70, 80]
    target_prompt = [10, 20, 30, 40, 50, 60, 70, 80]
    req2 = Request(
        req_id="req-2",
        prompt_text="p2",
        prompt_ids=target_prompt,
        max_new_tokens=1,
        temperature=0.0,
    )
    scheduler.waiting_queue.append(req2)
    step_out = scheduler.step()
    assert step_out is not None
    assert step_out.cached_prefix_lens == [6], (
        f"预期命中 6 个 Token 的混合状态锚点，实际为 {step_out.cached_prefix_lens}"
    )
    cached_first_token = req2.output_ids[0]

    # 3. 构造完全无缓存的独立 Baseline：从零开始一次算完 [10, 20, 30, 40, 50, 60, 70, 80]
    fresh_scheduler = Scheduler(
        config=mini_cfg,
        max_reqs=4,
        max_context_len=32,
        max_total_tokens=64,
        max_snapshots=4,
        chunked_prefill_size=16,  # 一步算完整个 Prompt
        device="cpu",
    )
    fresh_scheduler.model_runner.model.load_state_dict(
        scheduler.model_runner.model.state_dict()
    )
    req_fresh = Request(
        req_id="req-fresh",
        prompt_text="p2",
        prompt_ids=target_prompt,
        max_new_tokens=1,
        temperature=0.0,
    )
    fresh_scheduler.waiting_queue.append(req_fresh)
    fresh_scheduler.step()
    fresh_first_token = req_fresh.output_ids[0]

    assert cached_first_token == fresh_first_token, (
        f"前缀缓存复用结果 ({cached_first_token}) 与全量重算结果 ({fresh_first_token}) 不一致！"
    )
    print(
        f"[PASS] 混合注意力前缀复用数值等价性验证通过！命中 6 Token 锚点后采样 Token={cached_first_token} 与全量重算完全一致！"
    )


def test_sparse_snapshot_skips_intermediate_chunks() -> None:
    """
    验证：
    1. Chunked Prefill 期间，中间分块不会触发线性状态快照捕获（不消耗 ReqToStatePool 快照槽位）。
    2. 仅在 Prompt 预填充完成点 (is_prefill_complete) 与请求结束点 (_retire_request) 挂载快照。
    3. 当到达失去线性快照的分支边界点 (prefilled_len == prefix_len_at_start 且 linear_state_slot is None) 时，
       仍能触发状态自愈。
    """
    if torch is None:
        return

    from engine.request import Request
    from engine.scheduler import Scheduler
    from qwen3_8 import Qwen3_8Config

    torch.manual_seed(0)
    config_path = PROJECT_ROOT / "qwen3.8 27B" / "config.json"
    full_cfg = Qwen3_8Config.from_json_file(config_path)
    mini_cfg = replace(
        full_cfg,
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
        num_key_value_heads=1,
        head_dim=32,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_num_value_heads=6,
        linear_value_head_dim=16,
    )

    scheduler = Scheduler(
        config=mini_cfg,
        max_reqs=4,
        max_context_len=32,
        max_total_tokens=64,
        max_snapshots=4,
        chunked_prefill_size=3,
        device="cpu",
    )

    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    req = Request(
        req_id="req-sparse",
        prompt_text="sparse",
        prompt_ids=prompt,
        max_new_tokens=2,
        temperature=0.0,
    )
    scheduler.waiting_queue.append(req)

    # 前 3 步为中间分块 (3, 6, 9)，不应消耗任何快照槽位
    for expected_len in (3, 6, 9):
        scheduler.step()
        assert req.prefilled_len == expected_len
        assert scheduler.req_to_state_pool.available_snapshot_size() == 4
        _, node = scheduler.radix_cache.match_prefix(prompt[:expected_len])
        assert node.linear_state_slot is None

    # 第 4 步完成 Prompt Prefill (10/10)，应捕获 1 个快照
    scheduler.step()
    assert req.is_prefill_complete
    assert scheduler.req_to_state_pool.available_snapshot_size() == 3
    _, node_10 = scheduler.radix_cache.match_prefix(prompt)
    assert node_10.linear_state_slot is not None

    # 第 5 步完成 Decode 并退休，归档终点挂载 1 个快照
    scheduler.step()
    assert scheduler.req_to_state_pool.available_snapshot_size() == 2

    # 模拟 node_10 的线性快照被单独剥离（如 LRU 驱逐），验证分支边界状态自愈
    slot_to_free = node_10.linear_state_slot
    node_10.linear_state_slot = None
    scheduler.req_to_state_pool.free_snapshots([slot_to_free])
    assert scheduler.req_to_state_pool.available_snapshot_size() == 3

    scheduler.chunked_prefill_size = 16
    req2 = Request(
        req_id="req-heal",
        prompt_text="heal",
        prompt_ids=prompt + [11, 12],
        max_new_tokens=1,
        temperature=0.0,
    )
    scheduler.waiting_queue.append(req2)
    # 第一步会在 prefix_len_at_start=10 处切块对齐并自愈 node_10 的线性快照
    scheduler.step()
    assert req2.prefilled_len == 10
    assert node_10.linear_state_slot is not None


def test_decode_preemption_and_recovery() -> None:
    """
    验证：当 Decode 阶段物理 KV Cache 耗尽（且无可驱逐的 LRU 叶子）时，
    调度器能够抢占低优先级请求回退至 waiting_queue 释放 KV 槽位，
    待高优先级请求完成释放资源后，被抢占请求重新准入并完整生成剩余 Token，无槽位泄漏。
    """
    if torch is None:
        return

    from engine.request import Request
    from engine.scheduler import Scheduler
    from qwen3_8 import Qwen3_8Config

    torch.manual_seed(0)
    config_path = PROJECT_ROOT / "qwen3.8 27B" / "config.json"
    full_cfg = Qwen3_8Config.from_json_file(config_path)
    mini_cfg = replace(
        full_cfg,
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
        num_key_value_heads=1,
        head_dim=32,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_num_value_heads=6,
        linear_value_head_dim=16,
    )

    # 总物理 KV 槽位仅 10 个，两条互不重叠的 4-token prompt 各生成 3 个 token，并发峰值需求为 12 > 10
    scheduler = Scheduler(
        config=mini_cfg,
        max_reqs=4,
        max_context_len=32,
        max_total_tokens=10,
        max_snapshots=4,
        chunked_prefill_size=8,
        device="cpu",
    )

    req1 = Request(
        req_id="req-1",
        prompt_text="p1",
        prompt_ids=[1, 2, 3, 4],
        max_new_tokens=3,
        temperature=0.0,
    )
    req2 = Request(
        req_id="req-2",
        prompt_text="p2",
        prompt_ids=[11, 12, 13, 14],
        max_new_tokens=3,
        temperature=0.0,
    )
    scheduler.waiting_queue.extend([req1, req2])

    saw_preemption = False
    steps = 0
    while scheduler.waiting_queue or scheduler.inflight_prefill_queue or scheduler.running_batch:
        scheduler.step()
        steps += 1
        assert steps < 30, "调度循环疑似死锁"
        if req2 in scheduler.waiting_queue:
            saw_preemption = True

    assert saw_preemption, "预期在 KV 池耗尽时触发 req2 抢占回退至 waiting_queue"
    assert len(req1.full_token_ids) == 4 + 3
    assert len(req2.full_token_ids) == 4 + 3
    assert len(scheduler.req_to_token_pool.free_req_slots) == 4
    # 驱逐所有可驱逐缓存后，全部可用物理 KV 槽位均应完整归还（slot 0 保留作 padding，共 9 个）
    scheduler.radix_cache.evict(10)
    assert scheduler.token_to_kv_pool.available_size() == 9


def test_budget_aware_admission_and_decode_non_starvation() -> None:
    """
    验证：
    1. `_admit_waiting_requests` 与步长预算对齐：当首个请求已占满当前步 `chunked_prefill_size` 预算时，
       后续等待请求不会提前准入占用 `req_pool_idx` 或锁死树节点。
    2. 当 `len(running_batch) >= chunked_prefill_size` 时，仍保留 `min_prefill_budget=1` 使新 Prefill 请求不被饿死。
    """
    if torch is None:
        return

    from engine.request import Request
    from engine.scheduler import Scheduler
    from qwen3_8 import Qwen3_8Config

    torch.manual_seed(0)
    config_path = PROJECT_ROOT / "qwen3.8 27B" / "config.json"
    full_cfg = Qwen3_8Config.from_json_file(config_path)
    mini_cfg = replace(
        full_cfg,
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
        num_key_value_heads=1,
        head_dim=32,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_num_value_heads=6,
        linear_value_head_dim=16,
    )

    scheduler = Scheduler(
        config=mini_cfg,
        max_reqs=4,
        max_context_len=32,
        max_total_tokens=64,
        max_snapshots=4,
        chunked_prefill_size=2,
        device="cpu",
    )

    req_a = Request("req-a", "a", [1, 2, 3, 4], max_new_tokens=5)
    req_b = Request("req-b", "b", [5, 6], max_new_tokens=5)
    scheduler.waiting_queue.extend([req_a, req_b])

    # Step 1: 预算为 2，仅够 req_a 的前 2 个 token；req_b 应保留在 waiting_queue 中不占用请求槽位
    scheduler.step()
    assert req_a in scheduler.inflight_prefill_queue
    assert req_b in scheduler.waiting_queue
    assert req_b.req_pool_idx is None
    assert len(scheduler.req_to_token_pool.free_req_slots) == 3

    # Step 2: req_a 完成剩余 2 个 token 的 Prefill 进入 running_batch
    scheduler.step()
    assert req_a in scheduler.running_batch

    # Step 3: req_a 占 1 个 Decode 预算，剩余 1 个预算准入 req_b (1/2)
    scheduler.step()
    assert req_b in scheduler.inflight_prefill_queue
    assert req_b.prefilled_len == 1

    # Step 4: req_b 完成 Prefill (2/2) 进入 running_batch；此时 running_batch 长度为 2 == chunked_prefill_size
    scheduler.step()
    assert len(scheduler.running_batch) == 2

    # 此时新提交 req_c：尽管 len(running_batch) == chunked_prefill_size=2，仍可通过 min_prefill_budget=1 准入并推进！
    req_c = Request("req-c", "c", [7, 8], max_new_tokens=1)
    scheduler.waiting_queue.append(req_c)
    scheduler.step()
    assert req_c.prefilled_len == 1


def test_mtp_and_incremental_vocab_and_fp8_cache() -> None:
    """
    验证：
    1. MTP 模块层索引与专属 `mtp_cache_pool` 闭环（直接传入主 `cache_pool` 或通过 `Scheduler(enable_mtp=True)` 均正常产出草稿 Token）。
    2. `TokenizerManager` -> `Scheduler` -> `DetokenizerManager` 动态增量词表 IPC 同步。
    3. `ModelRunner.req_to_token_tensor` 常驻复用（地址不变）与 `Qwen3_8FP8Linear` 反量化缓存及版本失效刷新。
    """
    if torch is None:
        return

    from engine.detokenizer_manager import DetokenizerManager
    from engine.request import Request
    from engine.scheduler import Scheduler
    from engine.tokenizer_manager import TokenizerManager
    from qwen3_8 import Qwen3_8Config
    from qwen3_8.fp8_linear import Qwen3_8FP8Linear

    torch.manual_seed(7)
    config_path = PROJECT_ROOT / "qwen3.8 27B" / "config.json"
    full_cfg = Qwen3_8Config.from_json_file(config_path)
    mini_cfg = replace(
        full_cfg,
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
        num_key_value_heads=1,
        head_dim=32,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_num_value_heads=6,
        linear_value_head_dim=16,
    )

    # 1. 验证启用 MTP 的 Scheduler / ModelRunner 与常驻 req_to_token_tensor
    scheduler = Scheduler(
        config=mini_cfg,
        max_reqs=4,
        max_context_len=32,
        max_total_tokens=64,
        max_snapshots=4,
        chunked_prefill_size=8,
        device="cpu",
        enable_mtp=True,
    )
    initial_ptr = scheduler.model_runner.req_to_token_tensor.data_ptr()

    # 2. 验证启动后动态新增词表通过 IPC 消息同步至 DetokenizerManager
    tok_mgr = TokenizerManager(config=mini_cfg)
    detok_mgr = DetokenizerManager(id_to_piece={})  # 初始词表为空
    req_input = tok_mgr.submit_request("req-mtp", "alpha beta gamma", max_new_tokens=2)
    assert len(req_input.vocab_updates) == 3

    scheduler.recv_requests(tok_mgr.queue_to_scheduler)
    step1_out = scheduler.step()
    assert step1_out is not None
    assert len(step1_out.vocab_updates) == 3
    assert len(scheduler.model_runner.last_mtp_draft_tokens) == 1
    assert scheduler.model_runner.req_to_token_tensor.data_ptr() == initial_ptr

    # Detokenizer 接收 step1_out 后自动更新 id_to_piece，能够正确解码新词 ID
    detok_mgr.handle_batch_token_out(step1_out)
    for tok_id, piece in req_input.vocab_updates.items():
        assert detok_mgr._decode_token(tok_id) == piece

    # 2.5 验证 MTP 投机解码闭环：启用与不启用 MTP 在相同权重下贪心输出 100% 等价且无 KV 槽位泄漏
    mtp_sched = Scheduler(
        config=mini_cfg,
        max_reqs=4,
        max_context_len=32,
        max_total_tokens=64,
        max_snapshots=4,
        chunked_prefill_size=3,
        device="cpu",
        enable_mtp=True,
    )
    base_sched = Scheduler(
        config=mini_cfg,
        max_reqs=4,
        max_context_len=32,
        max_total_tokens=64,
        max_snapshots=4,
        chunked_prefill_size=3,
        device="cpu",
        enable_mtp=False,
    )
    base_sched.model_runner.model.load_state_dict(
        mtp_sched.model_runner.model.state_dict(), strict=False
    )
    r_mtp = Request("r-mtp", "p", [10, 20, 30, 40, 50], max_new_tokens=5, temperature=0.0)
    r_base = Request("r-base", "p", [10, 20, 30, 40, 50], max_new_tokens=5, temperature=0.0)
    mtp_sched.waiting_queue.append(r_mtp)
    base_sched.waiting_queue.append(r_base)
    while mtp_sched.waiting_queue or mtp_sched.inflight_prefill_queue or mtp_sched.running_batch:
        mtp_sched.step()
    while base_sched.waiting_queue or base_sched.inflight_prefill_queue or base_sched.running_batch:
        base_sched.step()
    assert r_mtp.output_ids == r_base.output_ids
    # 验证启用 MTP 时命中 RadixCache 前缀（且在前缀边界接续不同后缀 Token）依然保持精确一致
    r_mtp2 = Request("r-mtp2", "p2", [10, 20, 30, 40, 50, 60, 70], max_new_tokens=3, temperature=0.0)
    r_base2 = Request("r-base2", "p2", [10, 20, 30, 40, 50, 60, 70], max_new_tokens=3, temperature=0.0)
    mtp_sched.waiting_queue.append(r_mtp2)
    base_sched.waiting_queue.append(r_base2)
    while mtp_sched.waiting_queue or mtp_sched.inflight_prefill_queue or mtp_sched.running_batch:
        mtp_sched.step()
    while base_sched.waiting_queue or base_sched.inflight_prefill_queue or base_sched.running_batch:
        base_sched.step()
    assert r_mtp2.output_ids == r_base2.output_ids
    mtp_sched.radix_cache.evict(64)
    assert mtp_sched.token_to_kv_pool.available_size() == 63

    # 3. 验证 Qwen3_8FP8Linear 权重反量化缓存与原地修改自动失效
    if hasattr(torch, "float8_e4m3fn"):
        fp8_layer = Qwen3_8FP8Linear(128, 128, bias=False, config=mini_cfg)
        fp8_layer.weight = torch.nn.Parameter(
            torch.ones((128, 128), dtype=torch.float8_e4m3fn), requires_grad=False
        )
        assert fp8_layer.weight_scale_inv is not None
        fp8_layer.weight_scale_inv.fill_(2.0)

        x = torch.ones((2, 128), dtype=torch.float32)
        out1 = fp8_layer(x)
        cached_ptr = fp8_layer._cached_dequant_weight.data_ptr()
        out2 = fp8_layer(x)
        assert fp8_layer._cached_dequant_weight.data_ptr() == cached_ptr
        assert torch.allclose(out1, out2)

        # 原地修改 weight_scale_inv 后，版本号变化应自动触发重新反量化
        fp8_layer.weight_scale_inv.fill_(4.0)
        out3 = fp8_layer(x)
        assert torch.allclose(out3, out1 * 2.0)


def test_dirty_from_sync_and_context_bounds() -> None:
    """
    验证：
    1. `ReqToTokenPool.dirty_from` 在前缀位置被覆写时自动回退脏游标，防止 `ModelRunner` 跳过重同步。
    2. `RadixCache.attach_node_linear_state` 在节点已挂载快照时跳过重复全量状态拷贝。
    3. `TokenizerManager.submit_request` 对空 Prompt 抛出 ValueError，并截断超长上下文。
    """
    req_pool = ReqToTokenPool(max_reqs=4, max_context_len=8)
    r0 = req_pool.alloc_req()
    assert r0 is not None
    assert req_pool.dirty_from[r0] == 0
    req_pool.write(r0, 0, [1, 2, 3])
    assert req_pool.dirty_from[r0] == 0
    # 模拟设备端已同步到位置 3
    req_pool.dirty_from[r0] = 3
    # 模拟并发去重覆写了 [0:2]
    req_pool.write(r0, 0, [10, 11])
    assert req_pool.dirty_from[r0] == 0

    if torch is None:
        return

    from engine.scheduler import Scheduler
    from engine.tokenizer_manager import TokenizerManager
    from qwen3_8 import Qwen3_8Config

    torch.manual_seed(11)
    config_path = PROJECT_ROOT / "qwen3.8 27B" / "config.json"
    full_cfg = Qwen3_8Config.from_json_file(config_path)
    mini_cfg = replace(
        full_cfg,
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
        num_key_value_heads=1,
        head_dim=32,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_num_value_heads=6,
        linear_value_head_dim=16,
    )

    tok_mgr = TokenizerManager(config=mini_cfg, max_context_len=6)
    try:
        tok_mgr.submit_request("empty", "   ")
        raise AssertionError("空 Prompt 应抛出 ValueError")
    except ValueError:
        pass

    msg = tok_mgr.submit_request("long", "w1 w2 w3 w4 w5 w6 w7 w8", max_new_tokens=10)
    assert len(msg.prompt_ids) == 5
    assert msg.max_new_tokens == 1

    scheduler = Scheduler(
        config=mini_cfg,
        max_reqs=4,
        max_context_len=16,
        max_total_tokens=64,
        max_snapshots=4,
        chunked_prefill_size=3,
        device="cpu",
    )
    # 并发准入两个共享前缀的请求，验证 r2 首块去重覆写后在下一分块触发 dirty_from 设备端重同步
    from engine.request import Request

    r1 = Request("r1", "p", [10, 11, 12, 13, 14, 15], max_new_tokens=1)
    r2 = Request("r2", "p", [10, 11, 12, 13, 14, 15], max_new_tokens=2)
    scheduler.waiting_queue.extend([r1, r2])
    scheduler._admit_waiting_requests()  # 两者在树为空时同时准入 (prefix_len_at_start=0)
    assert r1.req_pool_idx is not None and r2.req_pool_idx is not None
    scheduler.step()  # r1 chunk 1: [0:3] 计算并首次插入 RadixCache
    scheduler.step()  # r1 chunk 2: [3:6] 完成并归档
    scheduler.step()  # r2 chunk 1: [0:3] 计算后触发并发前缀去重，覆写 req_to_token[r2][:3] 并将 dirty_from[r2] 回退至 0
    scheduler.step()  # r2 chunk 2: [3:6] ModelRunner 根据 dirty_from[r2]=0 重新同步设备端 req_to_token_tensor[r2, :3]
    assert scheduler.model_runner.req_to_token_tensor[r2.req_pool_idx, :3].tolist() == (
        scheduler.req_to_token_pool.req_to_token[r2.req_pool_idx][:3]
    )

    # 验证 eos_token_id=0 不会被误覆盖为 config.eos_token_id
    import queue as py_queue
    from engine.ipc_messages import TokenizedReqInput

    q_in: py_queue.Queue = py_queue.Queue()
    q_in.put(
        TokenizedReqInput(
            req_id="eos-zero",
            prompt_text="p",
            prompt_ids=[1, 2],
            max_new_tokens=2,
            eos_token_id=0,
        )
    )
    scheduler.recv_requests(q_in)  # type: ignore[arg-type]
    assert scheduler.waiting_queue[-1].eos_token_id == 0


def test_prefill_kv_livelock_and_mtp_eos_rollback_and_locked_state_protection() -> None:
    """
    回归验证：
    1. `RadixCache.evict_linear_state` 绝不驱逐 `lock_ref > 0` 的锁定节点线性状态快照。
    2. `TokenizerManager.submit_request` 拒绝重复的活跃 `req_id`。
    3. 当多个长请求处于 `inflight_prefill_queue` 且物理 KV 槽位耗尽时，自动缩减分块或触发尾部抢占，不发生活锁。
    4. 启用 MTP 投机解码时，若第 1 个验证命中的草稿 Token 即为 `eos_token_id` 提前终止，线性状态精确回滚至单步，不污染 `RadixCache` 归档快照。
    """
    # 1. 验证 lock_ref > 0 的节点快照不会被 evict_linear_state 强制剥离
    req_pool = ReqToTokenPool(max_reqs=2, max_context_len=8)
    kv_pool = TokenToKVPool(max_total_num_tokens=8)
    state_pool = ReqToStatePool(max_reqs=2, max_snapshots=1)
    cache = RadixCache(req_pool, kv_pool, state_pool)

    r0 = req_pool.alloc_req()
    assert r0 is not None
    s0 = kv_pool.alloc(2)
    assert s0 is not None
    req_pool.write(r0, 0, s0)
    _, locked_node = cache.cache_unfinished_req(
        req_idx=r0,
        token_ids=[1, 2],
        prefix_len_at_start=0,
        last_locked_node=cache.root,
        capture_linear_state=True,
    )
    assert locked_node.lock_ref == 1
    assert locked_node.linear_state_slot is not None
    evicted_snaps = cache.evict_linear_state(1)
    assert evicted_snaps == [], "正在被锁定的节点 (lock_ref > 0) 的线性状态快照不应被驱逐"
    assert locked_node.linear_state_slot is not None

    if torch is None:
        return

    from engine.request import Request
    from engine.scheduler import Scheduler
    from engine.tokenizer_manager import TokenizerManager
    from qwen3_8 import Qwen3_8Config

    torch.manual_seed(19)
    config_path = PROJECT_ROOT / "qwen3.8 27B" / "config.json"
    full_cfg = Qwen3_8Config.from_json_file(config_path)
    mini_cfg = replace(
        full_cfg,
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
        num_key_value_heads=1,
        head_dim=32,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_num_value_heads=6,
        linear_value_head_dim=16,
    )

    # 2. 验证重复 req_id 拒绝
    tok_mgr = TokenizerManager(config=mini_cfg)
    tok_mgr.submit_request("dup-id", "hello world")
    try:
        tok_mgr.submit_request("dup-id", "hello again")
        raise AssertionError("重复提交活跃 req_id 应抛出 ValueError")
    except ValueError:
        pass

    # 3. 验证 inflight_prefill_queue 在极小 KV 池下的分块自适应与抢占防活锁
    tight_sched = Scheduler(
        config=mini_cfg,
        max_reqs=4,
        max_context_len=16,
        max_total_tokens=9,  # 仅 8 个可用物理槽位
        max_snapshots=2,
        chunked_prefill_size=3,
        device="cpu",
    )
    p_a = Request("pa", "pa", [1, 2, 3, 4, 5, 6], max_new_tokens=1)
    p_b = Request("pb", "pb", [11, 12, 13, 14, 15, 16], max_new_tokens=1)
    tight_sched.waiting_queue.extend([p_a, p_b])
    tight_sched._admit_waiting_requests()  # 强制同时准入两个长 Prefill 请求
    steps = 0
    while (
        tight_sched.waiting_queue
        or tight_sched.inflight_prefill_queue
        or tight_sched.running_batch
    ):
        out = tight_sched.step()
        assert out is not None, "Chunked Prefill 发生 KV 槽位活锁，step() 返回了 None"
        steps += 1
        assert steps < 30

    # 4. 验证 MTP 投机解码下首个命中草稿 Token 触发 EOS 提前终止时，归档至 RadixCache 的线性状态与基准完全一致
    mtp_sched = Scheduler(
        config=mini_cfg,
        max_reqs=4,
        max_context_len=32,
        max_total_tokens=64,
        max_snapshots=4,
        chunked_prefill_size=8,
        device="cpu",
        enable_mtp=True,
    )
    ref_sched = Scheduler(
        config=mini_cfg,
        max_reqs=4,
        max_context_len=32,
        max_total_tokens=64,
        max_snapshots=4,
        chunked_prefill_size=8,
        device="cpu",
        enable_mtp=False,
    )
    ref_sched.model_runner.model.load_state_dict(
        mtp_sched.model_runner.model.state_dict(), strict=False
    )

    # 先探明在 [10, 20, 30] 后第 2 步生成的真实 Token，并将其设为 eos_token_id
    probe_req = Request("probe", "p", [10, 20, 30], max_new_tokens=3, temperature=0.0)
    ref_sched.waiting_queue.append(probe_req)
    while ref_sched.waiting_queue or ref_sched.inflight_prefill_queue or ref_sched.running_batch:
        ref_sched.step()
    tok_step1, tok_step2 = probe_req.output_ids[0], probe_req.output_ids[1]

    # 在 mtp_sched 中强制将 pending_draft_token 设为 tok_step2，并将 eos_token_id 设为 tok_step2
    r_eos = Request(
        "r-eos",
        "p",
        [10, 20, 30],
        max_new_tokens=5,
        eos_token_id=tok_step2,
        temperature=0.0,
    )
    mtp_sched.waiting_queue.append(r_eos)
    mtp_sched.step()  # Prefill 完成，产出 tok_step1
    assert r_eos.output_ids == [tok_step1]
    r_eos.pending_draft_token = tok_step2  # 确保下一步 MTP 验证必中且 target_1 == eos_token_id
    mtp_sched.step()  # Decode 步：MTP 验证命中 tok_step2 并因 EOS 在首 Token 处立即终止归档
    assert r_eos.is_finished()
    assert r_eos.output_ids == [tok_step1, tok_step2]

    # 此时 RadixCache 中 [10, 20, 30, tok_step1] 节点的线性状态快照必须与单步基准完全一致
    follow_prompt = [10, 20, 30, tok_step1, 99, 100]
    r_follow_mtp = Request("f-mtp", "f", follow_prompt, max_new_tokens=2, temperature=0.0)
    mtp_sched.waiting_queue.append(r_follow_mtp)
    while mtp_sched.waiting_queue or mtp_sched.inflight_prefill_queue or mtp_sched.running_batch:
        mtp_sched.step()

    fresh_sched = Scheduler(
        config=mini_cfg,
        max_reqs=4,
        max_context_len=32,
        max_total_tokens=64,
        max_snapshots=4,
        chunked_prefill_size=8,
        device="cpu",
        enable_mtp=False,
    )
    fresh_sched.model_runner.model.load_state_dict(
        mtp_sched.model_runner.model.state_dict(), strict=False
    )
    r_follow_ref = Request("f-ref", "f", follow_prompt, max_new_tokens=2, temperature=0.0)
    fresh_sched.waiting_queue.append(r_follow_ref)
    while fresh_sched.waiting_queue or fresh_sched.inflight_prefill_queue or fresh_sched.running_batch:
        fresh_sched.step()

    assert r_follow_mtp.output_ids == r_follow_ref.output_ids


if __name__ == "__main__":
    test_asymmetric_eviction_and_state_healing()
    test_hybrid_scheduler_prefix_reuse_exact_equivalence()
    test_sparse_snapshot_skips_intermediate_chunks()
    test_decode_preemption_and_recovery()
    test_budget_aware_admission_and_decode_non_starvation()
    test_mtp_and_incremental_vocab_and_fp8_cache()
    test_dirty_from_sync_and_context_bounds()
    test_prefill_kv_livelock_and_mtp_eos_rollback_and_locked_state_protection()
