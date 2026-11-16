# mini-SGLang 代码 Review 整改建议

> Review 日期：2026-11-16
> 范围：`engine/`、`qwen3_8/`（未细看 `fp8_linear.py` / `fp8_triton.py` / `tokenizer_manager.py`）、`radix_tree.py`、`req_to_*_pool.py`、`token_to_kv_pool.py`
> 现状：`pytest` 10 passed。下列问题大多是随机权重测试覆盖不到的场景。

优先级说明：🔴 必须修复（影响正确性） / 🟠 建议修复（潜在 Bug / 健壮性） / 🟡 优化（性能）

---

## 🔴 P0：正确性问题（加载真实权重后会出错）

### 1. Full Attention 输出门控应使用 `sigmoid` 而非 `silu`

- 位置：`qwen3_8/full_attention.py` → `Qwen3_8FullAttention.forward`
- 问题：Qwen3-Next / Qwen3.5 系列的 `attn_output_gate` 为 `attn_output * sigmoid(gate)`，当前实现为 `F.silu(gate)`。
- 整改：

```python
# before
if gate is not None:
    attn_output = attn_output * F.silu(gate)
# after
if gate is not None:
    attn_output = attn_output * torch.sigmoid(gate)
```

同时修正类 docstring 中 “Swish 门控” 的描述。

### 2. RMSNorm 权重约定需对齐 HF 参考实现（`1 + weight`）

- 位置：`qwen3_8/rms_norm.py`
- 问题：Qwen3-Next 家族的主干 RMSNorm（`input_layernorm` / `post_attention_layernorm` / `q_norm` / `k_norm` / 顶层 `norm` / MTP 各 norm）采用零初始化 + `x * (1 + w)` 约定；当前为 `w * x`。若 checkpoint 按 `1+w` 存储，直接加载会导致输出完全错误。
- 整改（先对照 HF `modeling_qwen3_next.py` / `modeling_qwen3_5.py` 确认）：

```python
class Qwen3_8RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))  # 零初始化
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        x_normed = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_normed * (1.0 + self.weight.float())).to(orig_dtype)
```

> 注意：`Qwen3_8RMSNormGated`（`linear_attn.norm`）在 HF 中是普通 `weight` 约定（ones 初始化），**不要**一起改。

### 3. `linear_attn.norm.weight` 形状与 checkpoint 不一致

- 位置：`qwen3_8/linear_attention.py`（`self.norm = Qwen3_8RMSNormGated(self.value_dim, ..., head_dim=self.head_v_dim)`）及 `load_tp_from_full_layer`
- 问题：HF 中为 `RMSNormGated(head_v_dim)`，权重形状 `[128]`，所有 Value Head 共享；当前为 `[value_dim]`，TP 加载时又按 `value_dim` 切片。
- 整改：

```python
# __init__
self.norm = Qwen3_8RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps, head_dim=self.head_v_dim)

# load_tp_from_full_layer：不再切片，每个 rank 保存完整 [head_v_dim] 权重
self.norm.weight.copy_(full_layer.norm.weight)
```

`Qwen3_8RMSNormGated.forward` 需相应地在 `[..., num_heads, head_dim]` 视图上乘 `weight`（广播）：

```python
x_heads = x_fp32.view(*orig_shape[:-1], -1, self.head_dim)
x_normed = x_heads * torch.rsqrt(x_heads.pow(2).mean(-1, keepdim=True) + self.eps)
out = (self.weight.float() * x_normed).view(orig_shape) * F.silu(gate.float())
```

---

## 🟠 P1：潜在 Bug / 健壮性

### 4. Scheduler 中 MTP “首 Token 即停止” 分支是死代码，且一旦触发会 Double-Free

- 位置：`engine/scheduler.py` → `step()` 的 Decode 分支（`if idx_tok == 0 and len(accepted_toks) == 2:`）
- 问题：`ModelRunner.execute_batch_speculative` 中 `stops_on_first` 与 `Request.is_finished()` 条件完全等价，此时只会返回 `[target_1]`，该分支不可达。若将来两处条件不一致：
  1. `out_cache_loc[1]` 会被 Scheduler 再次释放（Runner 已释放过）→ Double-Free 断言；
  2. 即便不重复释放，线性工作区状态已多推进了 1 个 Token，`_retire_request` 会把错误状态挂成快照，污染 RadixCache。
- 整改：删除该分支，改为显式契约断言，由 Runner 单点负责回滚：

```python
for idx_tok, tok in enumerate(accepted_toks):
    req.output_ids.append(tok)
    emitted_toks.append(tok)
    if req.is_finished():
        assert idx_tok == len(accepted_toks) - 1, (
            "ModelRunner 必须在首 Token 触发停止时回滚第 2 个草稿位置"
        )
        break
```

### 5. 线性状态追赶区间重算时覆写共享的树上 KV 槽位

- 位置：`engine/scheduler.py` → `_build_mixed_chunked_batch` 的 Chunked Prefill 部分
- 问题：处于 `[state_anchor_len, prefix_len_at_start)` 区间时，`out_cache_loc` 直接取自树上已缓存的共享槽位，前向会把重算的 K/V 写回这些槽位。同批次/其它请求正在读取它们；bf16 + 不同组批形态下数值不是 bit-exact，会造成其它请求输出的非确定性。
- 整改：这些位置的 KV 早已存在，无需写入；将其写入目标重定向到保留的 padding 槽位 0（Attention 读取仍走 `req_to_token` 中的树上槽位）：

```python
slots = self.req_to_token_pool.get_req_slots(req.req_pool_idx, end_pos)[start_pos:end_pos]
reuse_end = min(end_pos, req.prefix_len_at_start)
req.out_cache_loc = [
    0 if (start_pos + i) < reuse_end else s for i, s in enumerate(slots)
]
```

同时需要修改 `ModelRunner._prepare_forward_batch`：对 `req_to_token_tensor[r_idx, cur_start_pos:cur_seq_len]` 不能再直接用 `out_cache_loc` 覆盖，而应使用 `req_to_token_table` 中的真实槽位：

```python
self.req_to_token_tensor[r_idx, req.cur_start_pos:req.cur_seq_len] = torch.tensor(
    req_to_token_table[r_idx][req.cur_start_pos:req.cur_seq_len],
    dtype=torch.long, device=self.device,
)
```

MTP 路径中 `_run_mtp_for_req(out_locs=req.out_cache_loc)` 也要同步改为真实槽位或跳过 0 号槽。

### 6. Decode 槽位耗尽时直接 `raise` 会使整个 Scheduler 进程崩溃

- 位置：`engine/scheduler.py` → `_build_mixed_chunked_batch`（`raise RuntimeError("KV Cache 物理池耗尽...")`）；Prefill 部分同理。
- 整改：对无法推进的单个请求执行 abort，并向 Detokenizer 回传带错误标记的 finished 消息，而不是让进程退出：

```python
if new_slot is None:
    if req not in self.running_batch:
        continue
    self.running_batch.pop(idx)
    self._abort_with_error(req, reason="kv_cache_oom")  # 释放资源 + 记录待回传的 finished 消息
    continue
```

### 7. 抢占逻辑与文档不一致

- 位置：`engine/scheduler.py` → `_preempt_request`
- 问题与整改：
  1. **已生成 Token 的 KV 被直接释放**：docstring 声称可复用 RadixCache 前缀，但 Decode 私有槽位 `[len(prompt), cached_token_count)` 被 free 掉，重新准入时会全部重算。建议抢占 Decode 请求时改为调用 `cache_unfinished_req`/`insert` 把 `full_token_ids[:cached_token_count]` 入树（可同时挂载线性快照），再解锁并释放行；若坚持当前做法，请修改 docstring。
  2. **多个 victim 的优先级被反转**：每个 victim 都 `insert(0, ...)`，后抢占的排到前面。建议收集本步 victims，最后按原顺序一次性 `self.waiting_queue[0:0] = victims`。
  3. **`free_mtp_slots` 重复调用**：先手动调用一次，`abort_req` 内的 `on_kv_free` 又调用一次。删除 `_preempt_request` 中的手动调用即可。

### 8. 其它小问题

| 问题 | 位置 | 整改建议 |
| --- | --- | --- |
| Token 预算把 MTP verify（2 Token）按 1 计 | `_build_mixed_chunked_batch`：`remaining_budget = chunked_prefill_size - len(active_batch)` | 改为 `sum(len(r.cur_chunk_tokens) for r in active_batch)` |
| 超长 Prompt 截尾保头，丢失最新内容 | `recv_requests` | 改为保留尾部 `prompt_ids[-(max_context_len-1):]`，或直接拒绝并回传错误 |
| 收到 `ShutdownSignal` 后在途请求被静默丢弃 | `run_process_loop` | 退出前 drain：对 waiting/inflight/running 中请求回传 aborted finished 消息 |
| `_estimate_inflight_budget_demand` 未被使用 | `scheduler.py` | 删除 |

---

## 🟡 P2：性能优化

### 9. 投机路径对所有 Token（含整段 Prefill Chunk）计算 lm_head

- 位置：`ModelRunner.execute_batch_speculative` → `self.model(..., return_all_logits=True)`
- 问题：词表 248320，`num_tokens × vocab` 的 logits 显存开销巨大。
- 整改：给 `Qwen3_8ForCausalLM.forward` 增加 `logits_indices: Optional[Tensor]` 参数，只对需要的位置计算 logits：
  - Prefill 请求：仅最后 1 个位置；
  - MTP verify 请求：2 个位置；
  - 普通 Decode：1 个位置。

  全量 `_last_hidden_states` 仍保留给 MTP 使用。

### 10. `_run_mtp_for_req` 每步全前缀扫描 + GPU→CPU 同步

- 位置：`engine/model_runner.py` → `_run_mtp_for_req` 开头的 prefix 校准循环
- 问题：每请求每步 O(seq_len) Python 循环 + `.tolist()` 同步；且不在 `_mtp_slot_hidden` 中的槽位被静默跳过，可能残留陈旧 MTP KV（只影响接受率）。
- 整改：
  - 只需检查“分叉点”：即请求准入时 `prefix_len_at_start - 1` 处的槽位（共享前缀的最后一个位置是唯一可能 next_token 不一致的位置）。在 `Request` 上记录 `mtp_fixed=False`，校准一次后置 True。
  - 用 CPU 侧 `req_to_token_pool` 读取槽位，避免设备同步。

### 11. 共享前缀第二次命中时线性状态无法复用

- 问题：快照仅在 Prompt 末尾和请求结束时挂载。A=`S+X` 后到达 B=`S+Y`：B 的 KV 命中 S，但 S 末尾无快照 → 线性层从 0 重算 S，第三个请求才受益。
- 整改（任选）：
  - 在 Chunked Prefill 的块边界（例如每 `N` 个 Token 或按 page 对齐位置）额外挂快照，受 `max_snapshots` 与 LRU 约束；
  - 或在 `insert` 产生分裂点时，若当前请求恰好经过该点，则在该 chunk 边界处切块并捕获快照（类似现有“自愈”逻辑，但对**首次**插入也生效）。

---

## 建议修复顺序

1. P0-1 / P0-2 / P0-3（接真实权重前必须完成，并补一个与 HF 参考实现逐层对齐的数值测试）
2. P1-4（零成本防御）
3. P1-5、P1-6（确定性与稳定性）
4. P1-7、P1-8
5. P2 性能项

## 建议新增测试

- [ ] 单层与 HF `Qwen3NextAttention` / `Qwen3NextGatedDeltaNet` / `Qwen3NextRMSNorm` 的数值对齐测试（相同随机权重）
- [ ] 追赶区间重算前后，树上共享 KV 槽位内容 bit-exact 不变
- [ ] 抢占 Decode 请求后重新准入，验证输出与不抢占时一致（greedy）
- [ ] KV 池极小时的 OOM 场景：单请求被 abort 而进程不崩溃
- [ ] MTP verify 拒绝 / EOS / max_new_tokens 边界下的槽位无泄漏（`token_to_kv_pool.available_size()` 前后守恒）
