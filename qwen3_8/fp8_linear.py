import math
from typing import Any, Callable, List, Optional
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3_8Config
from .fp8_triton import can_use_triton_fp8_gemm, triton_block_fp8_linear


class Qwen3_8FP8Linear(nn.Module):
    """
    对齐 `Qwen3.8-27B-FP8` (`model.safetensors.index.json` 与 `config.json`) 的块级 FP8 线性层基类：
    - 量化配置：`quant_method = "fp8"`, `fmt = "e4m3"`, `activation_scheme = "dynamic"`, `weight_block_size = [128, 128]`
    - 参数与缓冲区命名严格对齐 `model.safetensors.index.json`：
      - `<layer>.weight`: `[out_features, in_features]`
      - `<layer>.weight_scale_inv`: `[ceil(out_features / block_h), ceil(in_features / block_w)]` (当启用 FP8 时注册)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        config: Optional[Qwen3_8Config] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        quant_cfg = config.quantization_config if config is not None else None
        self.is_fp8 = (
            isinstance(quant_cfg, dict) and quant_cfg.get("quant_method") == "fp8"
        )
        block_size = (
            quant_cfg.get("weight_block_size", [128, 128])
            if isinstance(quant_cfg, dict)
            else [128, 128]
        )
        self.block_h: int = int(block_size[0])
        self.block_w: int = int(block_size[1])

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        if self.is_fp8:
            scale_h = math.ceil(out_features / self.block_h)
            scale_w = math.ceil(in_features / self.block_w)
            self.register_buffer(
                "weight_scale_inv",
                torch.ones((scale_h, scale_w), dtype=torch.float32),
                persistent=True,
            )
        else:
            self.register_buffer("weight_scale_inv", None, persistent=False)

        self._cached_dequant_weight: Optional[torch.Tensor] = None
        self._cached_dequant_meta: Optional[tuple] = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self._cached_dequant_weight = None
        self._cached_dequant_meta = None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def dequantize_weight(self, target_dtype: torch.dtype) -> torch.Tensor:
        """
        将 `128x128` 块级量化的 `weight` 与 `weight_scale_inv` 反量化至 `target_dtype`。
        """
        if self.weight_scale_inv is None:
            return self.weight.to(target_dtype)

        w_fp = self.weight.to(torch.float32)
        scale = self.weight_scale_inv.to(torch.float32)
        scale_expanded = scale.repeat_interleave(self.block_h, dim=0).repeat_interleave(
            self.block_w, dim=1
        )
        scale_expanded = scale_expanded[: self.out_features, : self.in_features]
        return (w_fp * scale_expanded).to(target_dtype)

    def _get_cached_dequantized_weight(self, target_dtype: torch.dtype) -> torch.Tensor:
        """
        获取反量化后的权重并按底层张量版本号缓存，避免在非 FP8 Tensor Core 设备上每步重复执行全量反量化。
        """
        scale_ver = (
            self.weight_scale_inv._version
            if self.weight_scale_inv is not None
            else -1
        )
        meta = (
            id(self.weight),
            self.weight._version,
            id(self.weight_scale_inv),
            scale_ver,
            target_dtype,
            self.weight.device,
        )
        if self._cached_dequant_weight is None or self._cached_dequant_meta != meta:
            self._cached_dequant_weight = self.dequantize_weight(target_dtype)
            self._cached_dequant_meta = meta
        return self._cached_dequant_weight

    def _can_use_cuda_scaled_mm(self, x: torch.Tensor) -> bool:
        if not (x.is_cuda and hasattr(torch, "_scaled_mm")):
            return False
        major, minor = torch.cuda.get_device_capability(x.device)
        if (major, minor) < (8, 9):
            return False
        m = x.numel() // x.shape[-1]
        return (
            m % 16 == 0
            and self.in_features % 16 == 0
            and self.out_features % 16 == 0
        )

    def _cuda_fp8_scaled_mm(
        self, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """在支持 FP8 Tensor Core (SM89+) 的 CUDA 设备上调用原生 `torch._scaled_mm` 算子。"""
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        fp8_max = 448.0
        amax = x_2d.abs().max().clamp(min=1e-12)
        x_scale = (amax / fp8_max).to(torch.float32)
        x_fp8 = (x_2d.float() / x_scale).clamp(-fp8_max, fp8_max).to(
            torch.float8_e4m3fn
        )

        if self.weight_scale_inv is not None and self.weight_scale_inv.numel() == 1:
            w_scale = self.weight_scale_inv.reshape(1).to(torch.float32)
            out = torch._scaled_mm(
                x_fp8,
                self.weight.t(),
                scale_a=x_scale,
                scale_b=w_scale,
                bias=bias,
                out_dtype=x.dtype,
            )
            return out.reshape(*orig_shape[:-1], self.out_features)

        # 128x128 块级缩放下，沿 K 维分块调用 _scaled_mm 累加
        out_acc = torch.zeros(
            (x_2d.shape[0], self.out_features), dtype=torch.float32, device=x.device
        )
        one_scale = torch.ones((1,), dtype=torch.float32, device=x.device)
        for k_idx, k_start in enumerate(range(0, self.in_features, self.block_w)):
            k_end = min(k_start + self.block_w, self.in_features)
            if (k_end - k_start) % 16 != 0:
                w = self._get_cached_dequantized_weight(x.dtype)
                return F.linear(x, w, bias)
            x_blk = x_fp8[:, k_start:k_end].contiguous()
            w_blk_t = self.weight[:, k_start:k_end].contiguous().t()
            blk_mm = torch._scaled_mm(
                x_blk,
                w_blk_t,
                scale_a=x_scale,
                scale_b=one_scale,
                out_dtype=torch.float32,
            )
            if self.weight_scale_inv is not None:
                col_scale = (
                    self.weight_scale_inv[:, k_idx]
                    .repeat_interleave(self.block_h)[: self.out_features]
                    .unsqueeze(0)
                )
                blk_mm = blk_mm * col_scale
            out_acc = out_acc + blk_mm

        if bias is not None:
            out_acc = out_acc + bias.float()
        return out_acc.to(x.dtype).reshape(*orig_shape[:-1], self.out_features)

    def _linear_op(
        self, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        is_float8_weight = hasattr(torch, "float8_e4m3fn") and (
            self.weight.dtype == torch.float8_e4m3fn
        )
        if is_float8_weight:
            if can_use_triton_fp8_gemm(x):
                return triton_block_fp8_linear(
                    x=x,
                    weight=self.weight,
                    weight_scale_inv=self.weight_scale_inv,
                    bias=bias,
                    block_h=self.block_h,
                    block_w=self.block_w,
                )
            if self._can_use_cuda_scaled_mm(x):
                return self._cuda_fp8_scaled_mm(x, bias)
            w = self._get_cached_dequantized_weight(x.dtype)
            return F.linear(x, w, bias)
        return F.linear(x, self.weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear_op(x, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, is_fp8={self.is_fp8}, "
            f"block_size=({self.block_h}, {self.block_w})"
        )


class ColumnParallelFP8Linear(Qwen3_8FP8Linear):
    """
    Megatron-LM 风格的【列并行 (Column-Parallel)】FP8 线性层（用于“先列并行、后行并行”的前段）：
    - 沿输出通道维度 (`dim=0`) 将权重切分为 `tp_size` 份，前向输出直接保留在各卡本地，**0 跨卡通信**。
    - 支持两种切分模式：
      1. 连续切分 (`partition_sizes=None`)：如 `gate_proj`、`up_proj`、`q_proj`、`in_proj_z`。
      2. 分组拼接切分 (`partition_sizes=[q_dim, k_dim, v_dim]`)：如 `linear_attn.in_proj_qkv`，
         分别对每个逻辑分段切取第 `tp_rank` 份后再拼接，确保每张卡本地同时获得配对的 `[Q_r, K_r, V_r]`。
      3. 当全局输出头数小于 `tp_size`（例如 `num_kv_heads=4, tp_size=8`）时，自动启用 `replicate_groups` 头复制模式。
    """

    def __init__(
        self,
        in_features: int,
        global_out_features: int,
        bias: bool = False,
        config: Optional[Qwen3_8Config] = None,
        partition_sizes: Optional[List[int]] = None,
        num_total_heads: Optional[int] = None,
    ):
        self.tp_size = config.tp_size if config is not None else 1
        self.tp_rank = config.tp_rank if config is not None else 0
        self.global_out_features = global_out_features
        self.partition_sizes = partition_sizes
        self.num_total_heads = num_total_heads

        if partition_sizes is not None:
            assert sum(partition_sizes) == global_out_features
            for p in partition_sizes:
                assert p % self.tp_size == 0, (
                    f"分段维度 {p} 无法被 tp_size={self.tp_size} 整除！"
                )
            local_out_features = global_out_features // self.tp_size
            self.kv_replicate_factor = 1
        elif (
            num_total_heads is not None
            and num_total_heads < self.tp_size
        ):
            assert self.tp_size % num_total_heads == 0
            self.kv_replicate_factor = self.tp_size // num_total_heads
            head_dim = global_out_features // num_total_heads
            local_out_features = head_dim
        else:
            assert global_out_features % self.tp_size == 0, (
                f"global_out_features={global_out_features} 无法被 tp_size={self.tp_size} 整除！"
            )
            local_out_features = global_out_features // self.tp_size
            self.kv_replicate_factor = 1

        super().__init__(
            in_features=in_features,
            out_features=local_out_features,
            bias=bias,
            config=config,
        )

    @torch.no_grad()
    def slice_full_tensor(
        self, full_tensor: torch.Tensor, is_scale: bool = False
    ) -> torch.Tensor:
        """从完整单卡权重 (或 weight_scale_inv) 中切出当前 `tp_rank` 对应的列并行分片。"""
        if self.tp_size == 1:
            return full_tensor.clone()

        divisor = self.block_h if is_scale else 1

        if self.partition_sizes is not None:
            if is_scale and any(
                (p // self.tp_size) < self.block_h for p in self.partition_sizes
            ):
                assert self.weight_scale_inv is not None
                local_scale_rows = self.weight_scale_inv.shape[0]
                row_indices: List[int] = []
                for local_scale_idx in range(local_scale_rows):
                    local_row = local_scale_idx * self.block_h
                    rem = local_row
                    global_offset = 0
                    chosen_global_row = 0
                    for p_size in self.partition_sizes:
                        local_p = p_size // self.tp_size
                        if rem < local_p:
                            chosen_global_row = (
                                global_offset + self.tp_rank * local_p + rem
                            )
                            break
                        rem -= local_p
                        global_offset += p_size
                    row_indices.append(
                        min(chosen_global_row // self.block_h, full_tensor.shape[0] - 1)
                    )
                idx_t = torch.tensor(
                    row_indices, dtype=torch.long, device=full_tensor.device
                )
                return full_tensor.index_select(0, idx_t).contiguous().clone()
            shards = []
            offset = 0
            for p_size in self.partition_sizes:
                scaled_p = math.ceil(p_size / divisor)
                shard_len = math.ceil((p_size // self.tp_size) / divisor)
                start = offset + min(
                    self.tp_rank * shard_len, max(0, scaled_p - shard_len)
                )
                shards.append(full_tensor[start : start + shard_len])
                offset += scaled_p
            return torch.cat(shards, dim=0).contiguous()

        eff_rank = self.tp_rank // self.kv_replicate_factor
        if is_scale:
            local_scale_rows = math.ceil(self.out_features / self.block_h)
            row_indices = [
                min(
                    (eff_rank * self.out_features + i * self.block_h) // self.block_h,
                    full_tensor.shape[0] - 1,
                )
                for i in range(local_scale_rows)
            ]
            idx_t = torch.tensor(
                row_indices, dtype=torch.long, device=full_tensor.device
            )
            return full_tensor.index_select(0, idx_t).contiguous().clone()

        shard_len = self.out_features
        total_len = full_tensor.shape[0]
        start = min(eff_rank * shard_len, max(0, total_len - shard_len))
        return full_tensor[start : start + shard_len].contiguous()

    @torch.no_grad()
    def load_from_full_linear(self, full_linear: Qwen3_8FP8Linear) -> None:
        """从完整单卡 `Qwen3_8FP8Linear` 切片加载本卡的列并行权重与 FP8 缩放系数。"""
        self._cached_dequant_weight = None
        self._cached_dequant_meta = None
        self.weight.copy_(self.slice_full_tensor(full_linear.weight, is_scale=False))
        if self.bias is not None and full_linear.bias is not None:
            self.bias.copy_(self.slice_full_tensor(full_linear.bias, is_scale=False))
        if (
            self.weight_scale_inv is not None
            and full_linear.weight_scale_inv is not None
        ):
            self.weight_scale_inv.copy_(
                self.slice_full_tensor(full_linear.weight_scale_inv, is_scale=True)
            )


class RowParallelFP8Linear(Qwen3_8FP8Linear):
    """
    Megatron-LM 风格的【行并行 (Row-Parallel)】FP8 线性层（用于“先列并行、后行并行”的后段）：
    - 沿输入通道维度 (`dim=1`) 将权重切分为 `tp_size` 份，直接接收前段列并行的本地激活值计算部分和 (Partial Sum)。
    - 在矩阵乘完成后，执行**全层该子块唯一一次 `AllReduce(Sum)`** 聚合各卡结果，最后加上 Bias。
    """

    def __init__(
        self,
        global_in_features: int,
        out_features: int,
        bias: bool = False,
        config: Optional[Qwen3_8Config] = None,
        tp_group: Optional[Any] = None,
        all_reduce_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.tp_size = config.tp_size if config is not None else 1
        self.tp_rank = config.tp_rank if config is not None else 0
        self.global_in_features = global_in_features
        self.tp_group = tp_group
        self.all_reduce_fn = all_reduce_fn

        assert global_in_features % self.tp_size == 0, (
            f"global_in_features={global_in_features} 无法被 tp_size={self.tp_size} 整除！"
        )
        local_in_features = global_in_features // self.tp_size

        super().__init__(
            in_features=local_in_features,
            out_features=out_features,
            bias=bias,
            config=config,
        )

    @torch.no_grad()
    def slice_full_tensor(
        self, full_tensor: torch.Tensor, is_scale: bool = False
    ) -> torch.Tensor:
        """从完整单卡权重 (或 weight_scale_inv) 中沿第 1 维 (`in_features`) 切出当前 `tp_rank` 的行并行分片。"""
        if self.tp_size == 1:
            return full_tensor.clone()

        if is_scale:
            local_scale_cols = math.ceil(self.in_features / self.block_w)
            col_indices = [
                min(
                    (self.tp_rank * self.in_features + i * self.block_w) // self.block_w,
                    full_tensor.shape[1] - 1,
                )
                for i in range(local_scale_cols)
            ]
            idx_t = torch.tensor(
                col_indices, dtype=torch.long, device=full_tensor.device
            )
            return full_tensor.index_select(1, idx_t).contiguous().clone()

        shard_len = self.in_features
        total_len = full_tensor.shape[1]
        start = min(self.tp_rank * shard_len, max(0, total_len - shard_len))
        return full_tensor[:, start : start + shard_len].contiguous()

    @torch.no_grad()
    def load_from_full_linear(self, full_linear: Qwen3_8FP8Linear) -> None:
        """从完整单卡 `Qwen3_8FP8Linear` 切片加载本卡的行并行权重与 FP8 缩放系数。"""
        self._cached_dequant_weight = None
        self._cached_dequant_meta = None
        self.weight.copy_(self.slice_full_tensor(full_linear.weight, is_scale=False))
        if self.bias is not None and full_linear.bias is not None:
            self.bias.copy_(full_linear.bias)
        if (
            self.weight_scale_inv is not None
            and full_linear.weight_scale_inv is not None
        ):
            self.weight_scale_inv.copy_(
                self.slice_full_tensor(full_linear.weight_scale_inv, is_scale=True)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 各卡本地计算部分和 (不加 bias，bias 需在 AllReduce 后只加一次)
        partial_out = self._linear_op(x, bias=None)

        # 2. 执行单次 AllReduce 汇总各 TP 分片部分和
        if self.tp_size > 1:
            if self.all_reduce_fn is not None:
                partial_out = self.all_reduce_fn(partial_out)
            elif dist.is_available() and dist.is_initialized():
                dist.all_reduce(partial_out, op=dist.ReduceOp.SUM, group=self.tp_group)

        # 3. 汇总后加上 Bias
        if self.bias is not None:
            partial_out = partial_out + self.bias

        return partial_out
