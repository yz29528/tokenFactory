from typing import Optional
import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _block_fp8_gemm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        w_scale_ptr,
        bias_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        stride_sn,
        stride_sk,
        BLOCK_H: tl.constexpr,
        BLOCK_W: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        单个 Triton Kernel 完成块级 FP8 (weight_block_size=[BLOCK_H, BLOCK_W]) 矩阵乘:
        C[M, N] = (A_fp8[M, K] * a_scale) @ (B_fp8[N, K]^T * w_scale[N//BLOCK_H, K//BLOCK_W]) + bias
        支持任意 M (含 Decode M=1) 与非对齐 K/N 边界掩码，消除 Python 循环开销。
        """
        pid = tl.program_id(axis=0)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        # B 存储为 [N, K]，按 [K, N] 视角加载
        b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

        a_scale = tl.load(a_scale_ptr).to(tl.float32)
        n_scale_idx = offs_n // BLOCK_H
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_mask = (k_start + offs_k) < K
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & k_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            k_scale_idx = k_start // BLOCK_W
            w_scale = tl.load(
                w_scale_ptr + n_scale_idx * stride_sn + k_scale_idx * stride_sk,
                mask=n_mask,
                other=1.0,
            ).to(tl.float32)

            partial = tl.dot(a, b, out_dtype=tl.float32)
            acc += partial * (a_scale * w_scale[None, :])

            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc += bias[None, :]

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & n_mask[None, :]
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def can_use_triton_fp8_gemm(x: torch.Tensor) -> bool:
    """检查当前环境与设备是否支持 Triton Block-FP8 GEMM (SM89+ CUDA 且已安装 triton)。"""
    if not (HAS_TRITON and x.is_cuda and hasattr(torch, "float8_e4m3fn")):
        return False
    major, minor = torch.cuda.get_device_capability(x.device)
    return (major, minor) >= (8, 9)


def triton_block_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    block_h: int = 128,
    block_w: int = 128,
) -> torch.Tensor:
    """
    使用单个 Triton Block-FP8 GEMM Kernel 执行动态激活量化 + 块级权重缩放线性投影。
    """
    assert HAS_TRITON, "未安装 triton，无法调用 triton_block_fp8_linear！"
    orig_shape = x.shape
    in_features = weight.shape[1]
    out_features = weight.shape[0]
    x_2d = x.reshape(-1, in_features).contiguous()
    m = x_2d.shape[0]

    fp8_max = 448.0
    amax = x_2d.abs().max().clamp(min=1e-12)
    x_scale = (amax / fp8_max).to(torch.float32)
    x_fp8 = (x_2d.float() / x_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    if weight_scale_inv is None:
        w_scale = torch.ones((1, 1), dtype=torch.float32, device=x.device)
        eff_block_h = out_features
        eff_block_w = in_features
    else:
        w_scale = weight_scale_inv.contiguous().to(torch.float32)
        eff_block_h = block_h
        eff_block_w = block_w

    out = torch.empty((m, out_features), dtype=x.dtype, device=x.device)

    BLOCK_M = 16 if m <= 16 else (32 if m <= 32 else 64)
    BLOCK_N = 64
    # 确保 BLOCK_K 为 2 的幂（Triton tl.arange 要求）且整除 block_w（默认 128 -> BLOCK_K=64），
    # 从而每个 K 分块仅对应单个权重缩放块索引；边界由 k_mask 处理。
    if weight_scale_inv is None:
        BLOCK_K = 64 if in_features >= 64 else 32
    elif eff_block_w >= 64 and eff_block_w % 64 == 0:
        BLOCK_K = 64
    elif eff_block_w >= 32 and eff_block_w % 32 == 0:
        BLOCK_K = 32
    else:
        BLOCK_K = 16

    w_contig = weight.contiguous()
    grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(out_features, BLOCK_N),)
    _block_fp8_gemm_kernel[grid](
        x_fp8,
        w_contig,
        out,
        x_scale,
        w_scale,
        bias if bias is not None else out,
        m,
        out_features,
        in_features,
        x_fp8.stride(0),
        x_fp8.stride(1),
        w_contig.stride(0),
        w_contig.stride(1),
        out.stride(0),
        out.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        BLOCK_H=eff_block_h,
        BLOCK_W=eff_block_w,
        HAS_BIAS=bias is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out.reshape(*orig_shape[:-1], out_features)
