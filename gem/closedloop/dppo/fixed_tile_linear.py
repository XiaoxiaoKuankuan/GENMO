"""当前stable_tf32x3主矩阵乘与IEEE条件/GRU矩阵乘。

保持已验收的固定K归约、输出tile、流水调度和TF32x3补偿乘法；batch、尾批和
置换只改变独立输出网格，既不复制样本也不切换随机核。参数、输入与累加均为
FP32；敏感最终投影在上层使用FP64。本模块不提供BF16、宽tile或自动调参候选，
不支持的设备明确失败，不以另一种算子静默替换正式路径。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _kernel():
    @triton.jit
    def multiply(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 AS0: tl.constexpr, AS1: tl.constexpr, BS0: tl.constexpr, BS1: tl.constexpr,
                 PRECISION: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 PIPELINED: tl.constexpr = False):
        if PIPELINED:
            # L2分组访问只改变独立输出tile的执行顺序，不改每个元素的K归约。
            pid = tl.program_id(0)
            number_m, number_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
            group = pid // (8 * number_n)
            first_m = group * 8
            group_m = tl.minimum(number_m - first_m, 8)
            row_tile = first_m + pid % group_m
            col_tile = (pid % (8 * number_n)) // group_m
        else:
            row_tile, col_tile = tl.program_id(0), tl.program_id(1)
        rows = row_tile*BM+tl.arange(0, BM)
        cols = col_tile*BN+tl.arange(0, BN)
        reduction = tl.arange(0, BK)
        total = tl.full((BM, BN), 0, tl.float32)
        for offset in range(tl.cdiv(K, BK)):
            kk = offset*BK+reduction
            left = tl.load(A+rows[:, None]*AS0+kk[None, :]*AS1,
                           (rows[:, None]<M)&(kk[None, :]<K), 0)
            right = tl.load(B+kk[:, None]*BS0+cols[None, :]*BS1,
                            (kk[:, None]<K)&(cols[None, :]<N), 0)
            if PIPELINED:
                # 让矩阵乘累加直接消费既有累加器，避免每个K块先物化独立乘积。
                total = tl.dot(left, right, total, input_precision=PRECISION)
            else:
                total += tl.dot(left, right, input_precision=PRECISION)
        tl.store(C+rows[:, None]*N+cols[None, :], total,
                 (rows[:, None]<M)&(cols[None, :]<N))
    return multiply


_MULTIPLY = None


def fixed_matmul(left, right, *, precision='ieee'):
    """输入为二维CUDA矩阵，允许转置权重；输出dtype与输入相同。"""
    global _MULTIPLY
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError('Invalid fixed-tile matrix dimensions')
    if not left.is_cuda or not right.is_cuda or left.dtype != right.dtype:
        raise ValueError('Fixed-tile candidate requires equal CUDA dtypes')
    pipelined = precision.endswith('_pipelined')
    precision = precision.removesuffix('_pipelined')
    if left.dtype!=torch.float32 or precision not in ('ieee','tf32x3'):
        raise ValueError('Only current FP32 IEEE/TF32x3 arithmetic is supported')
    if _MULTIPLY is None: _MULTIPLY = _kernel()
    m, k = left.shape; n = right.shape[1]
    output = torch.empty((m, n), device=left.device, dtype=left.dtype)
    bm,bn,bk=(32,64,32) if pipelined or precision=='ieee' else (64,128,64)
    if m and n:
        grid = (((m+bm-1)//bm)*((n+bn-1)//bn),) if pipelined else ((m+bm-1)//bm, (n+bn-1)//bn)
        _MULTIPLY[grid](left, right, output, m, n, k,
            *left.stride(), *right.stride(), precision, bm, bn, bk, pipelined, num_warps=4, num_stages=3)
    return output
