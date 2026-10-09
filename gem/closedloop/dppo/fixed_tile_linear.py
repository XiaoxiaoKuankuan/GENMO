"""Stage10固定归约顺序的联合矩阵乘法候选。

原逐样本BMM避免了批量变化时cuBLAS换算法造成的概率漂移，但重复读取权重、
小矩阵调度和标量FP32算力成为大批量瓶颈。本实现将全部真实样本/时间行连续
组织；IEEE 使用固定32×64×32块，Tensor Core候选使用64×128×64块。
batch、置换、尾批只改变输出网格，
不改变任何有效输出元素的归约顺序，不复制有效样本，也不修改概率或随机核。

FP32 fast显式使用IEEE乘法，TF32候选显式使用三项TF32分解，BF16候选使用
BF16乘法和FP32累加。主参数仍为FP32，转换由可微的显式cast完成。输入梯度使用
同一固定算子，权重梯度保留已独立验证的联合GEMM/分块选择。此模块不启用自动
调参，以免batch或运行时选核改变数值合同；不支持的设备明确失败、不静默回退。

仅CUDA生产候选会导入Triton。参考路径和历史checkpoint继续使用原实现。
该候选须通过真实模型零更新、全部梯度、Adam和闭环验证后才能进入正式训练。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


def _kernel():
    @triton.jit
    def multiply(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 AS0: tl.constexpr, AS1: tl.constexpr, BS0: tl.constexpr, BS1: tl.constexpr,
                 PRECISION: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        rows = tl.program_id(0)*BM+tl.arange(0, BM)
        cols = tl.program_id(1)*BN+tl.arange(0, BN)
        reduction = tl.arange(0, BK)
        total = tl.full((BM, BN), 0, tl.float32)
        for offset in range(tl.cdiv(K, BK)):
            kk = offset*BK+reduction
            left = tl.load(A+rows[:, None]*AS0+kk[None, :]*AS1,
                           (rows[:, None]<M)&(kk[None, :]<K), 0)
            right = tl.load(B+kk[:, None]*BS0+cols[None, :]*BS1,
                            (kk[:, None]<K)&(cols[None, :]<N), 0)
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
    if left.dtype not in (torch.float32, torch.bfloat16) or precision not in ('ieee', 'tf32x3'):
        raise ValueError('Unsupported explicit fixed-tile precision')
    if _MULTIPLY is None: _MULTIPLY = _kernel()
    m, k = left.shape; n = right.shape[1]
    output = torch.empty((m, n), device=left.device, dtype=left.dtype)
    bm, bn, bk = (64, 128, 64) if left.dtype == torch.bfloat16 or precision == 'tf32x3' else (32, 64, 32)
    if m and n:
        _MULTIPLY[((m+bm-1)//bm, (n+bn-1)//bn)](left, right, output, m, n, k,
            *left.stride(), *right.stride(), precision, bm, bn, bk, num_warps=4, num_stages=3)
    return output
