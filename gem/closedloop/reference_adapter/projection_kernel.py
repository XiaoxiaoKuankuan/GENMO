"""有限时域QP的GPU批量前向核：每个残差通道一个十变量凸问题。

将ADMM迭代放在单个Triton核内，避免每20ms控制周期发射数千个小Torch算子。
输入仅是提案和线性导数约束，不读取执行未来状态。收敛容差和最大迭代明确固定；
返回原始数值解及对偶变量，随后仍由Torch完成KKT精化与可执行证书。反向导数
在reference_constraints的隐式活跃面实现，本核不做直通梯度或偷偷跳过约束。
CPU回归使用相同ADMM公式，独立SciPy作为最近点数值参照。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _solve(Y, L, U, A, INV, X, DUAL, ITERATIONS: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, 16)
    j = tl.arange(0, 32)
    y = tl.load(Y+row*10+i, i < 10, 0)
    lower = tl.load(L+row*30+j, j < 30, 0)
    upper = tl.load(U+row*30+j, j < 30, 0)
    a = tl.load(A+j[:, None]*10+i[None, :], (j[:, None] < 30) & (i[None, :] < 10), 0)
    inv = tl.load(INV+i[:, None]*10+i[None, :], (i[:, None] < 10) & (i[None, :] < 10), 0)
    x = y
    ax = tl.sum(a*x[None, :], 1)
    z = tl.minimum(tl.maximum(ax, lower), upper)
    dual = z*0
    for step in range(ITERATIONS):
        rhs = y+10*tl.sum(a*(z-dual)[:, None], 0)
        previous = x
        x = tl.sum(inv*rhs[:, None], 0)
        ax = tl.sum(a*x[None, :], 1)
        z = tl.minimum(tl.maximum(ax+dual, lower), upper)
        dual = dual+ax-z
        if step % 16 == 15:
            if (tl.max(tl.abs(ax-z), 0) < 1e-7) & (tl.max(tl.abs(x-previous), 0) < 1e-7):
                break
    tl.store(X+row*10+i, x, i < 10)
    tl.store(DUAL+row*30+j, dual, j < 30)


def solve(proposal, lower, upper, matrix, inverse, iterations):
    proposal, lower, upper = proposal.contiguous(), lower.contiguous(), upper.contiguous()
    x, dual = torch.empty_like(proposal), torch.empty_like(lower)
    _solve[(len(proposal),)](proposal, lower, upper, matrix.contiguous(), inverse.contiguous(), x, dual,
                            iterations, num_warps=4)
    return x, dual
