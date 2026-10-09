"""第二阶段实验性行独立 FP32 线性算子与真实批量执行身份。

旧实现将全部 batch/time 行压入一个 GEMM，改变 batch 会改变矩阵形状及舍入路径，
微小均值误差在末端标准差 0.001 下被联合概率放大。本模块尝试将每个样本保持为
独立矩阵，以 strided batched GEMM 执行 Linear；没有复制样本充当有效训练数据，
没有改变联合概率、CFG 或随机核方差。反向使用全批矩阵乘法聚合权重梯度，避免
建立 B 份大型参数梯度。它属于新的数值执行契约，必须在真实模型上通过原概率门槛
后显式启用；不能用于旧 run 的透明恢复。模型参数对象、键名和冻结规则保持不变。
"""
from __future__ import annotations

import torch
from torch import nn

ROW_BMM = 'sample_matrix_bmm_fp32.v1'


class _SampleLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, bias):
        ctx.save_for_backward(value, weight)
        ctx.has_bias = bias is not None
        shaped = value.reshape(value.shape[0], -1, value.shape[-1])
        result = torch.bmm(shaped, weight.t().unsqueeze(0).expand(len(value), -1, -1))
        if bias is not None:
            result = result + bias
        return result.reshape(*value.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, gradient):
        value, weight = ctx.saved_tensors
        flat = gradient.reshape(-1, gradient.shape[-1])
        inputs = value.reshape(-1, value.shape[-1])
        grad_value = (flat @ weight).reshape_as(value) if ctx.needs_input_grad[0] else None
        grad_weight = flat.t() @ inputs if ctx.needs_input_grad[1] else None
        grad_bias = flat.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return grad_value, grad_weight, grad_bias


class SampleMatrixLinear(nn.Linear):
    def forward(self, value):
        if value.ndim < 2:
            raise ValueError('SampleMatrixLinear requires explicit sample dimension')
        return _SampleLinear.apply(value, self.weight, self.bias)


def set_sample_linear(module, enabled):
    """原位切换实现，不替换 Parameter，因此优化器引用和共享权重关系不变。"""
    count = 0
    for child in module.modules():
        if type(child) in (nn.Linear, SampleMatrixLinear):
            child.__class__ = SampleMatrixLinear if enabled else nn.Linear
            count += 1
    return count


def batch_accounting(effective_rows, *, cfg, network_rows=None):
    actual = effective_rows if network_rows is None else network_rows
    if effective_rows < 1 or actual < effective_rows:
        raise ValueError('Invalid effective/network batch rows')
    branches = 2 if cfg else 1
    return dict(effective_internal_rows=effective_rows, network_rows=actual,
                cfg_rows=actual*branches, padding_rows=(actual-effective_rows)*branches,
                useful_row_fraction=effective_rows/actual)
