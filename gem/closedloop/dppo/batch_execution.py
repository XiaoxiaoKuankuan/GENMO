"""第二阶段实验性行独立 FP32 线性算子与真实批量执行身份。

旧实现将全部 batch/time 行压入一个 GEMM，改变 batch 会改变矩阵形状及舍入路径，
微小均值误差在末端标准差 0.001 下被联合概率放大。本模块尝试将每个样本保持为
独立矩阵，以 strided batched GEMM 执行 Linear；没有复制样本充当有效训练数据，
没有改变联合概率、CFG 或随机核方差。反向逐样本矩阵计算后以FP64归约权重梯度，
微批之间也以FP64累计后一次转回参数精度；较快的联合GEMM仅保留为显式诊断候选。
它属于新的数值执行契约，必须在真实模型上通过原概率和全部梯度门槛
后显式启用；不能用于旧 run 的透明恢复。模型参数对象、键名和冻结规则保持不变。
"""
from __future__ import annotations

import torch
from torch import nn
from gem.network.base_arch.transformer.encoder_rope import EncoderRoPEBlock

ROW_BMM = 'sample_matrix_bmm_fp32.v1'


class _SampleLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, bias, stable_weight_rows):
        ctx.save_for_backward(value, weight)
        ctx.has_bias = bias is not None
        ctx.stable_weight_rows = stable_weight_rows
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
        # 输入梯度也逐样本保持相同矩阵形状。原flatten GEMM会随微批改变
        # 舍入路径；上游gate等接近相消的梯度会放大这个差异。
        grad_value = (torch.bmm(gradient.reshape(value.shape[0], -1, gradient.shape[-1]),
                               weight.unsqueeze(0).expand(value.shape[0], -1, -1)).reshape_as(value)
                      if ctx.needs_input_grad[0] else None)
        grad_weight = None
        if ctx.needs_input_grad[1] and not ctx.stable_weight_rows:
            grad_weight = flat.t() @ inputs
        elif ctx.needs_input_grad[1]:
            # 每个样本沿时间维的归约也固定形状；随后高精度合并样本贡献。
            per_sample = torch.bmm(gradient.reshape(value.shape[0], -1, gradient.shape[-1]).transpose(1, 2),
                                  value.reshape(value.shape[0], -1, value.shape[-1]))
            grad_weight = per_sample.sum(0, dtype=torch.float64).to(weight.dtype)
        grad_bias = (gradient.reshape(value.shape[0], -1, gradient.shape[-1]).sum(1).sum(0, dtype=torch.float64)
                     .to(gradient.dtype) if ctx.has_bias and ctx.needs_input_grad[2] else None)
        return grad_value, grad_weight, grad_bias, None


class SampleMatrixLinear(nn.Linear):
    def forward(self, value):
        if value.ndim < 2:
            raise ValueError('SampleMatrixLinear requires explicit sample dimension')
        return _SampleLinear.apply(value, self.weight, self.bias, getattr(self,'stable_weight_rows',True))


class _SampleGate(torch.autograd.Function):
    """保持广播乘法前向，门控梯度先按单样本时间维归约，再以FP64合并样本。"""
    @staticmethod
    def forward(ctx, gate, value):
        ctx.save_for_backward(gate, value)
        return gate * value

    @staticmethod
    def backward(ctx, gradient):
        gate, value = ctx.saved_tensors
        gate_gradient = None
        if ctx.needs_input_grad[0]:
            per_sample = (gradient * value).sum(1)
            gate_gradient = per_sample.sum(0, dtype=torch.float64).to(gate.dtype).reshape_as(gate)
        return gate_gradient, gradient * gate if ctx.needs_input_grad[1] else None


class SampleEncoderRoPEBlock(EncoderRoPEBlock):
    """只替换两处门控的反向归约；所有参数、归一化和attention公式保持原样。"""
    def forward(self, x, attn_mask=None, tgt_key_padding_mask=None):
        x = x + _SampleGate.apply(self.gate_msa, self._sa_block(
            self.norm1(x), attn_mask=attn_mask, key_padding_mask=tgt_key_padding_mask))
        return x + _SampleGate.apply(self.gate_mlp, self.mlp(self.norm2(x)))


def set_sample_linear(module, enabled):
    """原位切换实现，不替换 Parameter，因此优化器引用和共享权重关系不变。"""
    count = 0
    for child in module.modules():
        if type(child) in (EncoderRoPEBlock, SampleEncoderRoPEBlock):
            child.__class__ = SampleEncoderRoPEBlock if enabled else EncoderRoPEBlock
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


class MicrobatchGradientAccumulator:
    """以FP64合并微批梯度，消除小微批反复FP32写回造成的抵消误差。

    网络前后向仍FP32；只对固定同一参数版本的梯度求和使用FP64缓冲。每次微批
    backward后归并并清空叶子梯度，完成后仅转回一次FP32，随后执行原SUM通信、
    BC、裁剪和Adam。默认无梯度的参数不分配缓存，条件图最后一次反传另行合并。
    """
    def __init__(self, module):
        self.parameters = tuple(p for p in module.parameters() if p.requires_grad)
        self.totals = {}

    @torch.no_grad()
    def add(self):
        existing, values = [], []
        for parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None: continue
            if parameter not in self.totals:
                self.totals[parameter] = gradient.double()
            else:
                existing.append(self.totals[parameter]); values.append(gradient)
            parameter.grad = None
        if existing: torch._foreach_add_(existing, values)

    @torch.no_grad()
    def finish(self):
        for parameter, total in self.totals.items():
            parameter.grad = total.to(parameter.dtype)
        self.totals.clear()
