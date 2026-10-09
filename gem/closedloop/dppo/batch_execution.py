"""第二阶段实验性行独立 FP32 线性算子与真实批量执行身份。

旧实现将全部 batch/time 行压入一个 GEMM，改变 batch 会改变矩阵形状及舍入路径，
微小均值误差在末端标准差 0.001 下被联合概率放大。本模块尝试将每个样本保持为
独立矩阵，以 strided batched GEMM 执行 Linear；没有复制样本充当有效训练数据，
没有改变联合概率、CFG 或随机核方差。保留逐样本FP64归约参考，另提供联合GEMM
和有界分块归约候选；微批累计也有显式FP32/选择性FP64模式。候选必须单独验收。
它属于新的数值执行契约，必须在真实模型上通过原概率和全部梯度门槛
后显式启用；不能用于旧 run 的透明恢复。模型参数对象、键名和冻结规则保持不变。
"""
from __future__ import annotations

import torch
from torch import nn
from gem.network.base_arch.transformer.encoder_rope import EncoderRoPEBlock

ROW_BMM = 'sample_matrix_bmm_fp32.v1'
WEIGHT_REDUCTIONS = ('sample_bmm', 'joint_gemm', 'chunked_gemm', 'bounded_sample_bmm', 'joint_gemm_fp64')
GRADIENT_ACCUMULATIONS = ('fp64_reference', 'fp32', 'selective_fp64')


def _blocked_fp32_gemm(value, matrix, *, capacity=64):
    """固定真实样本块的FP32联合GEMM候选，M维不随外部microbatch改变。

    每个完整块含64个网络行（CFG后的条件/无条件行）；尾块仅补零并立即裁掉，
    不复制任何有效样本，也不把补零行计入概率、梯度或吞吐。外部B不受64限制。
    固定每块时间长度与GEMM几何，允许cuBLAS使用大矩阵算子，同时单独验收
    置换、不同batch和尾块的数值一致性；它不是已通过验收的默认路径。
    """
    shaped = value.reshape(value.shape[0], -1, value.shape[-1])
    outputs = []
    for begin in range(0, len(shaped), capacity):
        part = shaped[begin:begin+capacity]
        count = len(part)
        if count < capacity:
            part = torch.cat((part, part.new_zeros(capacity-count,*part.shape[1:])),0)
        result = (part.reshape(-1, part.shape[-1]) @ matrix).reshape(capacity, shaped.shape[1], matrix.shape[1])
        outputs.append(result[:count])
    result = outputs[0] if len(outputs)==1 else torch.cat(outputs,0)
    return result.reshape(*value.shape[:-1], matrix.shape[1])


class _SampleLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, bias, reduction, forward_backend='sample_bmm'):
        ctx.save_for_backward(value, weight)
        ctx.has_bias = bias is not None
        ctx.reduction = ('sample_bmm' if reduction else 'joint_gemm') if isinstance(reduction, bool) else reduction
        ctx.forward_backend = forward_backend
        shaped = value.reshape(value.shape[0], -1, value.shape[-1])
        if forward_backend == 'blocked64_fp32_gemm':
            result = _blocked_fp32_gemm(shaped, weight.t())
        elif forward_backend.startswith('fixed_tile_'):
            from .fixed_tile_linear import fixed_matmul
            result = fixed_matmul(shaped.reshape(-1, shaped.shape[-1]), weight.t(),
                precision=forward_backend.removeprefix('fixed_tile_')).reshape(*shaped.shape[:-1], weight.shape[0])
        else:
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
                      if ctx.needs_input_grad[0] and ctx.forward_backend=='sample_bmm' else None)
        if ctx.needs_input_grad[0] and ctx.forward_backend=='blocked64_fp32_gemm':
            grad_value = _blocked_fp32_gemm(gradient, weight)
        if ctx.needs_input_grad[0] and ctx.forward_backend.startswith('fixed_tile_'):
            from .fixed_tile_linear import fixed_matmul
            grad_value = fixed_matmul(flat, weight,
                precision=ctx.forward_backend.removeprefix('fixed_tile_')).reshape_as(value)
        grad_weight = None
        if ctx.needs_input_grad[1] and ctx.reduction == 'joint_gemm':
            grad_weight = flat.t() @ inputs
        elif ctx.needs_input_grad[1] and ctx.reduction == 'joint_gemm_fp64':
            # 仅已定位的敏感窄输出投影使用；不物化逐样本完整权重梯度。
            grad_weight = (flat.double().t() @ inputs.double()).to(weight.dtype)
        elif ctx.needs_input_grad[1] and ctx.reduction == 'sample_bmm':
            # 每个样本沿时间维的归约也固定形状；随后高精度合并样本贡献。
            per_sample = torch.bmm(gradient.reshape(value.shape[0], -1, gradient.shape[-1]).transpose(1, 2),
                                  value.reshape(value.shape[0], -1, value.shape[-1]))
            grad_weight = per_sample.sum(0, dtype=torch.float64).to(weight.dtype)
        elif ctx.needs_input_grad[1] and ctx.reduction == 'chunked_gemm':
            # 高精度仅用于归并最终矩阵；每个GEMM仍FP32，临时空间不随样本数增长。
            total = torch.zeros_like(weight, dtype=torch.float64)
            for start in range(0, len(flat), 2048):
                total.add_(flat[start:start+2048].t() @ inputs[start:start+2048])
            grad_weight = total.to(weight.dtype)
        elif ctx.needs_input_grad[1] and ctx.reduction == 'bounded_sample_bmm':
            # 原参考的逐样本时间归约保持不变，但最多4份临时权重矩阵。
            total = torch.zeros_like(weight, dtype=torch.float64)
            shaped_gradient = gradient.reshape(value.shape[0], -1, gradient.shape[-1])
            shaped_input = value.reshape(value.shape[0], -1, value.shape[-1])
            for start in range(0, len(value), 4):
                partial = torch.bmm(shaped_gradient[start:start+4].transpose(1, 2), shaped_input[start:start+4])
                total.add_(partial.sum(0, dtype=torch.float64))
            grad_weight = total.to(weight.dtype)
        grad_bias = (gradient.reshape(value.shape[0], -1, gradient.shape[-1]).sum(1).sum(0, dtype=torch.float64)
                     .to(gradient.dtype) if ctx.has_bias and ctx.needs_input_grad[2] else None)
        result = (grad_value, grad_weight, grad_bias, None, None)
        return result[:len(ctx.needs_input_grad)]


class SampleMatrixLinear(nn.Linear):
    def forward(self, value):
        if value.ndim < 2:
            raise ValueError('SampleMatrixLinear requires explicit sample dimension')
        reduction = getattr(self, 'weight_reduction', None)
        if reduction is None:
            reduction = 'sample_bmm' if getattr(self, 'stable_weight_rows', True) else 'joint_gemm'
        reduction = getattr(self, 'weight_reduction_override', None) or reduction
        dtype = getattr(self, 'compute_dtype', self.weight.dtype)
        result = _SampleLinear.apply(value.to(dtype), self.weight.to(dtype),
                                     None if self.bias is None else self.bias.to(dtype), reduction,
                                     getattr(self, 'forward_backend', 'sample_bmm'))
        return result.to(getattr(self, 'result_dtype', result.dtype))


def sample_gru_cell(value, state, cell):
    """批量但逐行固定的GRU矩阵形状；CUDA融合门运算沿用PyTorch GRUCell定义。"""
    reduction = getattr(cell, 'weight_reduction', 'joint_gemm')
    dtype = getattr(cell, 'gate_accumulation_dtype', value.dtype)
    backend = getattr(cell, 'forward_backend', 'sample_bmm')
    inputs = _SampleLinear.apply(value[:, None].to(dtype), cell.weight_ih.to(dtype), None, reduction, backend).squeeze(1).to(value.dtype)
    hidden = _SampleLinear.apply(state[:, None].to(dtype), cell.weight_hh.to(dtype), None, reduction, backend).squeeze(1).to(value.dtype)
    if value.is_cuda:
        return torch.ops.aten._thnn_fused_gru_cell(inputs, hidden, state, cell.bias_ih, cell.bias_hh)[0]
    # CPU仅为可读数学参考，生产验收必须在服务器1 CUDA进行。
    ir, iz, inn = (inputs + cell.bias_ih).chunk(3, 1)
    hr, hz, hn = (hidden + cell.bias_hh).chunk(3, 1)
    reset, update = (ir+hr).sigmoid(), (iz+hz).sigmoid()
    new = (inn+reset*hn).tanh()
    return new + update*(state-new)


def configure_gradients(actor, *, weight_reduction='sample_bmm', accumulation='fp64_reference',
                        sensitive_names=()):
    """只选择反向与累计实现；不修改参数、前向、旧概率或优化器状态。"""
    if weight_reduction not in WEIGHT_REDUCTIONS or accumulation not in GRADIENT_ACCUMULATIONS:
        raise ValueError('Unknown explicit gradient execution candidate')
    for module in actor.modules():
        if isinstance(module, SampleMatrixLinear):
            module.weight_reduction = weight_reduction
    known = dict(actor.named_parameters())
    if any(name not in known for name in sensitive_names):
        raise ValueError('Selective FP64 requires exact existing parameter names')
    actor.gradient_accumulation = accumulation
    actor.gradient_sensitive_names = frozenset(sensitive_names)


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
    def __init__(self, module, mode=None):
        self.mode = mode or getattr(module, 'gradient_accumulation', 'fp64_reference')
        if self.mode not in GRADIENT_ACCUMULATIONS:
            raise ValueError('Unknown gradient accumulation mode')
        self.parameters = tuple((name, p) for name, p in module.named_parameters() if p.requires_grad)
        self.sensitive = getattr(module, 'gradient_sensitive_names', frozenset())
        self.totals = {}
        self.traffic_bytes = 0

    @torch.no_grad()
    def add(self):
        existing, values = [], []
        for name, parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None: continue
            dtype = torch.float64 if self.mode == 'fp64_reference' or self.mode == 'selective_fp64' and name in self.sensitive else torch.float32
            if parameter not in self.totals:
                self.totals[parameter] = gradient.to(dtype=dtype, copy=True)
                self.traffic_bytes += gradient.numel() * (gradient.element_size() + self.totals[parameter].element_size())
            else:
                existing.append(self.totals[parameter]); values.append(gradient)
                self.traffic_bytes += gradient.numel() * (gradient.element_size() + 2*self.totals[parameter].element_size())
            parameter.grad = None
        if existing: torch._foreach_add_(existing, values)

    @torch.no_grad()
    def finish(self):
        for parameter, total in self.totals.items():
            parameter.grad = total.to(parameter.dtype)
            if total.dtype != parameter.dtype:
                self.traffic_bytes += total.numel() * (total.element_size() + parameter.element_size())
        self.totals.clear()
