"""当前历史GRU的固定IEEE矩阵乘与可解析CPU数学路径。

GPU输入/隐藏矩阵乘沿用已验收固定归约IEEE算子，权重梯度直接联合GEMM；门运算
沿用PyTorch GRUCell定义与CUDA融合实现。CPU分支用于无GPU回归和Stage1数学核对。
不再提供旧固定64行网络切块、BF16候选、逐样本权重归约或梯度配置扫描接口。
"""
from __future__ import annotations
import torch


class _SampleLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,value,weight,bias,reduction,forward_backend='sample_bmm'):
        if reduction!='joint_gemm' or forward_backend not in ('sample_bmm','fixed_tile_ieee'):
            raise ValueError('Only current GRU joint-gradient arithmetic is supported')
        ctx.save_for_backward(value,weight);ctx.has_bias=bias is not None
        ctx.forward_backend=forward_backend
        shaped=value.reshape(value.shape[0],-1,value.shape[-1])
        if forward_backend=='fixed_tile_ieee':
            from .fixed_tile_linear import fixed_matmul
            result=fixed_matmul(shaped.reshape(-1,shaped.shape[-1]),weight.t(),precision='ieee').reshape(*shaped.shape[:-1],weight.shape[0])
        else:
            result=torch.bmm(shaped,weight.t().unsqueeze(0).expand(len(value),-1,-1))
        if bias is not None:result=result+bias
        return result.reshape(*value.shape[:-1],weight.shape[0])

    @staticmethod
    def backward(ctx,gradient):
        value,weight=ctx.saved_tensors
        flat=gradient.reshape(-1,gradient.shape[-1]);inputs=value.reshape(-1,value.shape[-1])
        grad_value=None
        if ctx.needs_input_grad[0]:
            if ctx.forward_backend=='fixed_tile_ieee':
                from .fixed_tile_linear import fixed_matmul
                grad_value=fixed_matmul(flat,weight,precision='ieee').reshape_as(value)
            else:
                grad_value=torch.bmm(gradient.reshape(value.shape[0],-1,gradient.shape[-1]),
                    weight.unsqueeze(0).expand(value.shape[0],-1,-1)).reshape_as(value)
        grad_weight=flat.t()@inputs if ctx.needs_input_grad[1] else None
        grad_bias=(gradient.reshape(value.shape[0],-1,gradient.shape[-1]).sum(1).sum(0,dtype=torch.float64).to(gradient.dtype)
                   if ctx.has_bias and ctx.needs_input_grad[2] else None)
        return grad_value,grad_weight,grad_bias,None,None


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
