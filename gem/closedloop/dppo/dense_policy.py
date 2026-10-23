"""当前FP32主参数、stable_tf32x3大矩阵与SDPA数学注意力执行。

Actor真实60帧输入随有效batch计算，条件按当前参数版本复用；采样和KL无梯度，
训练保留完整条件编码梯度。敏感输出使用局部FP64，概率独立FP64，权重梯度使用
联合GEMM。GPU正式配置只接受已验收路径；CPU原生GEMM保留为数学回归工具。
旧BF16、TF32近似、编译、CUDA Graph及矩阵核扫描实现不再保留或自动回退。
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from .policy import DPPODiffusionPolicy
from gem.network.base_arch.transformer.encoder_rope import EncoderRoPEBlock, RoPEAttention


class _DenseLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, dtype, precise_output, backend, weight_gradient):
        arithmetic=torch.float64 if precise_output else dtype
        x,w=inputs.to(arithmetic),weight.to(arithmetic)
        ctx.save_for_backward(x,w)
        ctx.input_dtype,ctx.weight_dtype=inputs.dtype,weight.dtype
        ctx.has_bias=bias is not None
        ctx.precise_output=precise_output
        ctx.backend=backend
        ctx.weight_gradient=weight_gradient
        if backend!='native_gemm' and not precise_output:
            from .fixed_tile_linear import fixed_matmul
            result=fixed_matmul(x.reshape(-1,x.shape[-1]),w.t(),precision=backend).reshape(*x.shape[:-1],w.shape[0])
            if bias is not None:result=result+bias.to(arithmetic)
        else:result=F.linear(x,w,None if bias is None else bias.to(arithmetic))
        return result.to(inputs.dtype)

    @staticmethod
    def backward(ctx, gradient):
        x,w=ctx.saved_tensors
        g=gradient.reshape(-1,gradient.shape[-1])
        inputs=x.reshape(-1,x.shape[-1])
        accumulation=torch.float64 if ctx.precise_output else torch.float32
        grad_x=None
        if ctx.needs_input_grad[0]:
            if ctx.backend!='native_gemm' and not ctx.precise_output:
                from .fixed_tile_linear import fixed_matmul
                grad_x=fixed_matmul(g.to(w.dtype),w,precision=ctx.backend)
            else:grad_x=F.linear(g.to(w.dtype),w.t())
            grad_x=grad_x.reshape_as(x).to(ctx.input_dtype)
        # 直接得到最终[out,in]矩阵；不创建[B,out,in]临时梯度。
        grad_w=None
        if ctx.needs_input_grad[1]:
            grad_w=(g.to(accumulation).t()@inputs.to(accumulation)).to(ctx.weight_dtype)
        grad_b=g.sum(0,dtype=accumulation).to(ctx.weight_dtype) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return grad_x,grad_w,grad_b,None,None,None,None


class RegionalDenseLinear(nn.Linear):
    def forward(self, inputs):
        return _DenseLinear.apply(inputs,self.weight,self.bias,
            getattr(self,'dense_compute_dtype',torch.float32),getattr(self,'dense_precise_output',False),
            getattr(self,'dense_linear_backend','native_gemm'),getattr(self,'dense_weight_gradient','direct_joint_gemm'))


def configure_dense_actor(actor, *, precision='native_fp32', attention='sdpa_math', precise_output=True,linear_execution='native_gemm',
                          weight_gradient='direct_joint_gemm'):
    if precision!='native_fp32':
        raise ValueError('Unknown explicit 当前 arithmetic contract')
    if attention!='sdpa_math':
        raise ValueError('Unknown forced attention backend')
    if any(p.dtype!=torch.float32 for p in actor.parameters()):
        raise TypeError('Actor master parameters must remain FP32')
    if linear_execution not in ('native_gemm','stable_tf32x3'):
        raise ValueError('Unknown 当前 dense reduction contract')
    if weight_gradient!='direct_joint_gemm':
        raise ValueError('Unknown explicit 当前 weight-gradient arithmetic contract')
    for name,module in actor.named_modules():
        if type(module) in (nn.Linear,RegionalDenseLinear):
            module.__class__=RegionalDenseLinear
            module.dense_compute_dtype=torch.float32
            module.dense_precise_output=bool(precise_output and name=='denoiser.final_layer.fc2')
            module.dense_weight_gradient=weight_gradient if name.startswith('denoiser.blocks.') else 'direct_joint_gemm'
            module.dense_linear_backend=('native_gemm' if linear_execution=='native_gemm' else
                linear_execution.removeprefix('stable_')+'_pipelined' if name.startswith('denoiser.blocks.') else 'ieee')
        if isinstance(module,RoPEAttention):
            module.execution_backend=attention
            module.attention_compute_dtype=torch.float32
    # 批量GRUCell原有跳过无效历史语义保留，避免按链递归batch=1。
    actor.history_encoder.execution_backend='native' if linear_execution=='native_gemm' else 'sample_bmm_fused_gru'
    if linear_execution!='native_gemm':
        actor.history_encoder.cell.forward_backend='fixed_tile_ieee'
        actor.history_encoder.cell.weight_reduction='joint_gemm'
    return dict(schema='genmo.stage10.native_dense_execution.v1',precision=precision,
        linear=linear_execution,weight_gradient=weight_gradient,attention=attention,attention_fallback=False,
        mode='compensated_tensorcore_fp32_storage_candidate' if linear_execution=='stable_tf32x3' else 'fp32_fast',
        matmul_tf32=False,
        master_dtype='float32',layer_residual_dtype='float32',probability_dtype='float64',
        final_projection='float64_local' if precise_output else 'float32',
        inference_requires_grad=False,condition_encoding='batched_unique_chains',
        network_rows='actual_batch_cfg_expanded',torch_version=torch.__version__)


@contextmanager
def dense_precision_scope(precision):
    cuda, cudnn = torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=precision=='tf32'
    torch.backends.cudnn.allow_tf32=precision=='tf32'
    try:yield
    finally:
        torch.backends.cuda.matmul.allow_tf32=cuda
        torch.backends.cudnn.allow_tf32=cudnn


class DenseDiffusionPolicy(DPPODiffusionPolicy):
    def __init__(self,actor,*,std_schedule,precision='native_fp32',attention='sdpa_math',compile_denoiser=False,
                 precise_output=True,linear_execution='native_gemm',weight_gradient='direct_joint_gemm',steps=20,eta=.1,guidance_scale=2.5,generation_frames=120,deterministic_steps=(),cuda_graph=False,music_context_frames=None):
        if steps not in (10,20) or len(std_schedule)!=steps:
            raise ValueError('Production behavior requires an explicit complete 10 or 20 step chain')
        super().__init__(actor,steps=steps,eta=eta,std_floor=min(v for v in std_schedule if v>0),std_schedule=std_schedule,
            deterministic_steps=deterministic_steps,
            guidance_scale=guidance_scale,cfg_batch=True,defer_checks=True)
        self.numerical_layout='native_dense'
        self.batched_conditions=True
        self.precision=precision
        if generation_frames not in (15,30,60,120) or generation_frames>actor.denoiser.max_len:raise ValueError('Invalid actual generation horizon')
        self.generation_frames=generation_frames
        self.music_context_frames=generation_frames if music_context_frames is None else music_context_frames
        if self.music_context_frames!=generation_frames and not (generation_frames in (15,30) and self.music_context_frames==60):
            raise ValueError("Invalid music-only context horizon")
        self.numerical_execution=configure_dense_actor(actor,precision=precision,attention=attention,
            precise_output=precise_output,linear_execution=linear_execution,weight_gradient=weight_gradient)
        if compile_denoiser or cuda_graph:
            raise ValueError('Retired compile/CUDA Graph candidates are not supported')
        self.numerical_execution['compiler']=None
        self.kernel_config.update(version='genmo.stochastic_ddim.actual_kernel',
            execution_contract=self.numerical_layout,numerical_execution=self.numerical_execution,
            condition_batch_size='unique_chains',execution_batch_size=None,
            network_batch='actual_rows_no_fixed64_tiles',effective_std_floors=list(std_schedule))
        self.kernel_config.update(generation_horizon_frames=generation_frames,generation_fps=30)
        if self.music_context_frames!=generation_frames:
            self.kernel_config.update(music_context_frames=self.music_context_frames,context_contract="music_only_tail.v1")

    def prepare_conditions(self,conditions):
        if conditions['future_valid'].shape[1]!=self.generation_frames:raise ValueError('Condition horizon differs from actual behavior kernel')
        if conditions['music_valid'].shape[1]!=self.music_context_frames:raise ValueError('Music context differs from kernel')
        with dense_precision_scope(self.precision):return super().prepare_conditions(conditions)

    def _transition_prepared(self,prepared,x_k,step_index,*,diagnostics=False):
        with dense_precision_scope(self.precision):
            return super()._transition_prepared(prepared,x_k,step_index,diagnostics=diagnostics)
