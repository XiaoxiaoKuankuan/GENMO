"""Stage10 v3 的真实无梯度采样、原生批量 GEMM 与区域精度策略合同。

复用 Stage1 Actor 的条件、CFG、20步DDIM均值、前缀约束和物理解码，不继承
旧固定64行编译器的有梯度推理补救。网络行随真实 batch 扩大，条件批量编码，
音乐/历史/前缀在一条链的20步中复用；训练条件图由 ConditionGraphCache 回传。
FP32主参数和Adam始终保留，原生GEMM直接归约权重梯度，不物化逐样本权重矩阵。
显式补偿BF16权重梯度候选复用已有三项/六项Tensor Core算子，只替换Transformer
反向权重归约；条件和敏感输出保持原精度。它不改变采样前向，仍须完整梯度及
非空Adam对照，默认继续使用原生联合GEMM，不能由可调用直接视为验收通过。

native_fp32是新数值基线，tf32和bf16_backbone属于显式候选合同。BF16只计算
Transformer大矩阵和指定SDPA，层间残差、条件、输出投影、CFG和DDIM仍FP32。
敏感最终窄投影可使用局部FP64计算；概率和KL在独立模块维持FP64。采样的old
statistics来自实际使用的路径，任何候选必须通过自身零更新和完整梯度/Adam
验收后才能用于生产；没有自动降低概率门槛或静默恢复旧run。

末端floor是新随机核身份，基础DDIM方向系数仍明确使用原eta日程；实际高斯
采样和概率均使用max(base_sigma,显式floor)。完整四秒自由坐标保持优化资格。
torch.compile只作用于纯denoiser候选，并分别验证no_grad与有梯度执行，不强制
启用梯度来伪造推理数值一致性。调用边界采用既有延后GPU检查机制。
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from .policy import DPPODiffusionPolicy
from .batch_execution import SampleMatrixLinear, SampleEncoderRoPEBlock
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
        if backend=='sample_bmm' and not precise_output:
            shaped=x.reshape(len(x),-1,x.shape[-1])
            result=torch.bmm(shaped,w.t().unsqueeze(0).expand(len(x),-1,-1)).reshape(*x.shape[:-1],w.shape[0])
            if bias is not None:result=result+bias.to(arithmetic)
        elif backend!='native_gemm' and not precise_output:
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
            if ctx.backend=='sample_bmm' and not ctx.precise_output:
                grad_x=torch.bmm(gradient.reshape(len(x),-1,gradient.shape[-1]).to(w.dtype),
                    w.unsqueeze(0).expand(len(x),-1,-1))
            elif ctx.backend!='native_gemm' and not ctx.precise_output:
                from .fixed_tile_linear import fixed_matmul
                grad_x=fixed_matmul(g.to(w.dtype),w,precision=ctx.backend)
            else:grad_x=F.linear(g.to(w.dtype),w.t())
            grad_x=grad_x.reshape_as(x).to(ctx.input_dtype)
        # 直接得到最终[out,in]矩阵；不创建[B,out,in]临时梯度。
        grad_w=None
        if ctx.needs_input_grad[1]:
            if ctx.weight_gradient!='direct_joint_gemm' and not ctx.precise_output:
                from .compensated_gemm import compensated_gemm
                grad_w=compensated_gemm(g.float().t(),inputs.float(),int(ctx.weight_gradient[-1])).to(ctx.weight_dtype)
            else:grad_w=(g.to(accumulation).t()@inputs.to(accumulation)).to(ctx.weight_dtype)
        grad_b=g.sum(0,dtype=accumulation).to(ctx.weight_dtype) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return grad_x,grad_w,grad_b,None,None,None,None


class RegionalDenseLinear(nn.Linear):
    def forward(self, inputs):
        return _DenseLinear.apply(inputs,self.weight,self.bias,
            getattr(self,'v3_compute_dtype',torch.float32),getattr(self,'v3_precise_output',False),
            getattr(self,'v3_linear_backend','native_gemm'),getattr(self,'v3_weight_gradient','direct_joint_gemm'))


def configure_dense_actor(actor, *, precision='native_fp32', attention='sdpa_math', precise_output=True,linear_execution='native_gemm',
                          weight_gradient='direct_joint_gemm'):
    if precision not in ('native_fp32','tf32','bf16_backbone'):
        raise ValueError('Unknown explicit v3 arithmetic contract')
    if attention not in ('manual','sdpa_math','sdpa_efficient','sdpa_flash','sdpa_cudnn'):
        raise ValueError('Unknown forced attention backend')
    if any(p.dtype!=torch.float32 for p in actor.parameters()):
        raise TypeError('Actor master parameters must remain FP32')
    if linear_execution not in ('native_gemm','stable_bmm','stable_ieee','stable_tf32x3','stable_bf16x3','stable_bf16x6'):
        raise ValueError('Unknown v3 dense reduction contract')
    if weight_gradient not in ('direct_joint_gemm','cublas_bf16x3','cublas_bf16x6'):
        raise ValueError('Unknown explicit v3 weight-gradient arithmetic contract')
    for name,module in actor.named_modules():
        if type(module) in (nn.Linear,SampleMatrixLinear,RegionalDenseLinear):
            module.__class__=RegionalDenseLinear
            module.v3_compute_dtype=(torch.bfloat16 if precision=='bf16_backbone' and name.startswith('denoiser.blocks.') else torch.float32)
            module.v3_precise_output=bool(precise_output and name=='denoiser.final_layer.fc2')
            module.v3_weight_gradient=weight_gradient if name.startswith('denoiser.blocks.') else 'direct_joint_gemm'
            module.v3_linear_backend=('native_gemm' if linear_execution=='native_gemm' else
                'sample_bmm' if linear_execution=='stable_bmm' and name.startswith('denoiser.blocks.') else
                'ieee' if linear_execution=='stable_bmm' else
                linear_execution.removeprefix('stable_')+'_pipelined' if name.startswith('denoiser.blocks.') else 'ieee')
        if type(module) is SampleEncoderRoPEBlock:module.__class__=EncoderRoPEBlock
        if isinstance(module,RoPEAttention):
            module.execution_backend=attention
            module.attention_compute_dtype=torch.bfloat16 if precision=='bf16_backbone' else torch.float32
    # 批量GRUCell原有跳过无效历史语义保留，避免按链递归batch=1。
    actor.history_encoder.execution_backend='native' if linear_execution=='native_gemm' else 'sample_bmm_fused_gru'
    if linear_execution!='native_gemm':
        actor.history_encoder.cell.forward_backend='fixed_tile_ieee'
        actor.history_encoder.cell.weight_reduction='joint_gemm'
    return dict(schema='genmo.stage10.native_dense_execution.v1',precision=precision,
        linear=linear_execution,weight_gradient=weight_gradient,attention=attention,attention_fallback=False,
        mode='bf16_backbone_candidate' if precision=='bf16_backbone' else
             'tf32_candidate' if precision=='tf32' else
             'compensated_tensorcore_fp32_storage_candidate' if linear_execution in ('stable_tf32x3','stable_bf16x3','stable_bf16x6') else 'fp32_fast',
        matmul_tf32=precision=='tf32',
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
                 precise_output=True,linear_execution='native_gemm',weight_gradient='direct_joint_gemm',steps=20,eta=.1,guidance_scale=2.5):
        if steps!=20 or len(std_schedule)!=20:
            raise ValueError('v3 production behavior must retain the full 20-step chain')
        super().__init__(actor,steps=steps,eta=eta,std_floor=min(std_schedule),std_schedule=std_schedule,
            guidance_scale=guidance_scale,cfg_batch=True,numerical_layout='legacy_step_lane',defer_checks=True)
        self.numerical_layout='native_dense_v3'
        self.batched_conditions=True
        self.precision=precision
        self.numerical_execution=configure_dense_actor(actor,precision=precision,attention=attention,
            precise_output=precise_output,linear_execution=linear_execution,weight_gradient=weight_gradient)
        if compile_denoiser:
            from .compiled_forward_v3 import install_shared_forward
            install_shared_forward(self)
        else:self.numerical_execution['compiler']=None
        self.kernel_config.update(version='genmo.stochastic_ddim.actual_kernel.v3',
            execution_contract=self.numerical_layout,numerical_execution=self.numerical_execution,
            condition_batch_size='unique_chains',execution_batch_size=None,
            network_batch='actual_rows_no_fixed64_tiles',effective_std_floors=list(std_schedule))

    def prepare_conditions(self,conditions):
        with dense_precision_scope(self.precision):return super().prepare_conditions(conditions)

    def _transition_prepared(self,prepared,x_k,step_index,*,diagnostics=False):
        with dense_precision_scope(self.precision):
            return super()._transition_prepared(prepared,x_k,step_index,diagnostics=diagnostics)
