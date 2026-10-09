"""Stage10显式数值执行合同与候选精度配置。

参数和Adam始终FP32，概率、ratio、KL沿用FP64。参考合同保留旧逐条件编码、
普通注意力和FP32 BMM；fast合同批量编码条件、使用指定SDPA后端及联合权重梯度。
TF32和BF16是不同的行为策略执行合同，必须重新采样、建立基线，不允许静默恢复
旧run。BF16只进入Transformer块的大矩阵，条件、RoPE旋转、输出投影和DDIM保持
FP32；SDPA的q/k在FP32旋转后转回显式骨干dtype。强制单一SDPA后端，失败即报告，
不会自动回退后宣称Flash生效。数值验收和实际有效输出变化由诊断/训练入口记录。

本模块只配置算子，不放宽任何概率门槛，也不修改噪声、CFG、PPO或奖励公式。
全局TF32开关使用可嵌套上下文恢复，避免污染同进程的Stage1基线与其他策略。
"""
from contextlib import contextmanager
from contextvars import ContextVar

import torch

MODES = ('fp32_reference', 'fp32_fast', 'tf32_candidate', 'bf16_backbone_candidate')
ATTENTION_BACKENDS = ('manual', 'sdpa_math', 'sdpa_efficient', 'sdpa_flash', 'sdpa_cudnn')
_ACTIVE_PRECISION = ContextVar('stage10_active_precision', default=None)


def configure_numerics(actor, mode, attention_backend=None):
    from .batch_execution import configure_gradients, set_sample_linear, SampleMatrixLinear
    from gem.network.base_arch.transformer.encoder_rope import RoPEAttention
    if mode not in MODES:
        raise ValueError('Unknown explicit numerical execution mode')
    backend = attention_backend or ('manual' if mode == 'fp32_reference' else
        'sdpa_cudnn' if mode == 'bf16_backbone_candidate' else 'sdpa_math')
    if backend not in ATTENTION_BACKENDS:
        raise ValueError('Unknown forced attention backend')
    set_sample_linear(actor, mode != 'fp32_reference')
    set_sample_linear(actor.denoiser, True)
    for name, module in actor.named_modules():
        if isinstance(module, SampleMatrixLinear):
            module.forward_backend = ('sample_bmm' if mode == 'fp32_reference' else
                'fixed_tile_tf32x3' if mode == 'tf32_candidate' and name.startswith('denoiser.blocks.') else 'fixed_tile_ieee')
            module.compute_dtype = (torch.bfloat16 if mode == 'bf16_backbone_candidate'
                                    and name.startswith('denoiser.blocks.') else torch.float32)
            module.result_dtype = module.compute_dtype
        if isinstance(module, RoPEAttention):
            module.execution_backend = backend
    actor.history_encoder.execution_backend = 'native' if mode == 'fp32_reference' else 'sample_bmm_fused_gru'
    actor.history_encoder.cell.gate_accumulation_dtype = torch.float32
    actor.history_encoder.cell.forward_backend = 'sample_bmm' if mode == 'fp32_reference' else 'fixed_tile_ieee'
    configure_gradients(actor, weight_reduction='sample_bmm' if mode == 'fp32_reference' else 'joint_gemm')
    return dict(version='stage10.numerical_execution.v4', mode=mode, attention_backend=backend,
        attention_fallback=False, condition_encoding='scalar_reference' if mode == 'fp32_reference' else 'batched_unique_chains',
        history_backend=actor.history_encoder.execution_backend, master_dtype='float32',
        history_projection_accumulation='fixed_tile_ieee' if mode != 'fp32_reference' else 'native_float32',
        condition_linear_accumulation='fixed_tile_ieee' if mode != 'fp32_reference' else 'native_float32',
        linear_backend='sample_bmm' if mode == 'fp32_reference' else 'fixed_tile_32x64x32',
        backbone_multiply='tf32x3' if mode == 'tf32_candidate' else 'bf16' if mode == 'bf16_backbone_candidate' else 'ieee',
        backbone_dtype='bfloat16' if mode == 'bf16_backbone_candidate' else 'float32',
        output_cfg_ddim_dtype='float32', probability_dtype='float64',
        matmul_tf32=False, torch_version=torch.__version__)


@contextmanager
def precision_scope(contract):
    if contract is None:
        yield
        return
    desired = bool(contract['matmul_tf32'])
    if _ACTIVE_PRECISION.get() == desired:
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    token = _ACTIVE_PRECISION.set(desired)
    torch.backends.cuda.matmul.allow_tf32 = desired
    torch.backends.cudnn.allow_tf32 = desired
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
        _ACTIVE_PRECISION.reset(token)
