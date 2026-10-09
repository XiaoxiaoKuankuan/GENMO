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
    if hasattr(actor.denoiser, '_stage10_eager_forward'):
        actor.denoiser.forward = actor.denoiser._stage10_eager_forward
        del actor.denoiser._stage10_eager_forward
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
            # IEEE Triton在真实大矩阵上更慢，保留已测更快的cuBLAS BMM；只有
            # T=1时间嵌入/条件需要固定归约消除batch=1时的GEMV切换。
            module.forward_backend = ('sample_bmm' if mode == 'fp32_reference' or
                mode == 'fp32_fast' and name.startswith('denoiser.') and not name.startswith('denoiser.embed_timestep.')
                else 'fixed_tile_tf32x3' if mode in ('tf32_candidate','bf16_backbone_candidate') and
                name.startswith('denoiser.') and module.out_features >= 64
                else 'sample_bmm' if name.startswith(('denoiser.final_layer.fc2','denoiser.static_conf_head.fc2'))
                else 'fixed_tile_ieee')
            module.compute_dtype = (torch.bfloat16 if mode == 'bf16_backbone_candidate'
                                    and name.startswith('denoiser.blocks.') else torch.float32)
            module.result_dtype = module.compute_dtype
        if isinstance(module, RoPEAttention):
            module.execution_backend = backend
    actor.history_encoder.execution_backend = 'native' if mode == 'fp32_reference' else 'sample_bmm_fused_gru'
    actor.history_encoder.cell.gate_accumulation_dtype = torch.float32
    actor.history_encoder.cell.forward_backend = 'sample_bmm' if mode == 'fp32_reference' else 'fixed_tile_ieee'
    configure_gradients(actor, weight_reduction='sample_bmm' if mode == 'fp32_reference' else 'joint_gemm')
    return dict(version='stage10.numerical_execution.v5', mode=mode, attention_backend=backend,
        attention_fallback=False, condition_encoding='scalar_reference' if mode == 'fp32_reference' else 'batched_unique_chains',
        history_backend=actor.history_encoder.execution_backend, master_dtype='float32',
        history_projection_accumulation='fixed_tile_ieee' if mode != 'fp32_reference' else 'native_float32',
        condition_linear_accumulation='fixed_tile_ieee' if mode != 'fp32_reference' else 'native_float32',
        linear_backend='sample_bmm' if mode == 'fp32_reference' else
            'sample_bmm_backbone_fixed_tile_conditions' if mode == 'fp32_fast' else 'fixed_tile_tensorcore_64x128x64',
        backbone_multiply='tf32x3' if mode == 'tf32_candidate' else 'bf16' if mode == 'bf16_backbone_candidate' else 'ieee',
        backbone_dtype='bfloat16' if mode == 'bf16_backbone_candidate' else 'float32',
        output_cfg_ddim_dtype='float32', probability_dtype='float64',
        matmul_tf32=False, torch_version=torch.__version__)


def compile_denoiser(policy):
    """只编译纯张量denoiser；不编译RPC、事务、校验和训练循环，无graph-break降级。"""
    if policy.numerical_execution is None:
        raise ValueError('Compilation requires an explicit numerical execution contract')
    if hasattr(policy.actor.denoiser, '_stage10_eager_forward'):
        raise ValueError('Denoiser already compiled')
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 32
    forward = policy.actor.denoiser.forward
    policy.actor.denoiser._stage10_eager_forward = forward
    policy.actor.denoiser.forward = torch.compile(forward, fullgraph=True, dynamic=False,
        backend='inductor', mode='max-autotune-no-cudagraphs')
    policy.numerical_execution.update(denoiser_compiler='inductor_fullgraph_no_cudagraphs_v1',
        compile_autotuning=True, compile_implicit_fallback=False)


def configure_blocked_fp32(policy):
    """诊断入口显式启用固定64网络行的FP32 GEMM，不改变默认参考算子。"""
    from .batch_execution import SampleMatrixLinear
    if policy.numerical_execution is None or policy.numerical_execution['mode'] != 'fp32_fast':
        raise ValueError('Blocked FP32 requires an explicit fp32_fast diagnostic contract')
    for name,module in policy.actor.denoiser.named_modules():
        if isinstance(module,SampleMatrixLinear) and not name.startswith('embed_timestep.'):
            module.forward_backend = 'blocked64_fp32_gemm'
    policy.numerical_execution.update(linear_backend='blocked64_fp32_gemm.v1',
        zero_padding_scope='internal_network_tail_only_not_effective_samples', network_row_capacity=64)


def configure_pipelined_tensorcore(policy):
    """仅诊断TF32x3固定小tile、直接累加器与L2分组候选；不静默改变v5。"""
    from .batch_execution import SampleMatrixLinear
    if policy.numerical_execution is None or policy.numerical_execution['mode'] != 'tf32_candidate':
        raise ValueError('Pipelined Tensor Core requires explicit tf32_candidate contract')
    for module in policy.actor.modules():
        if isinstance(module, SampleMatrixLinear) and module.forward_backend == 'fixed_tile_tf32x3':
            module.forward_backend = 'fixed_tile_tf32x3_pipelined'
    policy.numerical_execution.update(linear_backend='fixed_tile_tf32x3_pipeline_32x64x32.v1',
        grouped_output_tiles=8, direct_dot_accumulator=True)


def configure_compensated_bf16(policy, products):
    """显式诊断残差补偿的BF16矩阵乘；主参数、层间激活及attention保持FP32。

    3项/6项乘积是不同近似合同，必须分别实际采样并验证梯度/Adam/有效输出。
    不能把一次BF16舍入导致的策略突变用降低检查精度掩盖，也不宣称此路径与
    原FP32逐位相同。默认生产配置不会启用该候选。
    """
    from .batch_execution import SampleMatrixLinear
    from gem.network.base_arch.transformer.encoder_rope import RoPEAttention
    if policy.numerical_execution is None or policy.numerical_execution['mode']!='bf16_backbone_candidate' or products not in (3,6):
        raise ValueError('Compensated BF16 requires explicit candidate mode and 3 or 6 products')
    for name,module in policy.actor.denoiser.named_modules():
        if isinstance(module,SampleMatrixLinear):
            module.compute_dtype=module.result_dtype=torch.float32
            module.forward_backend=(f'fixed_tile_bf16x{products}_pipelined' if name.startswith('blocks.')
                else 'fixed_tile_ieee' if name.startswith('embed_timestep.') else 'blocked64_fp32_gemm')
        if isinstance(module,RoPEAttention):module.execution_backend='sdpa_math'
    policy.numerical_execution.update(linear_backend=f'compensated_bf16x{products}_fp32_output.v1',
        backbone_multiply=f'bf16x{products}',backbone_dtype='float32',attention_backend='sdpa_math',
        master_weight_precast=False,layer_output_dtype='float32',compensation_products=products)


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
