"""复用单次全批量更新已经完成的全部可训练步KL审查，消除同状态的重复网络前向。

回溯事务完成后创建仅在本进程、本轮有效的凭据：绑定Actor参数与buffer版本、
实际执行设置、冻结先验、rollout缓存张量和全局有效链清单。凭据不进入checkpoint，
不缓存可训练特征，也不改变KL阈值。任何权重恢复、额外更新、缓存修改或覆盖变化
都使凭据失效；所有rank一致选择复用或完整重算，避免分布式通信顺序分叉。
"""
from __future__ import annotations

import json
import torch


def _tensor_signature(value):
    return (id(value), value.data_ptr(), value._version, tuple(value.shape), value.dtype, value.device)


def _signature(policy, cache, manifest):
    tensors = list(policy.actor.named_parameters()) + list(policy.actor.named_buffers())
    runtime = ('dense_linear_backend', 'dense_weight_gradient', 'dense_compute_dtype',
               'attention_compute_dtype', 'dense_precise_output', 'execution_backend', 'forward_backend')
    settings = tuple((name, module.training, tuple((key, repr(getattr(module, key)))
                     for key in runtime if hasattr(module, key)))
                     for name, module in policy.actor.named_modules())
    cached = list(cache.host.items())
    for index, block in cache.blocks.items():
        cached.extend((f'block/{index}/{name}', value) for name, value in block.items())
    return (id(cache), cache.closed, tuple((name, _tensor_signature(value)) for name, value in cached),
            tuple((name, _tensor_signature(value)) for name, value in tensors), settings,
            json.dumps(policy.kernel_config, sort_keys=True), tuple(policy.trainable_steps),
            tuple(policy.timestep_map), policy.eta, policy.guidance_scale,
            tuple(policy.std_schedule or ()),
            tuple((m['owner_rank'], m['local_index'], m['valid'], m['has_free']) for m in manifest))


def seal_accepted_kl(policy, cache, manifest, report, *, clip, reduction):
    """只封存实际覆盖全部有效链及配置中的四步或十步的已接受报告。"""
    count = sum(m['valid'] and m['has_free'] for m in manifest)
    if (report['covered_chains'] != count or report['covered_internal_transitions'] != count * len(policy.trainable_steps)
            or tuple(report['gaussian_kl_steps']) != tuple(policy.trainable_steps)):
        raise ValueError('Accepted KL does not cover the current complete rollout')
    return dict(signature=_signature(policy, cache, manifest), report=report,
                clip=clip, reduction=reduction)


def reusable_accepted_kl(result, policy, cache, manifest, distributed, *, clip, reduction):
    """凭据使用一次；任意rank失配则全体回退到完整KL，不使用局部旧结果。"""
    sealed = getattr(policy,'_accepted_kl_receipt',None)
    policy._accepted_kl_receipt=None
    valid = (sealed is not None and result.get('complete_accepted_kl_available',False)
             and not result.get('step_rejected', False)
             and sealed['clip'] == clip and sealed['reduction'] == reduction
             and sealed['signature'] == _signature(policy, cache, manifest))
    valid = all(distributed.all_gather_object(bool(valid)))
    if not valid:
        return None
    policy.assert_frozen_identity()
    report = dict(sealed['report'])
    report['reused_accepted_optimizer_step'] = True
    report['reuse_scope'] = f'same_actor_buffers_kernel_cache_manifest_complete_{len(policy.trainable_steps)}_steps'
    report['additional_final_forward_seconds'] = 0.
    return report
