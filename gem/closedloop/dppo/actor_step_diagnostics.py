"""高学习率有限实验的真实参数、固定条件输出与优势诊断。

逐次Adam尝试在设备上比较FP32参数，按条件编码器、Transformer、qpos输出层等模块
报告有效变化比例、RMS、最大量、相对范数、梯度及PPO/BC投影。额外使用本次Adam
已经计算的动量，在FP64中重建尚未减到参数上的更新，区分零梯度和浮点舍入丢失。
此重建仅解释参数减法的精度，不冒充另一套优化器，也不修改实际梯度或权重。

固定输出探针保持音乐、观测、承诺前缀、冻结前15步状态及四次真实创新噪声不变，
重新执行四个训练步和冻结确定性末步，记录最终qpos/contact。旧链上的均值与真实
joint log-ratio另算，优势相关性是统计诊断，不把它称为真实物理性能改善。
探针和逐模块统计有明确额外成本；完整四步KL仍由原updater覆盖全部有效链。
"""
from __future__ import annotations

import math
import torch

from .gradient_merge import _group
from .execution_checks import policy_phase
from .gaussian_probability import gaussian_statistics


class StepDiagnostics:
    """所有输入快照均独占存储，不跨候选污染，也不参与反向传播。"""
    def __init__(self, actor):
        self.actor = actor
        self.before = {n: p.detach().clone() for n, p in actor.named_parameters() if p.requires_grad}
        self.components = {}

    @torch.no_grad()
    def capture_component(self, name, ppo, bc):
        self.components[name] = (None if ppo is None else ppo.clone(), None if bc is None else bc.clone())

    @torch.no_grad()
    def finish(self, optimizer, bc_weight, clip_factor):
        if len(optimizer.param_groups) != 1:
            raise ValueError('Uniform Actor AdamW requires exactly one parameter group')
        group = optimizer.param_groups[0]
        totals = {}
        for name, p in self.actor.named_parameters():
            if not p.requires_grad:
                continue
            if p.dtype != torch.float32:
                raise ValueError('Actor master parameters must remain FP32')
            old = self.before[name].double()
            delta = p.double() - old
            grad = torch.zeros_like(p) if p.grad is None else p.grad
            state = optimizer.state.get(p, {})
            ideal = torch.zeros_like(old)
            if p.grad is not None and state:
                step = float(state['step'])
                beta1, beta2 = group['betas']
                second = state['max_exp_avg_sq'] if group.get('amsgrad', False) else state['exp_avg_sq']
                ideal = -group['lr'] * state['exp_avg'].double() / (1-beta1**step)
                ideal = ideal / (second.double().sqrt() / math.sqrt(1-beta2**step) + group['eps'])
                ideal -= group['lr'] * group['weight_decay'] * old
            a, b = self.components.get(name, (None, None))
            ppo_dot = delta.new_zeros(()) if a is None else -(delta*a.double()).sum()
            bc_dot = delta.new_zeros(()) if b is None else -(delta*b.double()).sum()*bc_weight
            values = torch.stack((delta.new_tensor(p.numel()), (delta != 0).sum(), old.square().sum(),
                delta.square().sum(), delta.abs().max(), grad.double().square().sum(), (grad != 0).sum(),
                ((grad != 0) & (delta == 0)).sum(), (ideal != 0).sum(),
                ((ideal != 0) & (delta == 0)).sum(), ideal.square().sum(), ppo_dot, bc_dot))
            for key in ('all', _group(name)):
                if key not in totals:
                    totals[key] = values.clone()
                else:
                    maximum = torch.maximum(totals[key][4], values[4])
                    totals[key] += values
                    totals[key][4] = maximum
        output = {}
        for name, v in zip(totals, torch.stack(list(totals.values())).cpu().tolist()):
            n, changed, norm2, delta2, maximum, grad2, nonzero, unchanged, ideal_n, rounded, ideal2, pdot, bdot = v
            output[name] = dict(elements=int(n), changed_elements=int(changed), changed_fraction=changed/n,
                update_rms=math.sqrt(delta2/n), update_max=maximum,
                relative_update_norm=math.sqrt(delta2/max(norm2, 1e-300)), parameter_rms=math.sqrt(norm2/n),
                gradient_rms_after_clip=math.sqrt(grad2/n), gradient_norm_after_clip=math.sqrt(grad2),
                gradient_norm_before_clip=math.sqrt(grad2)/max(clip_factor, 1e-300), clip_factor=clip_factor,
                nonzero_gradient_elements=int(nonzero), nonzero_gradient_unchanged_fraction=unchanged/max(nonzero, 1),
                nonzero_adam_delta_elements=int(ideal_n), fp32_rounded_to_zero_fraction=rounded/max(ideal_n, 1),
                adam_subtraction_fp64_delta_rms=math.sqrt(ideal2/n),
                update_dot_negative_ppo_gradient=pdot if self.components else None,
                update_dot_negative_weighted_bc_gradient=bdot if self.components else None)
        return dict(modules=output, master_dtype='float32',
            rounding_scope='FP64_reconstruction_from_actual_FP32_Adam_moments_before_parameter_subtraction',
            gradient_projection_measured=bool(self.components),
            gradient_projection_is_not_separable_Adam_contribution=True)


@torch.no_grad()
def output_probe(policy, rows, cache, max_chains=16):
    """固定选择前16条真实有效链，旧路径均值及同噪声最终输出分开保存。"""
    selected = rows[:max_chains]
    if not selected:
        return None
    with policy_phase(policy):
        context = cache.context(selected)
        prepared = policy.prepare_conditions(context)
        state = cache.get('chain', selected, [15]*len(selected)).clone()
        means, logratios, stds = [], [], []
        for step in policy.trainable_steps:
            old_state = cache.get('chain', selected, [step]*len(selected))
            params = policy.transition_parameters(context, old_state, step, prepared=prepared)
            stats = gaussian_statistics(cache.get('chain', selected, [step+1]*len(selected)),
                cache.get('old_means', selected, [step]*len(selected)),
                cache.get('old_stds', selected, [step]*len(selected)),
                params['mean'], params['std'], params['free_mask'])
            means.append(params['mean'].clone())
            logratios.append(stats['joint_log_ratio'].clone())
            stds.append(params['std'].clone())
            moving = policy.transition_parameters(context, state, step, prepared=prepared)
            state = policy.actor._constrain(moving['mean'] + moving['std']*
                cache.get('sampling_noise', selected, [step]*len(selected)), prepared['adapted'])
        final = policy.transition_parameters(context, state, 19, prepared=prepared)
        qpos = policy.actor.endecoder.codec.decode_to_canonical_qpos(policy.actor.endecoder.denormalize(final['mean']))
        boundaries=None
        if hasattr(policy,'experimental_joint_limits'):
            from .joint_limits import generated_boundaries_tensors
            boundaries=generated_boundaries_tensors(qpos,cache.get('free_mask',selected)[:,:,9:],
                [policy.experimental_joint_limits]*len(selected),policy.actor.endecoder.kinematics.joint_order)
        return dict(mean=torch.stack(means, 1), joint_log_ratio=torch.stack(logratios, 1), boundaries=boundaries,
            actual_std=torch.stack(stds, 1), final=final['mean'].clone(), qpos=qpos,
            contact=final['contact_logits'].sigmoid(), mask=cache.get('free_mask', selected),
            advantages=cache.get('advantages', selected),
            sources=[r.metadata.get('training_task', {}).get('dataset', 'unknown') for r in selected],
            identities=[r.identity for r in selected])


def _correlation(x, y):
    x, y = x.double(), y.double()
    x, y = x-x.mean(), y-y.mean()
    den = x.square().sum().sqrt()*y.square().sum().sqrt()
    return None if den == 0 else float((x*y).sum()/den)


@torch.no_grad()
def compare_probes(before, after, distributed):
    local = []
    if before is not None:
        for i in range(len(before['advantages'])):
            mask = before['mask'][i]
            delta = after['final'][i].double()-before['final'][i].double()
            joint_mask = mask[:, 9:]
            qdelta = after['qpos'][i, :, 7:].double()-before['qpos'][i, :, 7:].double()
            mean_delta = after['mean'][i].double()-before['mean'][i].double()
            from .updater import _quaternion_angle
            valid=mask.any(-1)
            record=dict(identity=before['identities'][i], source=before['sources'][i],
                final_root_position_rms_m=float((after['qpos'][i,:,:3]-before['qpos'][i,:,:3]).double().square()[valid].mean().sqrt()),
                final_root_orientation_rms_rad=float(_quaternion_angle(before['qpos'][i,:,3:7],after['qpos'][i,:,3:7])[valid].square().mean().sqrt()),
                advantage=float(before['advantages'][i]),
                joint_log_ratio_gain=float((after['joint_log_ratio'][i]-before['joint_log_ratio'][i]).sum()),
                mean_change_rms=float(mean_delta[:, mask].square().mean().sqrt()),
                final_normalized_rms=float(delta[mask].square().mean().sqrt()),
                final_joint_rms_rad=float(qdelta[joint_mask].square().mean().sqrt()) if joint_mask.any() else 0.,
                final_joint_max_rad=float(qdelta[joint_mask].abs().max()) if joint_mask.any() else 0.,
                contact_rms=float((after['contact'][i]-before['contact'][i]).square().mean().sqrt()),
                protected_unchanged=bool(torch.equal(after['final'][i][~mask], before['final'][i][~mask])))
            if after.get('boundaries') is not None:
                for j,kind in enumerate(('hard','soft')):
                    record[kind+'_max_excess_before']=float(before['boundaries']['max_excess_rad'][i,j])
                    record[kind+'_max_excess_after']=float(after['boundaries']['max_excess_rad'][i,j])
                    record[kind+'_violation_fraction_before']=float(before['boundaries']['violation_fraction'][i,j])
                    record[kind+'_violation_fraction_after']=float(after['boundaries']['violation_fraction'][i,j])
            local.append(record)
    samples = [r for shard in distributed.all_gather_object(local) for r in shard]
    a = torch.tensor([r['advantage'] for r in samples], dtype=torch.float64)
    ratio = torch.tensor([r['joint_log_ratio_gain'] for r in samples], dtype=torch.float64)
    motion = torch.tensor([r['final_joint_rms_rad'] for r in samples], dtype=torch.float64)
    if samples and not all(r['protected_unchanged'] for r in samples):
        raise RuntimeError('Optimizer changed a committed reference coordinate')
    return dict(samples=samples, covered_chains=len(samples),
        advantage_joint_score_correlation=_correlation(a, ratio) if samples else None,
        advantage_motion_magnitude_correlation=_correlation(a, motion) if samples else None,
        advantage_weighted_joint_score_gain=float((a*ratio).mean()) if samples else None,
        final_joint_rms_mean=float(motion.mean()) if samples else None,
        final_joint_max_rad=max((r['final_joint_max_rad'] for r in samples), default=None),
        scope='same_music_observation_initial_noise_old_chain_means_and_regenerated_final_reference_not_physics')


def advantage_diagnostics(rows, targets, distributed):
    local = []
    for i, row in enumerate(rows):
        if bool(targets['valid'][i]):
            local.append((row.metadata.get('training_task', {}).get('dataset', 'unknown'),
                float(targets['advantages'][i]), float(targets['advantages_raw'][i]), int(row.free_mask.sum())))
    records = [r for shard in distributed.all_gather_object(local) for r in shard]
    result = {}
    for source in ['all', *sorted({r[0] for r in records})]:
        values = torch.tensor([r[1:] for r in records if source == 'all' or r[0] == source], dtype=torch.float64)
        result[source] = dict(count=len(values), normalized_mean=float(values[:, 0].mean()),
            normalized_std=float(values[:, 0].std(unbiased=False)), positive_fraction=float((values[:, 0]>0).double().mean()),
            raw_mean=float(values[:, 1].mean()), raw_std=float(values[:, 1].std(unbiased=False)),
            normalized_quantiles=torch.quantile(values[:, 0], torch.tensor([0., .05, .5, .95, 1.], dtype=torch.float64)).tolist(),
            free_coordinate_min=int(values[:, 2].min()), free_coordinate_max=int(values[:, 2].max()))
    return dict(by_source=result, normalization_scope=targets.get('advantage_normalization_scope'),
        returns_and_raw_advantages_preserved=True)
