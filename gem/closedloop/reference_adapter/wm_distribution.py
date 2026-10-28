"""WM 真实采集分布诊断与保持原物理函数的归一化迁移。

仅操作独立世界模型，不改变 Actor、机器人状态定义、输入输出维度或预测头。
先在真实 train 转移上比较旧统计和新分布；默认不迁移，不凭 count 达到上限就重置。
确有分布漂移时，可用输入首层/输出末层的解析仿射换元更新统计，保证迁移前后物理
状态预测和输入导数等价。Adam 首末层的旧逐元素动量不是新坐标下等价动量，因此
显式重建这几层的优化器状态，其他层保持；迁移报告必须与 checkpoint 一起保存。
这里的噪声窗口评估只预测真实执行过的动作序列，不把离线误差当真实策略收益。
"""

import torch
from torch import nn


def _check_statistics(norm, mean, var, count):
    mean, var = mean.to(norm.mean), var.to(norm.var)
    if mean.shape != norm.mean.shape or var.shape != norm.var.shape:
        raise ValueError('归一化迁移维度不能改变')
    if not bool(torch.isfinite(mean).all() & torch.isfinite(var).all()) or bool((var < 0).any()):
        raise ValueError('归一化统计必须有限且方差非负')
    if float(count) <= 0:
        raise ValueError('归一化统计需要真实非零样本数')
    return mean, var


@torch.no_grad()
def rebase_input_normalizer(norm, first, new_mean, new_var, new_count, *, optimizer=None):
    if not isinstance(first, nn.Linear) or first.bias is None:
        raise ValueError('函数等价输入换元要求带 bias 的 Linear 首层')
    new_mean, new_var = _check_statistics(norm, new_mean, new_var, new_count)
    old_mean, old_std = norm.mean.clone(), norm.std().clone()
    new_std = (new_var+norm.eps**2).sqrt()
    old_weight = first.weight.clone()
    first.bias.add_(old_weight@((new_mean-old_mean)/old_std))
    first.weight.mul_((new_std/old_std)[None])
    norm.mean.copy_(new_mean); norm.var.copy_(new_var); norm.count.fill_(float(new_count))
    reset = 0
    if optimizer is not None:
        for p in (first.weight, first.bias):
            reset += int(optimizer.state.pop(p, None) is not None)
    return dict(input_rebased=True, optimizer_tensors_reinitialized=reset,
                max_input_scale_change=float((new_std/old_std).amax()), new_count=float(norm.count))


@torch.no_grad()
def rebase_wm_normalization(dynamics, inputs, targets, *, split, optimizer=None):
    """输入原物理 inputs/targets 的真实 train 样本，不接受已归一化数值或验证集。"""
    if split != 'train' or dynamics._graph_active:
        raise ValueError('仅允许在无 BPTT 计算图的真实 train 迁移边界更新独立 WM')
    model = dynamics.model
    old_counts = dict(input=float(model.in_norm.count), output=float(model.out_norm.count))
    if len(inputs) < 2 or inputs.ndim != 2 or targets.ndim != 2 or len(inputs) != len(targets):
        raise ValueError('统计迁移需要配对真实输入与 targets 样本')
    first, last = model.net[0], model.net[-1]
    if not isinstance(last, nn.Linear) or last.bias is None or first is last:
        raise ValueError('函数等价换元要求独立 Linear 首层与末层')
    xmean, xvar = inputs.mean(0), inputs.var(0, unbiased=False)
    ymean, yvar = targets.mean(0), targets.var(0, unbiased=False)
    _check_statistics(model.in_norm, xmean, xvar, len(inputs))
    _check_statistics(model.out_norm, ymean, yvar, len(targets))
    old_mean, old_std = model.out_norm.mean.clone(), model.out_norm.std().clone()
    report = rebase_input_normalizer(model.in_norm, first, xmean, xvar, len(inputs), optimizer=optimizer)
    new_std = (yvar+model.out_norm.eps**2).sqrt()
    ratio = old_std/new_std
    last.weight.mul_(ratio[:, None])
    last.bias.copy_((old_std*last.bias+old_mean-ymean)/new_std)
    model.out_norm.mean.copy_(ymean); model.out_norm.var.copy_(yvar)
    model.out_norm.count.fill_(len(targets))
    if optimizer is not None:
        for p in (last.weight, last.bias):
            report['optimizer_tensors_reinitialized'] += int(optimizer.state.pop(p, None) is not None)
    # 换元后统计固定，避免下一批 RunningNorm.update 单独改变函数；下一次迁移需再次
    # 走同一解析边界。原 count 作为报告保留，不能偷偷归零后逐批裸更新。
    model.in_norm.max_count = float(len(inputs))
    model.out_norm.max_count = float(len(targets))
    dynamics.model_version += 1
    return report | dict(output_rebased=True, target_count=len(targets), split=split, old_counts=old_counts,
        normalizer_max_count=dict(input=model.in_norm.max_count, output=model.out_norm.max_count),
        normalizer_update_policy='frozen_between_explicit_function_preserving_rebases',
        optimizer_policy='reinitialize_only_affine_reparameterized_boundary_layers')


@torch.no_grad()
def distribution_shift(norm, values):
    values = values.flatten(0, -2)
    if not len(values):
        raise ValueError('分布诊断需要真实样本')
    mean, std = values.mean(0), values.var(0, unbiased=False).sqrt()
    shift = ((mean-norm.mean)/norm.std()).abs()
    ratio = std/norm.std()
    return dict(samples=len(values), original_count=float(norm.count), max_count=float(norm.max_count),
        running_statistics_frozen=bool(norm.count >= norm.max_count),
        mean_shift_in_old_std_mean=float(shift.mean()), mean_shift_in_old_std_p95=float(shift.quantile(.95)),
        mean_shift_in_old_std_max=float(shift.max()), std_ratio_p05=float(ratio.quantile(.05)),
        std_ratio_p95=float(ratio.quantile(.95)), std_ratio_max=float(ratio.max()),
        fraction_fields_mean_shift_above_2=float((shift > 2).float().mean()))


def restore_rebased_normalization_policy(dynamics, report):
    """模型 state_dict 不含原 RunningNorm.max_count 属性，完整恢复必须显式补回。

    调用者应把 rebase_wm_normalization 的报告保存到完整 checkpoint extra_state；
    载入模型和优化器后调用本函数，防止恢复后的裸 RunningNorm 更新改变物理函数。
    """
    if report.get('normalizer_update_policy') != 'frozen_between_explicit_function_preserving_rebases':
        raise ValueError('不是保持函数不变的归一化迁移记录')
    for key, name in (('input', 'in_norm'), ('output', 'out_norm')):
        norm = getattr(dynamics.model, name)
        maximum = float(report['normalizer_max_count'][key])
        if maximum <= 0 or float(norm.count) < maximum:
            raise ValueError('恢复的归一化计数与冻结迁移记录不一致')
        norm.max_count = maximum


@torch.no_grad()
def evaluate_executed_action_windows(dynamics, windows, *, horizons=(1, 8, 16, 32), kin=None):
    """从真实首态递归使用实际动作；有效性只保留同episode连续未终止前缀。"""
    from supermimic.mdp import unflatten_state, normalize_xi
    if windows.get('origin') != 'real':
        raise ValueError('WM 分布诊断只允许真实窗口')
    s, target, action, xi, mask = (windows[k] for k in ('state', 'next_state', 'action', 'xi', 'mask'))
    if any(h > action.shape[1] for h in horizons) or mask.shape != action.shape[:2]:
        raise ValueError('没有足够真实连续标签，禁止外推')
    state = s[:, 0]
    valid = torch.ones(len(state), device=state.device, dtype=torch.bool)
    reports = {}
    names = ('root_position', 'root_orientation', 'root_linear_velocity', 'root_angular_velocity',
             'joint_position', 'joint_velocity')
    for t in range(max(horizons)):
        state = dynamics.step(state, action[:, t], xi[:, t])
        valid &= mask[:, t].bool()
        if t+1 not in horizons:
            continue
        count = int(valid.sum())
        row = dict(valid_windows=count, total_windows=len(state))
        if count:
            pred, real = unflatten_state(state[valid]), unflatten_state(target[valid, t])
            error = dynamics.model.state_error(pred, real)
            row.update({name: float(value.square().mean().sqrt()) for name, value in
                        zip(names, error.split((3, 3, 3, 3, 21, 21), -1))})
            if kin is not None:
                row['body_angular_velocity'] = float((kin.forward(*pred)[3]-kin.forward(*real)[3]).square().mean().sqrt())
        reports[str(t+1)] = row
    flat = mask.bool()
    raw_inputs = dynamics.model.inputs(unflatten_state(s[flat]), action[flat], normalize_xi(xi[flat]))
    raw_targets = dynamics.model.targets(unflatten_state(s[flat]), unflatten_state(target[flat]))
    return dict(horizons=reports, input_distribution=distribution_shift(dynamics.model.in_norm, raw_inputs),
                target_distribution=distribution_shift(dynamics.model.out_norm, raw_targets))
