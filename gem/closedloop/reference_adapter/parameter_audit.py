"""绑定确切WM/策略/优化目标的完整参数更新物理对照，替代第一步动作方向门禁。

候选更新暂存后，基线、基线重复、候选和反向参数步分别闭环运行同一个真实因果起点。
整个十帧Adapter、冻结Actor、历史、约束和所有奖励项都参与H8/H16比较；另报告旧WM
单步局部审计但H1不能放行。审计使用独立audit歌曲，不能拿这些转移拟合WM或Critic。
记录完整参数SHA与WM更新版本；每次模型变化必须重新审计，源码SHA仅作来源记录。
实际候选收益、模型预测收益和沿真实参数步的autograd分别报告，禁止混为一种证据。
"""

import copy
import torch
from .vector_control import VectorControl, select_state, detach_state
from .vector_learning import rollout
from .dynamics import frozen_parameters
from .audit import _state_error
from .update_diagnostics import parameter_identity


class ParameterControl(VectorControl):
    def __init__(self, control, snapshots):
        self.__dict__.update(control.__dict__)
        self.snapshots = [{k.removeprefix('net.'): v for k, v in s.items() if k.startswith('net.')}
                          for s in snapshots]

    def propose(self, z, **kwargs):
        if kwargs.get('zero') or 'parameter_variant' not in z:
            return super().propose(z, **kwargs)
        x = self.packed(self.fields(z))
        latent = x.new_empty(len(x), 270)
        # 只有4个参数条件的分组前向，不是逐环境Python调用。
        for index, params in enumerate(self.snapshots):
            rows = (z['parameter_variant'] == index).nonzero().flatten()
            if len(rows):
                latent[rows] = torch.func.functional_call(self.adapter.net, params, (x[rows],))
        return super().propose(z, latent=latent, **kwargs)


def parameter_update_audit(collector, control, dynamics, critic, before, *, terminal_value=False,
                           horizons=(8, 16), prefixes=(0, 50), criteria=None):
    if terminal_value:
        raise ValueError('修复期不允许以未经导数校准的Critic终端价值做参数审计')
    bank = control.bank
    after = copy.deepcopy(control.adapter.state_dict())
    adapter_after = parameter_identity(control.adapter)
    control.adapter.load_state_dict(before)
    adapter_before = parameter_identity(control.adapter)
    reverse = {k: 2*before[k]-after[k] for k in before}
    variants = ParameterControl(control, [before, before, after, reverse])
    selected = bank.by_split['audit']
    count = len(selected)*4
    if collector.num_envs < count:
        raise ValueError('完整参数审计需要每首审计歌曲四个真实物理条件')
    identity = dict(wm_version=dynamics.model_version, wm_sha256=parameter_identity(dynamics),
                    adapter_before_sha256=adapter_before, adapter_candidate_sha256=adapter_after,
                    critic_sha256=parameter_identity(critic), terminal_value=False,
                    objective=dict(sm='original_six_plus_original_action_and_limit_costs', **control.weights,
                        root_non_saturating=control.root_cost_weight, reference_velocity=control.velocity_cost_weight),
                    constraint_beta=control.constraint_beta, disable_root_xy=control.adapter.disable_root_xy)
    cases = []
    try:
        for prefix in prefixes:
            with torch.no_grad():
                songs = selected[(torch.arange(collector.num_envs, device=bank.device)//4)%len(selected)]
                frames = torch.zeros_like(songs)
                collector.control = control
                collector.reset(songs=songs, frames=frames, perturb=False)
                for _ in range(prefix):
                    collector.step(auto_reset=False, split='audit', perturb=False)
                primary = torch.arange(0, count, 4, device=bank.device)
                source = detach_state(select_state(collector.state, primary))
                # 从真实因果历史复制到相同fresh-solver起点，不用名义或想象状态替代。
                mapping = (torch.arange(collector.num_envs, device=bank.device)//4)%len(selected)
                repeated = {k: v[mapping].clone() for k, v in source.items()}
                collector.reset(songs=repeated['song'], frames=repeated['frame'], initial=repeated['state'],
                    joint_bias=repeated['xi'][:, 1:22], perturb=False)
                collector.state = repeated
                collector.state['state'] = collector.env.observe()['state']
                collector.env.a_prev.copy_(repeated['action'])
                collector.env.xi.copy_(repeated['xi'])
                collector.state['parameter_variant'] = torch.arange(collector.num_envs, device=bank.device)%4
                start = detach_state(select_state(collector.state, slice(0, count)))
            predicted = {}
            for horizon in horizons:
                with torch.no_grad():
                    pred = rollout(variants, dynamics, critic, start, horizon, terminal_value=False)
                    predicted[horizon] = pred
            derivative = {}
            params = dict(control.adapter.named_parameters())
            for horizon in horizons:
                with dynamics.adapter_phase(), frozen_parameters(critic):
                    result = rollout(control, dynamics, critic, source, horizon, terminal_value=False)
                    gradients = torch.autograd.grad(result['objective'].sum(), tuple(params.values()))
                    derivative[horizon] = float(sum((g*(after[k]-before[k])).sum()
                        for (k, _), g in zip(params.items(), gradients)))
            with torch.no_grad():
                collector.control = variants
                scores = start['state'].new_zeros(count)
                tracks = scores.clone()
                active = torch.ones(count, device=bank.device, dtype=torch.bool)
                actual_states = []
                for t in range(max(horizons)):
                    row = collector.step(auto_reset=False, split='audit', perturb=False)
                    scores += .99**t*row['reward'][:count]*active
                    tracks += .99**t*row['raw']['track'][:count]*active
                    active &= ~row['done'][:count]
                    actual_states.append(row['end']['state'][:count].clone())
                    h = t+1
                    if h not in horizons:
                        continue
                    physical = scores.reshape(-1, 4)
                    tracking = tracks.reshape(-1, 4)
                    model = predicted[h]['objective'].reshape(-1, 4)
                    for i, song in enumerate(selected.tolist()):
                        real_delta = float(physical[i, 2]-physical[i, 0])
                        model_delta = float(model[i, 2]-model[i, 0])
                        repeat = float((physical[i, 1]-physical[i, 0]).abs())
                        noise = max(1e-7, 10*repeat)
                        resolved = abs(real_delta) > noise and abs(model_delta) > 1e-8
                        errors = _state_error(predicted[h]['states'][-1][4*i:4*i+1], actual_states[-1][4*i:4*i+1])
                        cases.append(dict(task=bank.tasks[song]['id'], prefix=prefix, horizon=h,
                            actual_full_objective_delta=real_delta, predicted_full_objective_delta=model_delta,
                            actual_tracking_delta=float(tracking[i, 2]-tracking[i, 0]),
                            reverse_full_objective_delta=float(physical[i, 3]-physical[i, 0]),
                            repeat_noise=repeat, noise_floor=noise, resolved=resolved,
                            direction_consistent=resolved and real_delta*model_delta > 0,
                            high_snr_opposite=resolved and real_delta*model_delta < 0,
                            model_state_error=errors, real_terminal=not bool(active[4*i:4*i+4].all()),
                            full_parameter_directional_derivative_all_songs=derivative[h]))
        gates = {}
        for h in horizons:
            rows = [c for c in cases if c['horizon'] == h]
            error_ok = all(all(c['model_state_error'].get(k, 0.) <= v for k, v in (criteria or {}).items()
                              if k in c['model_state_error']) for c in rows)
            actual = sum(c['actual_full_objective_delta'] for c in rows)
            noise = sum(c['noise_floor'] for c in rows)
            predicted_total = sum(c['predicted_full_objective_delta'] for c in rows)
            passed = (h >= 8 and error_ok and not any(c['high_snr_opposite'] or c['real_terminal'] for c in rows)
                and sum(c['resolved'] for c in rows) >= len(rows)//2 and actual > noise and predicted_total > 0)
            gates[str(h)] = dict(passed=passed, resolved=sum(c['resolved'] for c in rows), cases=len(rows),
                high_snr_opposites=sum(c['high_snr_opposite'] for c in rows), state_errors_within_limits=error_ok,
                actual_delta_sum=actual, predicted_delta_sum=predicted_total, noise_floor_sum=noise)
        return dict(identity=identity, cases=cases, gates=gates,
                    passed_horizons=[h for h in horizons if gates[str(h)]['passed']],
                    gate_scope='exact_current_WM_and_complete_candidate_parameter_update',
                    h1_role='local_diagnostic_only_never_training_permission')
    finally:
        control.adapter.load_state_dict(after)
        collector.control = control
        collector.state = None
        collector.reset(split='train', perturb=True)


def audit_matches(report, dynamics, adapter):
    identity = report['identity']
    return (identity['wm_version'] == dynamics.model_version and identity['wm_sha256'] == parameter_identity(dynamics)
            and identity['adapter_candidate_sha256'] == parameter_identity(adapter))
