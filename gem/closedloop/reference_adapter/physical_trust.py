"""WM-BPTT候选Adam参数步的真实物理信任域，不使用PPO或评价集选择更新。

对训练内部预留的真实因果起点同时执行旧策略、旧策略重复、完整候选和反向参数步。
绑定WM版本及参数SHA、旧/候选Adapter身份和完整精度目标；H1只诊断，H8以上才可能
验收。按1、0.5、0.25、0.125缩放同一Adam参数增量，失败完全回滚参数和优化器。
实际Adam步仅在真实收益超过重复噪声且无可信反向时计数一次。失败候选真实数据可
用于训练WM；最终evaluation歌曲禁止进入本模块。未配对的其余环境继续真实采集，
不让2048环境为了少数审计起点空跑，也不将它们伪记成配对审计转移。
"""

import copy
import torch
from .parameter_audit import ParameterControl, summarize_horizon
from .stage2a_physics import install_repeated_start
from .vector_control import select_state, detach_state
from .vector_learning import rollout
from .audit import _state_error
from .update_diagnostics import parameter_identity


class PairedParameters(ParameterControl):
    def __init__(self, control, snapshots, count):
        super().__init__(control, snapshots)
        self.paired_count = count

    def propose(self, z, **kwargs):
        ids = torch.arange(len(z['state']), device=z['state'].device)
        variant = torch.where(ids < self.paired_count, ids % 4, 0)
        return super().propose(dict(z, parameter_variant=variant), **kwargs)


@torch.no_grad()
def evaluate_candidate(collector, source, dynamics, critic, before, after, criteria, *, horizon=8, replay=None, response_replay=None):
    control = collector.control
    if horizon < 8:
        raise ValueError('H1不能作为Adapter参数更新许可')
    train = control.bank.by_split['train']
    if not bool(torch.isin(source['song'], train).all()):
        raise ValueError('信任域只允许训练内部预留起点，禁止使用evaluation/audit/validation歌曲选参数')
    control.adapter.load_state_dict(before)
    count, mismatch = install_repeated_start(collector, source, 4)
    reverse = {k: 2*before[k]-after[k] for k in before}
    variant = PairedParameters(control, [before, before, after, reverse], count)
    start = detach_state(select_state(collector.state, slice(0, count)))
    prediction = rollout(variant, dynamics, critic, start, horizon, terminal_value=False)
    # 与“模型预测候选差分”分开记录完整闭环autograd沿本次参数步的方向导数。
    with torch.enable_grad(), dynamics.adapter_phase():
        differentiable = rollout(control, dynamics, critic, detach_state(source), horizon, terminal_value=False)
        named = tuple(control.adapter.named_parameters())
        gradient = torch.autograd.grad(differentiable['objective'].sum(), tuple(p for _, p in named))
        directional = float(sum((g*(after[k]-before[k])).sum() for (k, _), g in zip(named, gradient)))
    total = start['state'].new_zeros(count)
    tracking = total.clone()
    alive = torch.ones(count, device=collector.device, dtype=torch.bool)
    repeat_error = torch.zeros(len(source['state']), device=collector.device)
    peak_excess = torch.zeros_like(repeat_error)
    physical_states = []
    physical_actions, reliable, switches = [], [], []
    previous_contact = None
    collector.control = variant
    try:
        for t in range(horizon):
            row = collector.step(replay, auto_reset=True, split='train', perturb=True, contact_count=count)
            total += .99**t*row['reward'][:count]*alive
            tracking += .99**t*row['raw']['track'][:count]*alive
            alive &= ~row['done'][:count]
            physical_states.append(row['end']['state'][:count].clone())
            states = physical_states[-1].reshape(-1, 4, 55)
            repeat_error = torch.maximum(repeat_error, (states[:, 1]-states[:, 0]).abs().amax(-1))
            actions = row['proposal']['action'][:count].reshape(-1, 4, 21)
            physical_actions.append(actions.clone())
            contact = row['contacts'][0].reshape(len(source['state']), 4, -1)
            changed = (contact[:, 2] != contact[:, 3]).any(-1)
            if previous_contact is not None:
                changed |= (contact != previous_contact).any((1, 2))
            action_size = (actions[:, 2]-actions[:, 3]).square().mean(-1).sqrt()
            good = ((mismatch <= 1e-5) & (repeat_error <= 1e-4) & ~changed & (action_size >= 1e-4)
                    & alive.reshape(-1, 4).all(-1))
            if reliable:
                good &= reliable[-1]
            reliable.append(good); switches.append(changed); previous_contact = contact
        scores, tracks, predictions = (x.reshape(-1, 4) for x in (total, tracking, prediction['objective']))
        cases = []
        for i, song in enumerate(source['song'].tolist()):
            actual, predicted = float(scores[i, 2]-scores[i, 0]), float(predictions[i, 2]-predictions[i, 0])
            cases.append(dict(task=control.bank.tasks[song]['id'], prefix=int(source['frame'][i]), horizon=horizon,
                actual_full_objective_delta=actual, predicted_full_objective_delta=predicted,
                actual_tracking_delta=float(tracks[i, 2]-tracks[i, 0]),
                reverse_full_objective_delta=float(scores[i, 3]-scores[i, 0]),
                repeat_noise=abs(float(scores[i, 1]-scores[i, 0])), repeat_state_max_abs=float(repeat_error[i]),
                initial_mismatch=float(mismatch[i]), real_terminal=not bool(alive[i*4:i*4+4].all()),
                model_state_error=_state_error(prediction['states'][-1][i*4:(i+1)*4], physical_states[-1][i*4:(i+1)*4])))
        gate = summarize_horizon(cases, horizon, criteria)
        gate['initial_states_match'] = bool((mismatch <= 1e-5).all())
        gate['passed'] &= gate['initial_states_match']
        if response_replay is not None:
            actual = torch.stack(physical_states, 1).reshape(len(source['state']), 4, horizon, 55).transpose(1, 2)
            actions = torch.stack(physical_actions, 1)
            valid, switch = torch.stack(reliable, 1), torch.stack(switches, 1)
            response_replay.add(dict(origin='current_policy_real_reference_response', split='train',
                state=start['state'][::4], xi=start['xi'][::4], song=source['song'], frame=source['frame'],
                plus_action=actions[:, :, 2], minus_action=actions[:, :, 3],
                plus_next=actual[:, :, 2], minus_next=actual[:, :, 3], reliable=valid, contact_switch=switch,
                policy_version=torch.full_like(source['song'], collector.policy_version),
                initial_mismatch=mismatch, repeated_states=actual[:, :, :2],
                rejected=dict(unreliable_steps=int((~valid).sum()), contact_switch_steps=int(switch.sum()),
                              initial_mismatch=int((mismatch > 1e-5).sum()))))
        return dict(identity=dict(wm_version=dynamics.model_version, wm_sha256=parameter_identity(dynamics),
            adapter_before_sha256=parameter_identity(control.adapter), objective_scales=control.precision.scales,
            precision_weight=control.precision.weight, terminal_value=False,
            soft_joint_limit_cost=control.soft_joint_limit_cost, reference_soft_limit_weight=control.weights['reference_limit']), cases=cases, gate=gate,
            paired_transitions=count*horizon, concurrent_training_transitions=(collector.num_envs-count)*horizon,
            selection_split='train_safety_real_start_bank', horizon=horizon,
            WM_closed_loop_autograd_parameter_direction=directional,
            derivative_scope='complete_H8_objective_all_270_parameters_history_and_projection')
    finally:
        collector.control = control
        control.adapter.load_state_dict(after)


def trust_region_update(collector, source, dynamics, critic, optimizer, before, adam_before,
                        criteria, *, horizon=8, replay=None, response_replay=None):
    control = collector.control
    candidate = copy.deepcopy(control.adapter.state_dict())
    adam_candidate = copy.deepcopy(optimizer.state_dict())
    reports = []
    accepted = False
    try:
        for fraction in (1., .5, .25, .125):
            scaled = {k: before[k]+fraction*(candidate[k]-before[k]) for k in before}
            control.adapter.load_state_dict(scaled)
            report = evaluate_candidate(collector, source, dynamics, critic, before, scaled,
                criteria, horizon=horizon, replay=replay, response_replay=response_replay)
            report['parameter_step_fraction'] = fraction
            report['identity']['adapter_candidate_sha256'] = parameter_identity(control.adapter)
            reports.append(report)
            if report['gate']['passed']:
                optimizer.load_state_dict(adam_candidate)
                accepted = True
                break
        return dict(accepted=accepted, attempts=reports, optimizer_steps=int(accepted),
                    adam_semantics='one_candidate_gradient_moment_update_with_accepted_parameter_step_scaling')
    finally:
        if not accepted:
            control.adapter.load_state_dict(before)
            optimizer.load_state_dict(adam_before)
