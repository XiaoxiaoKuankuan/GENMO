"""真实同起点H1至H64的三条WM预测路径，用于定位长时方向反转的来源。

每个真实因果起点复制为基线、重复、候选和反向参数条件。使用相同实际动作与名义
参考，分别比较每步输入真实状态的单步预测、真实动作驱动的连续预测，以及Adapter、
冻结Actor和WM闭环预测。逐步保存q/qd、根w、现有SM FK的body_w、原六项、累计完整
目标与接触。真实终止后的状态不计分；诊断标签只用于独立审计，不进入训练Replay。
它定位误差来源，不能替代绑定当前完整参数候选的真实信任域验收，也不放宽任何阈值。
"""

import copy
import torch
from supermimic.motion import select_reference
from supermimic.mdp import unflatten_state
from .stage2a_physics import install_repeated_start
from .vector_control import select_state, detach_state
from .reference_response import body_velocity
from .update_diagnostics import parameter_identity

TRACKING = ('body_pos', 'body_ori', 'body_linvel', 'body_angvel', 'root_pos', 'root_ori')


def sliced_proposal(proposal, count):
    return {k: (v[:count] if isinstance(v, torch.Tensor) else select_reference(v, slice(0, count)))
            for k, v in proposal.items()}


@torch.no_grad()
def trace_parameter_paths(collector, source, before, after, criteria, *, horizon=64, split='audit'):
    from .physical_trust import PairedParameters
    control = collector.control
    if split != 'audit' or not bool(torch.isin(source['song'], control.bank.by_split['audit']).all()):
        raise ValueError('三路径独立审计只读audit歌曲，不能混入WM训练')
    if not bool((source['frame']+horizon < control.bank.ends[source['song']]).all()):
        raise ValueError('H64实际名义支持不足，不能外推或补齐真实状态')
    count, mismatch = install_repeated_start(collector, source, 4)
    reverse = {k: 2*before[k]-after[k] for k in before}
    variant = PairedParameters(control, [before, before, after, reverse], count)
    start = detach_state(select_state(collector.state, slice(0, count)))
    fixed, closed = detach_state(start), detach_state(start)
    live = torch.ones(count, dtype=torch.bool, device=collector.device)
    totals = {k: torch.zeros(count, device=collector.device) for k in ('real', 'teacher_forced', 'real_actions', 'closed_loop')}
    components = {k: {} for k in totals}
    model_live = {k: live.clone() for k in ('real_actions', 'closed_loop')}
    states = {k: [] for k in totals}
    actions, contacts, rows = [], [], []
    collector.control = variant
    try:
        for t in range(horizon):
            real = collector.step(auto_reset=False, split=split, perturb=False, contact_count=count)
            true_z = select_state(real['start'], slice(0, count))
            proposal = sliced_proposal(real['proposal'], count)
            truth = real['end']['state'][:count]
            teacher = control.engine.dynamics.step(true_z['state'], proposal['action'], true_z['xi'])
            one = variant.advance(true_z, proposal, teacher)
            recurrent = control.engine.dynamics.step(fixed['state'], proposal['action'], fixed['xi'])
            fixed_step = variant.advance(fixed, proposal, recurrent)
            closed_proposal = variant.propose(closed)
            predicted = control.engine.dynamics.step(closed['state'], closed_proposal['action'], closed['xi'])
            closed_step = variant.advance(closed, closed_proposal, predicted)
            truth_raw = {k: v[:count] for k, v in real['raw'].items() if isinstance(v, torch.Tensor)}
            truth_terms = {k: v[:count] for k, v in real['weighted'].items()}
            paths = dict(real=(truth, real['reward'][:count], truth_raw, truth_terms),
                teacher_forced=(teacher, one[1], one[2], one[3]),
                real_actions=(recurrent, fixed_step[1], fixed_step[2], fixed_step[3]),
                closed_loop=(predicted, closed_step[1], closed_step[2], closed_step[3]))
            frame = dict(step=t+1, actual_alive=live.reshape(-1, 4).cpu().tolist(), paths={})
            true_bw = truth_raw['tracked_body_w']
            for name, (state, reward, raw, terms) in paths.items():
                mask = live if name not in model_live else live & model_live[name]
                totals[name] += .99**t*reward*mask
                for key, value in terms.items():
                    components[name][key] = components[name].get(key, torch.zeros_like(reward))+.99**t*value*mask
                fields = dict(q=(state[:, 13:34]-truth[:, 13:34]).square().mean(-1).sqrt(),
                    qd=(state[:, 34:]-truth[:, 34:]).square().mean(-1).sqrt(),
                    root_w=(state[:, 10:13]-truth[:, 10:13]).square().mean(-1).sqrt(),
                    body_w=(raw['tracked_body_w']-true_bw).square().mean((1, 2)).sqrt())
                frame['paths'][name] = dict(
                    errors={k: v.reshape(-1, 4).cpu().tolist() for k, v in fields.items()},
                    tracking={k: raw[k].reshape(-1, 4).cpu().tolist() for k in TRACKING},
                    cumulative_return=totals[name].reshape(-1, 4).cpu().tolist(),
                    cumulative_terms={k: v.reshape(-1, 4).cpu().tolist() for k, v in components[name].items()})
                states[name].append(state.detach().clone())
            actions.append(proposal['action'].clone()); contacts.append(real['contacts'][0])
            rows.append(frame)
            live &= ~real['done'][:count]
            model_live['real_actions'] &= fixed_step[4].bool()
            model_live['closed_loop'] &= closed_step[4].bool()
            fixed, closed = fixed_step[0], closed_step[0]
        cases = []
        for i, song in enumerate(source['song'].tolist()):
            case = dict(task=control.bank.tasks[song]['id'], start_frame=int(source['frame'][i]),
                        initial_mismatch=float(mismatch[i]), paths={})
            for name in ('teacher_forced', 'real_actions', 'closed_loop'):
                first_opposite, first_resolved_error = None, None
                for frame in rows:
                    actual = frame['paths']['real']['cumulative_return'][i]
                    predicted = frame['paths'][name]['cumulative_return'][i]
                    da, dm = actual[2]-actual[0], predicted[2]-predicted[0]
                    noise = max(criteria['direction_noise_floor'], 10*abs(actual[1]-actual[0]))
                    if abs(da)>noise and abs(dm)>1e-8 and da*dm<0 and first_opposite is None:
                        first_opposite = frame['step']
                    # 这里只标记可见的速度预测误差，不把它变成更宽/更窄的新方向门槛。
                    if max(frame['paths'][name]['errors']['qd'][i])>1e-4 and first_resolved_error is None:
                        first_resolved_error = frame['step']
                case['paths'][name] = dict(first_cumulative_high_snr_opposite=first_opposite,
                    first_qd_error_over_diagnostic_1e4=first_resolved_error,
                    H64_actual_delta=rows[-1]['paths']['real']['cumulative_return'][i][2]-rows[-1]['paths']['real']['cumulative_return'][i][0],
                    H64_predicted_delta=rows[-1]['paths'][name]['cumulative_return'][i][2]-rows[-1]['paths'][name]['cumulative_return'][i][0])
            cases.append(case)
        data = dict(states={k: torch.stack(v) for k, v in states.items()}, actions=torch.stack(actions),
            contacts=torch.stack(contacts), start=start, origin='independent_audit_never_WM_training',
            parameter_conditions=('baseline', 'repeat', 'candidate', 'reverse'))
        return dict(horizon=horizon, cases=cases, steps=rows,
            wm_version=control.engine.dynamics.model_version, wm_sha256=parameter_identity(control.engine.dynamics),
            physical_criteria=dict(criteria), objective_names=list(components['real']),
            evidence='same_real_start_same_parameter_perturbation_three_prediction_paths',
            teacher_forced_semantics='one_step_prediction_with_actual_state_history_reference_and_action',
            real_actions_semantics='recurrent_WM_with_actual_actions_and_reference_costs',
            closed_loop_semantics='recurrent_Adapter_frozen_Actor_WM_and_history',
            score_mask='actual_termination_for_all_paths_plus_predicted_termination_for_recurrent_paths'), data
    finally:
        collector.control = control


@torch.no_grad()
def independent_long_trace(collector, before, criteria, *, prefixes=(0, 50)):
    control = collector.control
    after = copy.deepcopy(control.adapter.state_dict())
    selected = control.bank.by_split['audit']
    reports, artifacts = [], []
    try:
        for prefix in prefixes:
            control.adapter.load_state_dict(before)
            songs = selected[(torch.arange(collector.num_envs, device=collector.device)//4)%len(selected)]
            collector.reset(songs=songs, frames=torch.zeros_like(songs), perturb=False)
            for _ in range(prefix):
                collector.step(zero=True, auto_reset=False, split='audit', perturb=False)
            indices = torch.arange(0, 4*len(selected), 4, device=collector.device)
            source = detach_state(select_state(collector.state, indices))
            report, data = trace_parameter_paths(collector, source, before, after, criteria)
            reports.append(report); artifacts.append(data)
        return dict(prefixes=list(prefixes), traces=reports, passed=False,
                    boundary='diagnosis_only_no_automatic_training_permission'), artifacts
    finally:
        control.adapter.load_state_dict(after)
