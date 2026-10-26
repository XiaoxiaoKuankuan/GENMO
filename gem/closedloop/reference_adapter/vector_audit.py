"""批量真实物理方向审计：同一次仿真中并列放置基线、重复和正负扰动机器人。

仅用审计歌曲比较H1/H4/H8/H16的实际回报、接触分支、重复噪声与独立WM预测。
每个case从固定初态重放0/50步前缀，不以直接写入预测状态冒充真实起点。接触分支
变化先保持不确定，再沿固定1/5、1/25、1/125、1/625序列复测，连续两档均在相同
接触分支且通过原噪声/重复性/状态门槛才确认局部方向。任何高信噪比反向明确失败，
不能由更小扰动、loss下降或短时通过覆盖H16反例；所有原始结果完整保存。
"""

import torch

from .audit import _state_error
from .dynamics import frozen_parameters
from .vector_control import select_state, detach_state
from .vector_learning import rollout


def contact_tensors(env, count):
    import mujoco
    import warp as wp
    solver = env.sim.backend_manager._solver
    if not hasattr(solver, 'mjw_data'):
        raise RuntimeError('真实接触审计需要MuJoCo-Warp求解器标签')
    data, model = solver.mjw_data, solver.mj_model
    n = int(wp.to_torch(data.nacon).item())
    contact = data.contact
    geometries = wp.to_torch(contact.geom)[:n].long()
    worlds = wp.to_torch(contact.worldid)[:n].long()
    valid = (wp.to_torch(contact.dist)[:n] < wp.to_torch(contact.includemargin)[:n])
    valid &= (geometries >= 0).all(-1) & (worlds >= 0) & (worlds < count)
    geom, world = geometries[valid], worlds[valid]
    lo, hi = geom.amin(-1), geom.amax(-1)
    signature = torch.zeros(count, model.ngeom**2, device=env.device, dtype=torch.bool)
    signature[world, lo*model.ngeom+hi] = True
    if not hasattr(env, '_adapter_foot_geoms'):
        masks = []
        for foot in ('l_ankle_roll', 'r_ankle_roll'):
            masks.append([foot in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or '')
                          for g in range(model.ngeom)])
        env._adapter_foot_geoms = torch.tensor(masks, device=env.device)
    feet = torch.zeros(count, 2, device=env.device, dtype=torch.bool)
    for i in range(2):
        hit = env._adapter_foot_geoms[i, geom].any(-1)
        feet[world[hit], i] = True
    return signature, feet


def _physical_directions_once(collector, control, dynamics, critic, criteria, horizons):
    bank = control.bank
    audit = bank.by_split['audit']
    if len(audit) != 4 or collector.num_envs < 24:
        raise ValueError('方向审计需要4首隔离歌曲和至少24个真实环境')
    count = len(audit)*6
    primary = torch.arange(0, count, 6, device=bank.device)
    cases = []
    for start_step in (0, 50):
        for horizon in horizons:
            with torch.no_grad():
                songs = audit[0].expand(collector.num_envs).clone()
                songs[:count] = audit.repeat_interleave(6)
                frames = torch.zeros_like(songs)
                collector.reset(songs=songs, frames=frames, perturb=False)
                prefix_failed = torch.zeros(count, dtype=torch.bool, device=bank.device)
                for _ in range(start_step):
                    row = collector.step(auto_reset=False, perturb=False, split='audit')
                    prefix_failed |= row['done'][:count]
                prefix_states = collector.state['state'][:count].reshape(len(audit), 6, 55)
                prefix_spread = (prefix_states-prefix_states[:, :1]).abs().flatten(1).amax(-1)
                if start_step:
                    # 各并排世界重放后可能发生混沌漂移，不能把不同起点的差当作动作响应。
                    # 来源是刚真实执行过的主机器人当前状态和因果历史，不是想象/名义状态。
                    source = detach_state(select_state(collector.state, primary))
                    repeated = {k: v.repeat_interleave(6, dim=0).clone() for k, v in source.items()}
                    ids = torch.arange(count, device=bank.device)
                    collector.reset(ids, songs=repeated['song'], frames=repeated['frame'],
                                    initial=repeated['state'], joint_bias=repeated['xi'][:, 1:22], perturb=False)
                    fresh_episodes = collector.state['episode'][:count].clone()
                    for key, value in repeated.items():
                        collector.state[key][:count] = value
                    collector.env.xi[:count] = repeated['xi']
                    collector.env.a_prev[:count] = repeated['action']
                    collector.state['state'][:count] = collector.env.observe()['state'][:count]
                    collector.state['episode'][:count] = fresh_episodes
                    collector.state['start_frame'][:count] = repeated['frame']
                    prefix_failed = prefix_failed.reshape(len(audit), 6)[:, :1].expand(-1, 6).reshape(-1).clone()
                start = detach_state(select_state(collector.state, primary))
                initial = collector.state['state'][:count].clone()
            directions, gradients = {}, {}
            for kind, shape, epsilon in (('residual_offset', (len(audit), 10, 27), criteria['residual_epsilon']),
                                         ('action_offset', (len(audit), 21), criteria['action_epsilon'])):
                offset = start['state'].new_zeros(shape, requires_grad=True)
                with dynamics.adapter_phase(), frozen_parameters(critic):
                    imagined = rollout(control, dynamics, critic, start, horizon, terminal_value=False,
                                       intervention={kind: offset})
                    value = imagined['weighted']['sm']+imagined['weighted']['music']
                    grad, = torch.autograd.grad(value.sum(), offset)
                norm = grad.flatten(1).norm(dim=-1)
                directions[kind] = grad/norm.clamp_min(1e-12).reshape((-1,)+(1,)*(grad.ndim-1))*epsilon
                gradients[kind] = norm
            with torch.no_grad():
                r_offset = initial.new_zeros(collector.num_envs, 10, 27)
                a_offset = initial.new_zeros(collector.num_envs, 21)
                r_offset[primary+2] = directions['residual_offset']; r_offset[primary+3] = -directions['residual_offset']
                a_offset[primary+4] = directions['action_offset']; a_offset[primary+5] = -directions['action_offset']
                returns = initial.new_zeros(count)
                done = prefix_failed.clone()
                repeats, contacts, predictions, actuals = [], [], [], []
                predicted = start['state']
                first_actions = None
                for t in range(horizon):
                    row = collector.step(auto_reset=False, perturb=False, split='audit',
                        residual_offset=r_offset if t == 0 else None, action_offset=a_offset if t == 0 else None)
                    if first_actions is None:
                        first_actions = row['proposal']['action'][:count].clone()
                    returns += .99**t*(row['weighted']['sm'][:count]+row['weighted']['music'][:count])*(~done)
                    done |= row['done'][:count]
                    repeats.append(row['end']['state'][:count].reshape(len(audit), 6, 55))
                    contacts.append(contact_tensors(collector.env, count)[0].reshape(len(audit), 6, -1))
                    predicted = dynamics.step(predicted, row['proposal']['action'][primary], start['xi'])
                    predictions.append(predicted); actuals.append(row['end']['state'][primary])
                returns = returns.reshape(len(audit), 6)
                state_series = torch.stack(repeats)
                contact_series = torch.stack(contacts)
                for k, song in enumerate(audit.tolist()):
                    initial_error = float((initial[k*6:k*6+6]-initial[k*6]).abs().max())
                    repeat_error = float((state_series[:, k, 0]-state_series[:, k, 1]).abs().max())
                    repeat_contacts = bool((contact_series[:, k, 0] == contact_series[:, k, 1]).all())
                    noise = max(criteria['direction_noise_floor'], abs(float(returns[k, 0]-returns[k, 1]))*5)
                    error = _state_error(torch.stack(predictions)[:, k], torch.stack(actuals)[:, k])
                    direction_reports = []
                    for kind, plus, minus, epsilon in (('residual_offset', 2, 3, criteria['residual_epsilon']),
                                                       ('action_offset', 4, 5, criteria['action_epsilon'])):
                        delta = float(returns[k, plus]-returns[k, minus])
                        changed = not bool((contact_series[:, k, plus] == contact_series[:, k, minus]).all())
                        norm = float(gradients[kind][k])
                        finite = bool(torch.isfinite(gradients[kind][k]))
                        rms = float((first_actions[k*6+plus]-first_actions[k*6+minus]).square().mean().sqrt())
                        status = ('nonfinite_or_zero_gradient' if not finite or norm <= 1e-12 else
                                  'negative' if delta < -noise else
                                  'uncertain_contact_change' if changed else 'uncertain_below_noise' if abs(delta) <= noise else
                                  'positive' if delta > 0 else 'negative')
                        direction_reports.append(dict(kind=kind, direction_status=status, passed=status == 'positive' and rms > 0,
                            physical_return_difference=delta, physical_central_derivative=delta/(2*epsilon),
                            model_directional_derivative=norm, noise_floor=noise, signal_to_noise=abs(delta)/noise,
                            contact_branch_changed=changed, action_rms_difference=rms,
                            contact_matches_baseline=bool((contact_series[:, k, plus] == contact_series[:, k, 0]).all()
                                and (contact_series[:, k, minus] == contact_series[:, k, 0]).all()), epsilon=epsilon,
                            action_offset_pair_error=(float((first_actions[k*6+plus]-first_actions[k*6+minus]
                                -2*directions[kind][k]).abs().max()) if kind == 'action_offset' else None)))
                    repeat_ok = max(initial_error, repeat_error) <= criteria['repeatability_max_abs'] and repeat_contacts
                    passed = repeat_ok and all(error[key] <= criteria[key] for key in error)
                    passed &= all(d['passed'] for d in direction_reports) and not bool(done[k*6:k*6+6].any())
                    cases.append(dict(task=bank.tasks[song]['id'], song_id=bank.tasks[song]['song_id'],
                        start_steps=start_step, horizon=horizon, passed=bool(passed), initial_max_abs=initial_error,
                        prefix_all_variant_spread_before_pair_reset=float(prefix_spread[k]),
                        pair_initialization='true_observed_prefix_state_and_causal_history_fresh_solver' if start_step else 'original_fixed_reset',
                        repeatability_max_abs=repeat_error, repeat_contacts_equal=repeat_contacts,
                        real_terminal=bool(done[k*6:k*6+6].any()), model_errors=error, directions=direction_reports))
    passed = [h for h in horizons if all(c['passed'] for c in cases if c['horizon'] == h)]
    return dict(cases=cases, passed_horizons=passed, criteria=criteria, model_version=dynamics.model_version,
                evidence='same_simulation_real_paired_reset_prefix_no_future_execution',
                finedance189=[c for c in cases if '189' in c['task']],
                h16_allowed=16 in passed)


def resolve_contact_case(coarse, trials, criteria):
    """只补证原本为正但跨接触的局部方向；负方向和其它失败不能被小步长覆盖。"""
    import copy
    result = copy.deepcopy(coarse)
    result['coarse_passed'] = coarse['passed']
    result['local_trials'] = trials
    def physics_ok(case):
        return (not case['real_terminal'] and case['repeat_contacts_equal']
                and max(case['initial_max_abs'], case['repeatability_max_abs']) <= criteria['repeatability_max_abs']
                and all(value <= criteria[key] for key, value in case['model_errors'].items()))
    for trial in trials:
        for probe in trial['directions']:
            if physics_ok(trial) and probe['direction_status'] == 'negative':
                result['passed'] = False
                result['local_high_snr_negative'] = probe
                return result
    if coarse['passed']:
        return result
    if not physics_ok(coarse) or len(trials) < 2:
        return result
    if any(d['direction_status'] not in ('positive', 'uncertain_contact_change')
           or d['physical_return_difference'] < -d['noise_floor'] for d in coarse['directions']):
        return result
    for direction in result['directions']:
        probes = [next(d for d in t['directions'] if d['kind'] == direction['kind']) for t in trials]
        # 任何复测发现明确反向都保留失败；不挑选一次好结果覆盖反例。
        if any(physics_ok(t) and d['direction_status'] == 'negative' for t, d in zip(trials, probes)):
            return result
        if direction['passed']:
            continue
        good = [physics_ok(t) and d['passed'] and d['contact_matches_baseline'] for t, d in zip(trials, probes)]
        if not any(a and b for a, b in zip(good[:-1], good[1:])):
            return result
        direction.update(coarse_direction_status=direction['direction_status'],
                         direction_status='positive_two_local_scales', passed=True)
    result['passed'] = all(d['passed'] for d in result['directions'])
    return result


def physical_directions(collector, control, dynamics, critic, criteria, horizons=(1, 4, 8, 16)):
    coarse = _physical_directions_once(collector, control, dynamics, critic, criteria, horizons)
    pending = sorted({c['horizon'] for c in coarse['cases'] if not c['passed']
        and any(d['direction_status'] == 'uncertain_contact_change' for d in c['directions'])
        and all(d['direction_status'] in ('positive', 'uncertain_contact_change') for d in c['directions'])})
    trials = []
    for factor in (.2, .04, .008, .0016):
        if not pending:
            break
        refined = dict(criteria, residual_epsilon=criteria['residual_epsilon']*factor,
                       action_epsilon=criteria['action_epsilon']*factor)
        trials.append(_physical_directions_once(collector, control, dynamics, critic, refined, pending))
    def identity(case):
        return case['task'], case['start_steps'], case['horizon']
    by_trial = [{identity(c): c for c in trial['cases']} for trial in trials]
    cases = [resolve_contact_case(c, [t[identity(c)] for t in by_trial if identity(c) in t], criteria)
             for c in coarse['cases']]
    passed = [h for h in horizons if all(c['passed'] for c in cases if c['horizon'] == h)]
    return dict(coarse, cases=cases, passed_horizons=passed, coarse_passed_horizons=coarse['passed_horizons'],
                local_refinement_factors=[.2, .04, .008, .0016], h16_allowed=16 in passed,
                finedance189=[c for c in cases if '189' in c['task']])
