#!/usr/bin/env python3
"""为已导出的联合GENMO舞蹈补充真实Newton跟踪记录，不重新生成或挑选动作。

输入是固定checkpoint导出的预览目录：manifest给出15首既定歌曲身份，generated.pt保存
30Hz生成轨迹及真正生成的9帧尾部halo。按文件SHA绑定每条轨迹，复用SM bridge的关节
映射、观测、目标、执行器和终止条件；只运行冻结Actor与Newton/MuJoCo-Warp，不调用WM。

每首歌分配独立物理world，一次显式reset后连续50Hz控制、200Hz物理推进。参考通过已有
ReferencePageCache按原插值/差分/FK读取，整个生成计划在执行前已知，无临时GT补帧。
结束槽位不复位、不继续记成成功轨迹；NaN、实际硬限位、SM终止和求解器溢出分别保留。
显式motor-clamp-continue预览模式对电机目标按机械上下限限幅，实测角度越界只记录不
终止；不裁剪测量状态，不修改生成参考，跌倒/非有限/求解器故障仍按原合同结束。
GPU批量接触按worldid归约，实际state55/action21/接触/时间按每首真正有效长度保存。
终止后的计算占位不会进入输出轨迹。既有原始参考和生成文件只读，不修改训练、模型或SM源码。

输出physics.pt可由原三栏渲染器读取；失败后画面留空，不复制终帧。这是固定生成参考的
真实执行评价，不是异步在线GENMO闭环、WM预测、硬件实测或正式128首质量统计。
"""

import argparse
from dataclasses import replace
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--numbers', type=int, nargs='+')
    parser.add_argument('--joint-limit-mode', choices=('stop', 'motor-clamp-continue'), default='stop')
    parser.add_argument('--max-control-steps', type=int, help='仅用于明确标记的有界接口检查')
    preliminary, _ = parser.parse_known_args()
    from gem.closedloop.joint_physics.config import read_config, build_adapter, seed_all
    cfg = read_config(preliminary.config)
    sys.path.insert(0, cfg['supermimic']['repo'])
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    import numpy as np
    import torch
    import mujoco
    from supermimic.env import make_sim
    from supermimic.motion import select_reference
    from gem.closedloop.joint_physics.batched_reference import ReferencePageCache
    from gem.closedloop.joint_physics.batched_simulation import BatchedStreamingPhysicalEnv
    from gem.closedloop.joint_physics.checkpoint import atomic_save
    from gem.closedloop.joint_physics.contracts import digest, asset_contract
    from gem.closedloop.joint_physics.preview_limits import clamp_motor_action

    torch.set_num_threads(4)
    seed_all(42)
    inputs, output = args.inputs.resolve(), args.output.resolve()
    manifest = json.loads((inputs/'inputs/manifest.json').read_text())
    sources = [x for x in manifest['sources'] if args.numbers is None or x['number'] in args.numbers]
    if not sources or (args.numbers is not None and set(args.numbers) != {x['number'] for x in sources}):
        raise ValueError('必须使用既定预览清单中的样本')
    if args.max_control_steps is not None and args.max_control_steps < 1:
        raise ValueError('检查步数必须大于0')
    output.mkdir(parents=True, exist_ok=False)
    adapter = build_adapter(cfg, 'cuda:0')
    assets = asset_contract(cfg, adapter)
    if assets != manifest['assets']:
        raise ValueError('SM/WM/执行器/机器人/stats/FK/时间合同与生成模型不同')
    frozen = adapter.bundle.frozen_snapshot()
    wm_calls = [0]

    def forbid_world_model(*_args, **_kwargs):
        wm_calls[0] += 1
        raise RuntimeError('真实执行入口禁止用WM代替Newton')

    adapter.bundle.step_model = forbid_world_model
    adapter.bundle.world_model.step = forbid_world_model
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'warp-lang', 'newton', 'mujoco', 'mujoco-warp', 'isaaclab', 'isaaclab-newton')}
    audit = dict(schema='genmo.joint_physics.fixed_reference_newton_preview.v1', step=manifest['step'],
                 checkpoint_sha256=manifest['checkpoint_sha256'], assets=assets, packages=versions,
                 reference_manifest_sha256=digest(inputs/'inputs/manifest.json'),
                 execution='real_newton_fixed_generated_reference_one_reset_per_source',
                 joint_limit_mode=args.joint_limit_mode,
                 max_control_steps=args.max_control_steps, num_envs=len(sources), sources=[])
    write_json(output/'manifest.json', audit)
    sim_cfg = make_sim('cuda:0', args)
    sim_cfg = replace(sim_cfg, physics=replace(sim_cfg.physics, class_type=None,
        solver_cfg=replace(sim_cfg.physics.solver_cfg, disable_sensors=True),
        deterministic_mode=cfg['physics']['determinism']))
    started = time.perf_counter()
    with launch_simulation(sim_cfg, args), torch.no_grad():
        count = len(sources)
        env = BatchedStreamingPhysicalEnv(sim_cfg, count, adapter.bundle)
        if bool((env.alpha <= 0).any()) or not torch.isfinite(env.alpha).all():
            raise ValueError('实际执行器动作缩放必须有限且严格为正')
        model = env._solver.mj_model
        limits = []
        for name in adapter.bundle.spec.joint_names:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0 or not model.jnt_limited[joint]:
                raise ValueError(f'Newton关节{name}未启用真实机械限位')
            limits.append(model.jnt_range[joint].tolist())
        compiled_limits = torch.tensor(limits, device='cuda:0')
        torch.testing.assert_close(compiled_limits[:, 0], adapter.runtime.kin.q_lo, atol=1e-6, rtol=0)
        torch.testing.assert_close(compiled_limits[:, 1], adapter.runtime.kin.q_hi, atol=1e-6, rtol=0)
        audit['newton_joint_limits'] = dict(zip(adapter.bundle.spec.joint_names, limits))
        pages = ReferencePageCache(count, 128, adapter, 'cuda:0')
        episodes, source_hashes, joint_names = [], [], []
        for slot, source in enumerate(sources):
            path = inputs/'videos'/source['directory']/'generated.pt'
            value = torch.load(path, map_location='cpu', weights_only=False)
            if value['model_step'] != manifest['step'] or value['fps'] != 30 or len(value['qpos']) != source['frames']:
                raise ValueError('生成轨迹的模型步数/帧率/有效长度改变')
            names = tuple(value['joint_names'])
            if names != tuple(adapter.bundle.spec.joint_names[i] for i in adapter.to_genmo):
                raise ValueError('生成轨迹关节顺序不符合GENMO/SM映射')
            if any(w['prefix_max_abs_error'] != 0 for w in value['windows']):
                raise ValueError('生成轨迹改变了已知前缀')
            halo = value['generated_halo']
            if halo.shape != (9, 28):
                raise ValueError('必须保留真正生成的9帧halo，不得复制尾帧')
            qpos = torch.cat((value['qpos'], halo))
            episode = dict(reference_qpos=qpos, source_versions=torch.zeros(len(qpos), dtype=torch.long),
                           old_plan_frames=len(qpos), old_plan_period_ticks=60,
                           old_plan_origin='entire_generated_reference_committed_before_execution')
            pages.assign(slot, episode)
            episodes.append(episode)
            source_hashes.append(digest(path))
            joint_names.append(names)
        initial_requested = torch.cat([pages.initial(i) for i in range(count)])
        env.reset(torch.arange(count, device='cuda:0'), initial_requested)
        initial = env.observe()
        # 四元数允许q/-q等价；根位置、关节和速度必须确实复位到同一参考初态。
        initial_error = (initial['state'][:, [*range(3), *range(7, 55)]] -
                         initial_requested[:, [*range(3), *range(7, 55)]]).abs().amax(-1)
        quaternion_dot = (initial['state'][:, 3:7] * initial_requested[:, 3:7]).sum(-1).abs()
        audit['initial_state_max_error'] = initial_error.cpu().tolist()
        if bool((initial_error > 1e-4).any()) or bool((quaternion_dot < 1-1e-5).any()):
            raise ValueError('Newton实际初态与固定参考初态不符，拒绝生成误导性执行视频')
        states = [initial['state'].clone()]
        actions, contacts, rewards, errors = [], [], [], []
        policy_actions, requested_targets, applied_targets = [], [], []
        saturation_counts = torch.zeros(count, 21, dtype=torch.long, device='cuda:0')
        actual_limit_counts = torch.zeros_like(saturation_counts)
        maximum_excess = torch.zeros(count, 21, device='cuda:0')
        ages = np.zeros(count, dtype=np.int64)
        active = np.ones(count, dtype=np.bool_)
        valid_steps = np.zeros(count, dtype=np.int64)
        reasons, failure_ticks, failed_states = [None]*count, [None]*count, {}
        history = torch.zeros(count, 10, 75, device='cuda:0')
        lengths = np.array([math.ceil(s['frames']*20/12) for s in sources])
        total_steps = 0
        while active.any():
            live = np.flatnonzero(active)
            first = int(live[0])
            mask = torch.as_tensor(active, device='cuda:0')
            current, future, goal = pages.read(ages, active, episodes)
            target = select_reference(future, (slice(None), 0))
            before = env.observe()
            # 结束world仅提供有限计算占位，不将这些占位保存成实际执行。
            safe = {k: torch.where(mask[:, None], before[k], before[k][first:first+1])
                    for k in ('state', 'previous_action', 'xi')}
            obs = adapter.runtime.observation(safe['state'], safe['previous_action'], safe['xi'], current)
            history = (obs[:, None].expand(-1, 10, -1).clone() if total_steps == 0
                       else torch.cat((history[:, 1:], obs[:, None]), 1))
            history = torch.where(mask[:, None, None], history, history[first:first+1])
            action = adapter.bundle.act(history, goal)
            finite_action = torch.isfinite(action).all(-1).cpu().numpy()
            for i in live[~finite_action[live]]:
                active[i], reasons[i], failure_ticks[i] = False, 'nonfinite_action', int(ages[i])*12
            if not active.any():
                break
            mask = torch.as_tensor(active, device='cuda:0')
            action = torch.where(mask[:, None], action, torch.zeros_like(action))
            raw_action = action
            limited, requested, applied, saturated = clamp_motor_action(
                raw_action, env, adapter.runtime.kin.q_lo, adapter.runtime.kin.q_hi)
            if args.joint_limit_mode == 'motor-clamp-continue':
                action = limited
                saturation_counts += saturated.to(torch.long)*mask[:, None]
            else:
                applied = requested
            state = env.step_physics(action)
            contact = env.contact_events_tensor()
            finite_state = torch.isfinite(state).all(-1)
            # 失败的原始state另存；解析诊断占位不得回写进真实记录。
            diagnostic_state = torch.where(finite_state[:, None], state, before['state'])
            reward, _ = adapter.runtime.reward(diagnostic_state, action, safe['previous_action'], target)
            error = (diagnostic_state[:, 13:34]-target.q).square().mean(-1)
            hard = ((state[:, 13:34] < adapter.runtime.kin.q_lo-1e-4) |
                    (state[:, 13:34] > adapter.runtime.kin.q_hi+1e-4)).any(-1)
            excess = torch.relu(adapter.runtime.kin.q_lo-state[:, 13:34])+torch.relu(state[:, 13:34]-adapter.runtime.kin.q_hi)
            finite_excess = torch.where(torch.isfinite(excess), excess, torch.zeros_like(excess))
            actual_limit_counts += (finite_excess > 1e-4).to(torch.long)*mask[:, None]
            maximum_excess = torch.maximum(maximum_excess, finite_excess*mask[:, None])
            terminated = adapter.runtime.termination(diagnostic_state, target)
            health = env.solver_health_tensor()
            flags = torch.stack((finite_state, hard, terminated, health[:, -1].bool()), -1).cpu().numpy()
            states.append(state.clone()); actions.append(action.clone()); contacts.append(contact)
            policy_actions.append(raw_action.clone()); requested_targets.append(requested.clone()); applied_targets.append(applied.clone())
            rewards.append(reward.clone()); errors.append(error.clone())
            total_steps += 1
            for i in np.flatnonzero(active):
                ages[i] += 1
                if flags[i, 0]:
                    valid_steps[i] += 1
                else:
                    failed_states[int(i)] = state[i].cpu()
                reason = ('nonfinite_measured_state' if not flags[i, 0] else
                          'solver_capacity_or_contact_failure' if flags[i, 3] else
                          'measured_hard_joint_limit' if flags[i, 1] and args.joint_limit_mode == 'stop' else
                          'sm_physical_termination' if flags[i, 2] else
                          'music_end' if ages[i] >= lengths[i] else
                          'explicit_control_budget' if args.max_control_steps is not None and ages[i] >= args.max_control_steps else None)
                if reason:
                    active[i], reasons[i] = False, reason
                    failure_ticks[i] = int(ages[i])*12
                    print(json.dumps(dict(number=sources[i]['number'], end_reason=reason,
                                          actual_control_steps=int(ages[i]), physical_seconds=int(valid_steps[i])*.02)), flush=True)
            if total_steps % 500 == 0:
                print(json.dumps(dict(control_tick=total_steps*12, active_worlds=int(active.sum()),
                                      wall_seconds=time.perf_counter()-started)), flush=True)
        health = env.solver_health_tensor().cpu()
        for slot, source in enumerate(sources):
            n = int(valid_steps[slot])
            trajectory = torch.stack([x[slot] for x in states[:n+1]]).cpu()
            action = torch.stack([x[slot] for x in actions[:n]]).cpu() if n else torch.empty(0, 21)
            contact = torch.stack([x[slot] for x in contacts[:n]]).cpu() if n else torch.empty(0, 3, dtype=torch.int32)
            qpos = torch.cat((trajectory[:, :7], trajectory[:, 13:34][:, adapter.to_genmo]), -1)
            directory = output/source['directory']
            directory.mkdir()
            atomic_save(dict(robot_name='bumi', quaternion_convention='wxyz', qpos_order='mujoco_native',
                joint_names=list(joint_names[slot]), fps=50, qpos=qpos, state55=trajectory, ticks=torch.arange(n+1)*12,
                actions=action, real_contacts=contact, contact_contract=env.contact_contract(), xi=initial['xi'][slot].cpu(),
                policy_actions=torch.stack([x[slot] for x in policy_actions[:n]]).cpu() if n else torch.empty(0, 21),
                requested_motor_targets=torch.stack([x[slot] for x in requested_targets[:n]]).cpu() if n else torch.empty(0, 21),
                applied_motor_targets=torch.stack([x[slot] for x in applied_targets[:n]]).cpu() if n else torch.empty(0, 21),
                joint_limit_mode=args.joint_limit_mode,
                max_actual_joint_excess_deg=float(maximum_excess[slot].max())*180/math.pi,
                saturated_joint_steps=int(saturation_counts[slot].sum()),
                end_reason=reasons[slot], physical_end_seconds=n*.02, evidence='measured_newton_state',
                failure_tick=failure_ticks[slot], failed_state=failed_states.get(slot), reset_count=1,
                generated_reference_sha256=source_hashes[slot], world_model_forward_calls=wm_calls[0]), directory/'physics.pt')
            result = dict(source, generated_reference_sha256=source_hashes[slot], physical_seconds=n*.02,
                joint_limit_mode=args.joint_limit_mode,
                saturated_joint_steps=int(saturation_counts[slot].sum()),
                actual_violating_joint_steps=int(actual_limit_counts[slot].sum()),
                max_actual_joint_excess_deg=float(maximum_excess[slot].max())*180/math.pi,
                actual_joint_limits={name: {'violating_steps':int(actual_limit_counts[slot,j]),
                    'max_excess_deg':float(maximum_excess[slot,j])*180/math.pi}
                    for j,name in enumerate(adapter.bundle.spec.joint_names)},
                actual_control_steps=int(ages[slot]), recorded_finite_control_steps=n,
                complete_song=reasons[slot]=='music_end', end_reason=reasons[slot], failure_tick=failure_ticks[slot],
                tracking_reward_mean=float(torch.stack([x[slot] for x in rewards[:n]]).mean()) if n else None,
                joint_tracking_rmse_rad=float(torch.stack([x[slot] for x in errors[:n]]).mean().sqrt()) if n else None,
                solver_health=health[slot].tolist(), reset_count=1, physics_sha256=digest(directory/'physics.pt'))
            write_json(directory/'result.json', result)
            audit['sources'].append(result)
        audit['reference_cache'] = pages.statistics()
    adapter.bundle.assert_unchanged(frozen)
    audit.update(frozen_unchanged=True, world_model_forward_calls=wm_calls[0], wall_seconds=time.perf_counter()-started)
    write_json(output/'manifest.json', audit)
    print(json.dumps(dict(completed_sources=len(sources), complete_songs=sum(s['complete_song'] for s in audit['sources']),
                          wall_seconds=audit['wall_seconds'], frozen_unchanged=True, wm_forward_calls=wm_calls[0])), flush=True)


if __name__ == '__main__':
    main()
