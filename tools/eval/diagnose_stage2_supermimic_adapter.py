"""独立重放已保存Adapter，导出真实物理视频轨迹和同初态残差消融证据。

本工具只读加载正式checkpoint中的冻结Actor、Adapter及诊断需要的WM/Critic；不恢复
训练采集器，不创建优化器，不修改任何正式模型、Replay或日志。collect模式复用原
VectorControl/VectorCollector及RobotSpec，在一个独立GPU仿真内同时运行零Adapter、
两个指定模型和根/关节/幅值消融。每组使用相同四首独立评价歌、初态、50Hz控制及
200Hz物理，只保存首个真实episode，失败后的状态不补分、不补姿态。渲染另由已有
render_stage10_model_comparison.py读取真实qpos完成，不把运动学回放冒充新动力学。

mechanism模式从真实重放保存的因果起点检验输入可观测性、位置奖励梯度、输出饱和、
局部响应训练动作覆盖和Critic终端梯度来源。诊断不使用未来真实状态，不更新参数，
每项结果保留模型身份、任务和时域；有限步消融不自动证明整个训练过程的唯一原因。

evaluation模式按各checkpoint自身参考约束恢复部署合同，依次评价零Adapter和两个
冻结模型，复用正式整歌评价的求解器接触脚滑、六项Tracking、参考导数和相关矩阵。
不创建优化器、不拟合Critic；逐模型重置到同样的名义初态，顺序求解器误差与同批次
collect配对结果分开报告，不把顺序差异伪称训练增益。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_adapter(path, engine):
    import torch
    # mmap只触碰本次实际读取的张量页，避免把百万Replay搬到GPU。
    ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if ck['input_spec'] != engine.step.adapter.input_spec:
        raise ValueError('checkpoint与2654维具名输入合同不符')
    if any(not torch.equal(v.cpu(), ck['frozen_bundle'][k])
           for k, v in engine.step.bundle.state_dict().items()):
        raise ValueError('冻结Actor/normalizer/资产与本次重放不符')
    model = copy.deepcopy(engine.step.adapter)
    model.load_state_dict(ck['adapter'])
    # 新修复checkpoint的部署合同由其配置恢复，不能只加载权重后丢失XY屏蔽/参考约束。
    repair = ck.get('config', {}).get('repair', {})
    model.disable_root_xy = bool(repair.get('root_xy_disabled', False))
    model.reference_constraint_beta = (float(repair.get('constraint_beta', 1.))
        if repair.get('variant') in ('B', 'C') else 1.)
    return model, ck


def variants():
    rows = [('zero', -1, 'all', 0.)]
    for model in range(2):
        for component, scale in [('all', 1.), ('root', 1.), ('joints', 1.), ('all', .1),
                                 ('translation', 1.), ('rotation', 1.), ('no_root_xy', 1.),
                                 ('no_root_xy_constrained', 1.)]:
            rows.append((f'model{model}_{component}_{scale:g}', model, component, scale))
    return rows


def evaluate(args, cfg, engine, bank):
    import torch
    from isaaclab.app import launch_simulation
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from gem.closedloop.reference_adapter.vector_control import VectorControl
    from gem.closedloop.reference_adapter.vector_collector import VectorCollector
    from gem.closedloop.reference_adapter.vector_evaluation import evaluate_physical
    from gem.closedloop.reference_adapter.vector_entry import compile_kinematics
    from gem.closedloop.reference_adapter.update_diagnostics import parameter_identity
    if not args.no_compile:
        compile_kinematics(engine, bank)
    frozen = engine.step.bundle.frozen_snapshot()
    models = [checkpoint_adapter(p, engine) for p in args.checkpoints]
    control = VectorControl(engine, bank)
    output = Path(args.output)
    reports = {}
    sim_cfg = make_sim('cuda:0', args)
    with launch_simulation(sim_cfg, args), torch.no_grad():
        env = StreamingPhysicalEnv(sim_cfg, args.num_envs, engine.step.bundle)
        collector = VectorCollector(env, control, seed=20261027)
        for index in (-1, 0, 1):
            name = 'zero' if index < 0 else f'model{index}'
            # 零基线使用相同新增成本与约束配置；其残差全零，部署动作保持零Adapter。
            model, ck = models[max(index, 0)]
            repair = ck.get('config', {}).get('repair', {})
            control.adapter.load_state_dict(model.state_dict())
            control.adapter.disable_root_xy = model.disable_root_xy
            control.constraint_beta = model.reference_constraint_beta
            control.root_cost_weight = float(repair.get('root_position_cost', 0.))
            control.velocity_cost_weight = float(repair.get('reference_velocity_cost', 0.))
            control.relative_critic_features = bool(repair.get('critic_feature_schema'))
            control.soft_imagination_continuation = bool(repair.get('imagination_soft_continuation', True))
            identity = parameter_identity(control.adapter)
            result = evaluate_physical(collector, zero=index < 0)
            assert parameter_identity(control.adapter) == identity
            engine.step.bundle.assert_unchanged(frozen)
            result.update(checkpoint=None if index < 0 else args.checkpoints[index],
                          accepted_updates=0 if index < 0 else ck['state']['accepted_updates'],
                          adapter_sha256=identity, optimizer_updates=0,
                          comparison='sequential_fixed_initial_state_complement_to_batched_paired_evaluation')
            write_json(output/f'evaluation_{name}.json', result)
            reports[name] = {k: result[k] for k in ('completion_rate', 'mean_tracking',
                'mean_root_position_rmse_m', 'mean_jerk_rms', 'mean_beat_f1')}
        write_json(output/'result.json', dict(models=reports, actual_transitions=collector.real_transitions,
            optimizer_updates=0, frozen_actor_unchanged=True, num_envs=args.num_envs,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20))


def collect(args, cfg, engine, tasks, bank):
    import numpy as np
    import torch
    from isaaclab.app import launch_simulation
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from supermimic.rotations import mat_to_quat
    from gem.closedloop.reference_adapter.vector_control import VectorControl, select_state
    from gem.closedloop.reference_adapter.vector_collector import VectorCollector
    from gem.closedloop.reference_adapter.evaluation import beat_report
    from gem.closedloop.reference_adapter.vector_entry import compile_kinematics
    from gem.closedloop.reference_adapter.reference_constraints import constrain_plan, quantiles
    if not args.no_compile:
        compile_kinematics(engine, bank)
    models, identities = [], []
    for path in args.checkpoints:
        model, ck = checkpoint_adapter(path, engine)
        models.append(model)
        identities.append(dict(path=str(Path(path).resolve()), accepted_updates=ck['state']['accepted_updates'],
                               source=ck['state']['source'], file_bytes=Path(path).stat().st_size))
        del ck
    rows = variants()
    ns = len(bank.by_split['evaluation']); count = ns*len(rows)
    if args.num_envs < count:
        raise ValueError(f'至少需要{count}个环境才能同初态并行消融')
    indices = torch.arange(args.num_envs, device=bank.device)
    variant = indices.div(ns, rounding_mode='floor') % len(rows)
    song_ids = bank.by_split['evaluation'][indices % ns]
    mask = torch.ones(len(rows), 27, device=bank.device)
    for i, (_, _, component, _) in enumerate(rows):
        if component == 'root': mask[i, 6:] = 0
        if component == 'joints': mask[i, :6] = 0
        if component == 'translation': mask[i, 3:] = 0
        if component == 'rotation': mask[i, :3] = 0; mask[i, 6:] = 0
        if component in ('no_root_xy', 'no_root_xy_constrained'): mask[i, :2] = 0
    forced_constraint = torch.tensor([r[2] == 'no_root_xy_constrained' for r in rows], device=bank.device)
    scale = torch.tensor([r[3] for r in rows], device=bank.device)
    model_id = torch.tensor([r[1] for r in rows], device=bank.device)[variant]
    subsets = [(model_id == i).nonzero().flatten() for i in range(2)]

    class PairedControl(VectorControl):
        def propose(self, z, **unused):
            residual = z['state'].new_zeros(len(z['state']), 10, 27)
            for model, selected in zip(models, subsets):
                zz = select_state(z, selected)
                residual[selected] = model.forward_packed(self.packed(self.fields(zz)))
            residual = residual*mask[variant, None]*scale[variant, None, None]
            raw_residual = residual.clone()
            for model, selected in zip(models, subsets):
                beta = model.reference_constraint_beta
                filtered = selected if beta < 1 else selected[forced_constraint[variant[selected]]]
                if len(filtered):
                    residual[filtered] = constrain_plan(residual[filtered], z['left'][filtered], z['anchor'][filtered],
                                                         beta if beta < 1 else .05)
            current, future, goal, halo = self.reference(z, residual)
            return dict(residual=residual, raw_residual=raw_residual, current=current, future=future, goal=goal, halo=halo,
                        action=self.bundle.act(z['history'], goal))

    control = PairedControl(engine, bank)
    control.detailed_reference = True
    output = Path(args.output)
    started = time.time()
    sim_cfg = make_sim('cuda:0', args)
    with launch_simulation(sim_cfg, args), torch.no_grad():
        env = StreamingPhysicalEnv(sim_cfg, args.num_envs, engine.step.bundle)
        collector = VectorCollector(env, control, seed=20261026)
        collector.reset(songs=song_ids, frames=torch.zeros_like(song_ids), perturb=False)
        initial = collector.state['state'][:count].clone()
        paired_error = float((initial.reshape(len(rows), ns, 55)-initial[:ns]).abs().max())
        if paired_error > 1e-4:
            raise ValueError(f'真实初态配对不符: {paired_error}')
        active = torch.ones(count, dtype=torch.bool, device=bank.device)
        failed = torch.zeros_like(active)
        series = {k: [] for k in ('state', 'action', 'residual', 'nominal', 'corrected', 'target_qd', 'reward', 'live')}
        raw_series, snapshots, reference_series = {}, {}, {}
        step = 0
        while bool(active.any()):
            if step in (0, 50, 100, 200):
                snapshots[step] = {k: v[:count].detach().cpu().clone() for k, v in collector.state.items()}
            row = collector.step(auto_reset=True, split='evaluation', perturb=False)
            nominal = bank.ref(row['start']['song'][:count], row['start']['frame'][:count]+1)
            target = row['proposal']['future']
            nominal_future = bank.ref(row['start']['song'][:count, None],
                row['start']['frame'][:count, None]+bank.offset11[None, 1:])
            row['raw']['planned_root_velocity_error'] = (target.root_v[:count]-nominal_future.root_v).square().mean((1, 2)).sqrt()
            row['raw']['planned_root_angular_velocity_error'] = (target.root_w[:count]-nominal_future.root_w).square().mean((1, 2)).sqrt()
            row['raw']['planned_joint_velocity_error'] = (target.qd[:count]-nominal_future.qd).square().mean((1, 2)).sqrt()
            nominal_pose = torch.cat((nominal.root_p, mat_to_quat(nominal.root_R), nominal.q), -1)
            corrected_pose = torch.cat((target.root_p[:count, 0], mat_to_quat(target.root_R[:count, 0]), target.q[:count, 0]), -1)
            values = dict(state=row['end']['state'][:count], action=row['proposal']['action'][:count],
                residual=row['proposal']['residual'][:count], nominal=nominal_pose, corrected=corrected_pose,
                target_qd=target.qd[:count], reward=row['reward'][:count], live=active)
            for k, v in values.items(): series[k].append(v.detach().cpu().clone())
            for k, v in row['raw'].items():
                if isinstance(v, torch.Tensor) and v.ndim == 1:
                    raw_series.setdefault(k, []).append(v[:count].detach().cpu().clone())
                elif k.startswith('reference_') and isinstance(v, torch.Tensor) and v.ndim > 2:
                    reference_series.setdefault(k, []).append(v[:count].detach().cpu().clone())
            failed |= active & row['failed'][:count]
            active &= ~row['done'][:count]
            step += 1
            if step % 200 == 0:
                print(json.dumps(dict(vector_steps=step, active=int(active.sum()), wall_s=time.time()-started)), flush=True)
        series = {k: torch.stack(v).numpy() for k, v in series.items()}
        raw_series = {k: torch.stack(v).numpy() for k, v in raw_series.items()}
        torch.save(snapshots, output/'causal_starts.pt')
        # Isaac内部关节序不等于MJCF qpos序；渲染必须按MJCF关节名称重新映射。
        import mujoco
        mjcf = engine.step.bundle.spec.mjcf_path
        render_model = mujoco.MjModel.from_xml_path(str(mjcf))
        names = [mujoco.mj_id2name(render_model, mujoco.mjtObj.mjOBJ_JOINT, i)
                 for i in range(render_model.njnt) if render_model.jnt_qposadr[i] >= 7]
        native = np.array([engine.step.bundle.spec.joint_names.index(name) for name in names])
        if render_model.nq != 28 or len(set(names)) != 21:
            raise ValueError('渲染MJCF必须包含21个具名关节和自由根')
        lo, hi = control.runtime.kin.q_lo.cpu().numpy()[native], control.runtime.kin.q_hi.cpu().numpy()[native]
        soft_lo, soft_hi = control.runtime.mdp.soft_lo.cpu().numpy()[native], control.runtime.mdp.soft_hi.cpu().numpy()[native]
        write_json(output/'render_assets.json', dict(source_mjcf_sha256=sha(mjcf), mjcf=str(mjcf), joint_names=names))
        reports = []
        for column in range(count):
            variant_id, song_column = divmod(column, ns)
            name, model_idx, component, scale_value = rows[variant_id]
            song = int(song_ids[column]); task = tasks[song]
            n = int(series['live'][:, column].sum())
            state = series['state'][:n, column]
            qpos = np.concatenate((state[:, :7], state[:, 13:34][:, native]), -1)
            nominal = series['nominal'][:n, column]
            nominal = np.concatenate((nominal[:, :7], nominal[:, 7:][:, native]), -1)
            corrected = series['corrected'][:n, column]
            corrected = np.concatenate((corrected[:, :7], corrected[:, 7:][:, native]), -1)
            raw = {k: v[:n, column] for k, v in raw_series.items()}
            distributions = {}
            for key, values in reference_series.items():
                sample = torch.stack([v[column] for v in values[:n]])
                if key in ('reference_root_v', 'reference_root_a', 'reference_root_w', 'reference_root_alpha',
                           'reference_goal_root_v', 'reference_goal_root_a', 'reference_goal_root_w', 'reference_goal_root_alpha'):
                    sample = sample.norm(dim=-1)
                distributions[key] = quantiles(sample)
            initial_qd = initial[column:column+1, 34:].cpu()
            qd = torch.cat((initial_qd, torch.from_numpy(state[:, 34:])))
            audio = bank.audio[song]
            beat = beat_report(qd.to(bank.device), torch.arange(n+1, device=bank.device)*12, audio)
            stem = f'{name}_song{song_column}'
            array_file = output/(stem+'.npz')
            np.savez_compressed(array_file, qpos=qpos, nominal_qpos=nominal, corrected_qpos=corrected,
                state=state, action=series['action'][:n, column], residual=series['residual'][:n, column],
                target_qd=series['target_qd'][:n, column], tick=np.arange(1, n+1)*12,
                actual_joint_gmt=qpos[:, 7:], safe_limits=np.stack((soft_lo, soft_hi), -1)[None],
                hard_limits=np.stack((lo, hi), -1)[None], reward=series['reward'][:n, column],
                activity=np.abs(state[:, 34:]).mean(-1), target_activity=raw['activity_nominal'],
                activity_gate=np.ones(n), **{'raw_'+k: v for k, v in raw.items()})
            render_task = dict(split=task['split'], dataset=task['source'], sample_id=task['id'], seed=42,
                sample=dict(row=dict(source_audio_sha256=task['audio_sha256'])))
            metadata = dict(schema='genmo.adapter.actual_physics_diagnostic.v1', task_id=task['id'], task=render_task,
                trajectory_file=array_file.name, sha256=sha(array_file), fps=50, quaternion_convention='wxyz',
                qpos_order='mujoco_native', joint_names=names, actual_control_steps=n,
                end_reason='physical_failure' if bool(failed[column]) else 'music_end',
                checkpoint=identities[model_idx] if model_idx >= 0 else None,
                variant=name, component=component, residual_scale=scale_value, audio_path=task['audio_path'],
                physics='original_SM_IsaacLab3_Newton_MuJoCoWarp_200Hz_50Hz_control',
                num_envs=args.num_envs, initial_pair_max_abs=paired_error)
            write_json(output/(stem+'.json'), metadata)
            reports.append(dict(variant=name, task=task['id'], steps=n, completion=not bool(failed[column]),
                reference_distributions=distributions,
                reference_derivative_scheme='plain=forward_segments_including_seam_and_halo; goal_=actual_Actor_central_difference_targets',
                failure_time_s=n*.02 if bool(failed[column]) else None, mean_tracking=float(raw['track'].mean()),
                root_position_rmse_m=float(np.sqrt(raw['error_root_pos'].mean())),
                root_orientation_rmse_rad=float(np.sqrt(raw['error_root_ori'].mean())),
                root_final_error_m=float(np.linalg.norm(qpos[-1, :3]-nominal[-1, :3])),
                jerk_rms=float((qd.diff(n=2, dim=0)/.02**2).square().mean().sqrt()),
                mean_abs_joint_speed=float(qd.abs().mean()), music=beat,
                raw_means={k: float(v.mean()) for k, v in raw.items()}, trajectory=str(array_file)))
        summaries = []
        for name, _, _, _ in rows:
            subset = [r for r in reports if r['variant'] == name]
            summaries.append(dict(variant=name, completion_rate=sum(r['completion'] for r in subset)/ns,
                **{k: sum(r[k] for r in subset)/ns for k in ('mean_tracking', 'root_position_rmse_m',
                       'root_orientation_rmse_rad', 'jerk_rms', 'mean_abs_joint_speed')},
                mean_beat_f1=sum(r['music']['f1'] or 0. for r in subset)/ns))
        write_json(output/'physical_ablation.json', dict(models=identities, variants=rows, episodes=reports,
            summaries=summaries, source='actual_physics_same_initial_state_no_optimizer_updates',
            actual_transitions=collector.real_transitions, vector_steps=step, wall_seconds=time.time()-started,
            initial_pair_max_abs=paired_error, torch_peak_mib=torch.cuda.max_memory_allocated()/2**20))
        print(json.dumps(summaries, ensure_ascii=False), flush=True)


def mechanism(args, engine, bank):
    import torch
    from gem.closedloop.reference_adapter.vector_control import VectorControl, select_state
    from gem.closedloop.reference_adapter.vector_learning import GroupNormalizedCritic, rollout
    from gem.closedloop.reference_adapter.dynamics import frozen_parameters
    snapshots = (torch.load(Path(args.trajectories)/'causal_starts.pt', map_location='cpu', weights_only=False)
                 if args.starts_source == 'trajectory' else None)
    control = VectorControl(engine, bank)
    ns = len(bank.by_split['evaluation'])
    output = Path(args.output)
    reports = []
    for model_index, path in enumerate(args.checkpoints):
        model, ck = checkpoint_adapter(path, engine)
        engine.step.adapter.load_state_dict(model.state_dict())
        engine.step.adapter.disable_root_xy = model.disable_root_xy
        control.constraint_beta = model.reference_constraint_beta
        repair = ck.get('config', {}).get('repair', {})
        control.root_cost_weight = float(repair.get('root_position_cost', 0.))
        control.velocity_cost_weight = float(repair.get('reference_velocity_cost', 0.))
        control.relative_critic_features = bool(repair.get('critic_feature_schema'))
        control.soft_imagination_continuation = bool(repair.get('imagination_soft_continuation', True))
        engine.dynamics.load_state_dict(ck['dynamics'])
        critic = GroupNormalizedCritic(2835, engine.step.adapter.input_spec).to(bank.device)
        critic.load_state_dict(ck['critic']); del model
        if snapshots is not None:
            layout = json.loads((Path(args.trajectories)/'physical_ablation.json').read_text())['variants']
            begin = next(i for i, (_, m, component, scale) in enumerate(layout)
                         if m == model_index and component == 'all' and scale == 1.)*ns
            states = [select_state(v, slice(begin, begin+ns)) for _, v in sorted(snapshots.items())]
            z = {k: torch.cat([v[k] for v in states]).to(bank.device) for k in states[0]}
        else:
            replay = ck['replay']; data = replay['data']
            valid = (data['version'] >= data['version'].max()-3) & ~data['done']
            valid &= data['z_frame']+16 < bank.ends.cpu()[data['z_song']]
            pairs = valid.nonzero()
            if len(pairs) < 16:
                raise ValueError('checkpoint中缺少16个当前策略附近的真实训练起点')
            row, column = pairs[torch.linspace(0, len(pairs)-1, 16).long()].unbind(-1)
            z = {k: data['z_'+k][row, column].to(bank.device) for k in replay['dynamic_keys']}
        parameters = tuple(control.adapter.parameters())
        with torch.no_grad():
            x = control.packed(control.fields(z)); proposal = control.propose(z)
            nominal = bank.ref(z['song'], z['frame'])
            zero = control.propose(z, zero=True)
            logits = control.adapter.net(x)
            action_difference = (proposal['action']-zero['action']).norm(dim=-1)
            saturation = (logits.tanh().abs() >= .98).float().mean()
        translation = []
        for shift in (0., .1, .3, .6, 1., 2., 5.):
            shifted = {k: v.detach().clone() for k, v in z.items()}
            shifted['state'][:, 0] += shift
            shifted['state'].requires_grad_()
            xx = control.packed(control.fields(shifted))
            pp = control.propose(shifted)
            _, terms = control.runtime.reward(shifted['state'], pp['action'].detach(), z['action'], nominal, diagnostics=True)
            root_reward = .01*terms['root_pos']
            gp, = torch.autograd.grad(root_reward.sum(), shifted['state'])
            translation.append(dict(shift_m=shift, input_max_abs=float((xx-x).detach().abs().max()),
                action_max_abs=float((pp['action']-proposal['action']).detach().abs().max()),
                mean_root_reward=float(root_reward.detach().mean()), mean_root_x_gradient_abs=float(gp[:, 0].abs().mean()),
                terminated=int(control.runtime.termination(shifted['state'], nominal).sum())))
        report = dict(model_index=model_index, checkpoint=str(path), accepted_updates=ck['state']['accepted_updates'],
            checkpoint_phase=ck['state']['phase'], checkpoint_critic_ready=ck['state']['critic_ready'],
            starts_source=args.starts_source,
            physical_starts=len(z['state']), start_frames=z['frame'].tolist(), translation=translation,
            residual_saturation=float(saturation), logits_abs_max=float(logits.abs().max()),
            logits_abs_median=float(logits.abs().median()),
            float32_tanh_derivative_zero_fraction=float((1-logits.tanh().square() == 0).float().mean()),
            action_change_from_zero_L2_median=float(action_difference.median()),
            action_change_over_tangent_radius_median=float(action_difference.median()/.02), horizons=[])
        def grad(value):
            values = torch.autograd.grad(value.mean(), parameters, retain_graph=True, allow_unused=True)
            return torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p, v in zip(parameters, values)])
        def cos(a, b):
            return float(torch.dot(a, b)/(a.norm()*b.norm()).clamp_min(1e-20))
        with engine.dynamics.adapter_phase(), frozen_parameters(critic):
            for horizon in (1, 4, 8):
                tr = rollout(control, engine.dynamics, critic, z, horizon, terminal_value=True)
                dense = tr['objective']-tr['terminal_contribution']
                g_dense, g_terminal, g_track = grad(dense), grad(tr['terminal_contribution']), grad(tr['execution_objective'])
                tracking_only = tr['execution_objective']-tr['weighted']['music']
                g_tracking_only = grad(tracking_only)
                live = z['state'].new_ones(len(z['state'])); rate = live*0; limit = live*0
                previous = z['action']
                for t, (action, state) in enumerate(zip(tr['actions'], tr['states'])):
                    rate = rate-.002*.99**t*live*(action-previous).square().sum(-1)
                    q = state[:, 13:34]
                    limit = limit-.04*.99**t*live*(torch.relu(q-control.runtime.mdp.soft_hi)
                                                    +torch.relu(control.runtime.mdp.soft_lo-q)).sum(-1)
                    live = live*tr['continuation'][t]; previous = action
                g_rate, g_limit = grad(rate), grad(limit)
                end = tr['end']
                physical = {k: v if k == 'state' else v.detach() for k, v in end.items()}
                context = {k: v.detach() if k == 'state' else v for k, v in end.items()}
                discount = .99**horizon*tr['continuation'].prod(0).detach()
                g_physical = grad(discount*critic(control.critic_features(physical)))
                g_context = grad(discount*critic(control.critic_features(context)))
                values = {k: float(v.detach().mean()) for k, v in tr['weighted'].items()}
                component_norms = {k: float(grad(v).norm()) for k, v in tr['weighted'].items()}
                report['horizons'].append(dict(horizon=horizon, mean_dense=float(dense.detach().mean()),
                    mean_terminal=float(tr['terminal_contribution'].detach().mean()),
                    dense_gradient_norm=float(g_dense.norm()), terminal_gradient_norm=float(g_terminal.norm()),
                    terminal_over_dense_gradient=float(g_terminal.norm()/g_dense.norm().clamp_min(1e-20)),
                    terminal_physical_state_gradient_norm=float(g_physical.norm()),
                    terminal_context_gradient_norm=float(g_context.norm()),
                    terminal_vs_tracking_cosine=cos(g_terminal, g_track), dense_vs_tracking_cosine=cos(g_dense, g_track),
                    full_objective_vs_tracking_cosine=cos(g_dense+g_terminal, g_tracking_only),
                    tracking_gradient_norm=float(g_tracking_only.norm()), action_rate_gradient_norm=float(g_rate.norm()),
                    joint_limit_gradient_norm=float(g_limit.norm()),
                    action_rate_over_tracking_gradient=float(g_rate.norm()/g_tracking_only.norm().clamp_min(1e-20)),
                    action_rate_vs_tracking_cosine=cos(g_rate, g_tracking_only),
                    values=values, weighted_gradient_norms=component_norms))
                del tr, dense, g_dense, g_terminal, g_track, g_physical, g_context
        reports.append(report)
        del ck
        print(json.dumps(report, ensure_ascii=False), flush=True)
    write_json(output/'mechanism.json', dict(models=reports,
        evidence='frozen_checkpoint_autograd_on_causal_real_starts_no_optimizer_update',
        caveat='value calibration does not establish value input-gradient accuracy; ablations are separate real physics'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoints', nargs=2, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-envs', type=int, default=2048)
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--mode', choices=['collect', 'mechanism', 'evaluation'], default='collect')
    parser.add_argument('--trajectories')
    parser.add_argument('--starts-source', choices=['trajectory', 'checkpoint_replay'], default='trajectory')
    preliminary, _ = parser.parse_known_args()
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(preliminary.config), resolve=True)
    sys.path.insert(0, cfg['supermimic']['repo'])
    from isaaclab.app import add_launcher_args
    add_launcher_args(parser)
    args = parser.parse_args()
    import torch
    from gem.closedloop.reference_adapter.application import build_engine, read_tasks, seed_all
    from gem.closedloop.reference_adapter.vector_bank import VectorBank
    if torch.cuda.device_count() != 1:
        raise ValueError('诊断仅使用一个独立可见GPU')
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); seed_all(42)
    write_json(output/'manifest.json', dict(command=[sys.executable, *sys.argv], config=cfg,
        pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), source=os.popen('git rev-parse HEAD').read().strip(),
        started_unix=time.time(), mode=args.mode, modifies_training=False))
    engine = build_engine(cfg, 'cuda:0')
    tasks = read_tasks(cfg['tasks']); bank = VectorBank(tasks, engine, 'cuda:0')
    if args.mode == 'collect': collect(args, cfg, engine, tasks, bank)
    elif args.mode == 'evaluation': evaluate(args, cfg, engine, bank)
    else: mechanism(args, engine, bank)


if __name__ == '__main__':
    main()
