"""验证最新SuperMimic资产在GENMO桥接中的真实物理，并采集训练精度标尺。

本入口只运行有限Newton/MuJoCo-Warp物理步，不更新Adapter、Actor或世界模型。
逐项回读底层质量、完整惯量、PD、电枢和被动摩擦，检查ξ119与实际参数一致、
大连杆碰撞过滤、部分reset隔离、动作裁剪以及XML可微FK与USD实体的一致性。
训练模式使用最新SM逐episode随机化；默认名义模式用64首训练歌曲零Adapter
误差重新计算P75精度标尺，禁止借用旧物理标尺或将短段统计称为整歌效果。
报告写入显式独立目录，包含checkpoint、资产和配置来源，不覆盖已有输出。
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import argparse
    import importlib.util
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--training-mode', action='store_true')
    parser.add_argument('--save-train-starts', action='store_true',
                        help='额外保存64首训练歌曲各两个真实零Adapter起点，供独立梯度标定')
    preliminary, _ = parser.parse_known_args()
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(preliminary.config), resolve=True)
    sys.path.insert(0, cfg['supermimic']['repo'])
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    import torch
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from supermimic.physics_context import context_slices, nominal_context
    from supermimic.rotations import quat_to_mat
    from gem.closedloop.reference_adapter.application import build_engine, read_tasks, seed_all
    from gem.closedloop.reference_adapter.vector_bank import VectorBank
    from gem.closedloop.reference_adapter.vector_control import VectorControl
    from gem.closedloop.reference_adapter.vector_collector import VectorCollector
    from gem.closedloop.reference_adapter.precision import PrecisionObjective, errors, calibrate, ERROR_NAMES
    from gem.closedloop.reference_adapter.vector_entry import compile_kinematics, json_write
    helper_spec = importlib.util.spec_from_file_location('sm_environment_validation',
        Path(cfg['supermimic']['repo'])/'tools/validate_robot_environment.py')
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_all(42)
    torch.set_num_threads(1)
    engine = build_engine(cfg, 'cuda:0')
    bank = VectorBank(read_tasks(cfg['tasks']), engine, 'cuda:0')
    compile_kinematics(engine, bank)
    control = VectorControl(engine, bank)
    control.precision = PrecisionObjective()
    control.detailed_reference = True
    sim_cfg = make_sim('cuda:0', args, eval_mode=not args.training_mode, spec=control.bundle.spec)
    with launch_simulation(sim_cfg, args), torch.no_grad():
        env = StreamingPhysicalEnv(sim_cfg, args.num_envs, control.bundle, eval_mode=not args.training_mode)
        collector = VectorCollector(env, control)
        ids = torch.arange(args.num_envs, device='cuda')
        songs = bank.by_split['train'][ids % len(bank.by_split['train'])]
        frames = torch.zeros_like(songs)
        collector.reset(songs=songs, frames=frames, perturb=False)
        initial = env.observe()['state'].clone()
        report = dict(config=cfg, training_mode=args.training_mode, xi_dimension=env.spec.xi_dim,
            num_envs=args.num_envs, checkpoint=control.bundle.metadata,
            joint_dynamics=helper.joint_dynamics_stats(env),
            domain_randomization=helper.domain_randomization_stats(env), collisions=helper.collision_stats(env))
        checks = dict(domain_randomization=report['domain_randomization']['passed'],
                      collisions=report['collisions']['passed'])
        for name, value in report['joint_dynamics'].items():
            if isinstance(value, dict):
                checks['joint_'+name] = all(value[k]['finite'] and value[k]['max_abs'] < 1e-6 for k in ('articulation','backend'))
        if args.training_mode:
            checks['armature_ranges'] = report['joint_dynamics']['training_armature_within_ranges']
        layout = context_slices(21)
        actual = dict(friction=env._mu_view[:, :1],
            torso_com_offset=env.robot.data.body_com_pos_b.torch[:, env.torso_id]-env.nominal_torso_com,
            torso_mass=env.robot.data.body_mass.torch[:, env.mass_body_id:env.mass_body_id+1],
            torso_inertia=env.robot.data.body_inertia.torch[:, env.mass_body_id].reshape(env.N,9))
        for name, field in [('joint_armature','joint_armature'),('joint_kp','joint_stiffness'),
                            ('joint_kd','joint_damping'),('joint_effort_limits','joint_effort_limits')]:
            actual[name] = getattr(env.robot.data, field).torch[:, env.to_policy]
        report['xi_errors'] = {name: helper.error_stats(env.xi[:, layout[name]], value) for name,value in actual.items()}
        checks.update({'xi_'+name: val['finite'] and val['max_abs'] < 1e-6 for name,val in report['xi_errors'].items()})
        if not args.training_mode:
            checks['nominal_xi'] = bool(torch.allclose(env.xi, nominal_context(env.spec, env.N, env.device), atol=1e-6, rtol=0))
        body_ids = [env.robot.body_names.index(name) for name in env.spec.tracked_bodies]
        poses = env.robot.data.body_link_pose_w.torch[:, body_ids]
        velocities = env.robot.data.body_link_vel_w.torch[:, body_ids]
        quat = poses[...,3:7]
        expected = (poses[...,:3]-env.scene.env_origins[:,None], quat_to_mat(torch.cat((quat[...,3:],quat[...,:3]),-1)),velocities[...,:3],velocities[...,3:])
        report['fk_errors'] = {name: helper.error_stats(a,b) for name,a,b in zip(
            ('p','R','v','w'),control.runtime.kin.forward(*env.state()), expected)}
        checks.update({'fk_'+name: val['finite'] and val['max_abs'] < 3e-4 for name,val in report['fk_errors'].items()})
        before = env.xi.clone()
        saved_mass = env.robot.data.body_mass.torch.clone()
        saved_kp = env.robot.data.joint_stiffness.torch.clone()
        env.reset(ids[:1], initial[:1])
        checks['partial_reset_context'] = bool(torch.equal(before[1:], env.xi[1:]))
        checks['partial_reset_physics'] = bool(torch.equal(saved_mass[1:], env.robot.data.body_mass.torch[1:]) and torch.equal(saved_kp[1:], env.robot.data.joint_stiffness.torch[1:]))
        if args.training_mode:
            checks['resampled_context'] = not torch.equal(before[0], env.xi[0])
        before = env.xi.clone()
        env.step_physics(torch.full((env.N,21),100.,device=env.device))
        checks['context_constant_inside_episode'] = torch.equal(before,env.xi)
        checks['physical_action_clipped'] = bool((env.a_prev == 20).all())
        collector.reset(songs=songs, frames=frames, perturb=False)
        collected = {name: [] for name in ERROR_NAMES}
        for _ in range(args.steps):
            row = collector.step(zero=True, split='train', perturb=False)
            nominal = bank.ref(row['end']['song'],row['end']['frame'])
            values = errors(control,row['start'],row['end']['state'],nominal,row['raw'])
            for name,value in values.items(): collected[name].append(value.detach())
        checks['finite_state'] = bool(torch.isfinite(env.observe()['state']).all())
        report.update(checks=checks, passed=all(checks.values()), measured_transitions=args.steps*env.N,
                      boundary='finite_physics_contract_and_train_only_scale_not_full_song_quality')
        json_write(output/'physics_report.json',report)
        if report['passed'] and not args.training_mode:
            json_write(output/'zero_train_calibration.json',dict(scales=calibrate({k:torch.cat(v) for k,v in collected.items()}),
                songs=len(set(songs.cpu().tolist())),scan_steps=args.steps,num_envs=env.N,
                checkpoint=cfg['supermimic']['checkpoint'],robot_asset_dir=cfg['supermimic']['robot_asset_dir'],
                sampling='zero_adapter_new_nominal_physics_train_only',seed=42))
        if report['passed'] and args.save_train_starts:
            if args.training_mode or args.num_envs != len(bank.by_split['train']):
                raise ValueError('标定起点要求名义物理，且每首训练歌曲对应一个环境')
            # 只采训练歌曲；随机歌曲内位置后真实执行32步，取得真实历史/计划/音乐状态。
            # episode中途失败的起点不混入幸存轨迹，重采次数及失败数明确保留。
            songs = bank.by_split['train'].clone()
            picked = {int(song): [] for song in songs.cpu().tolist()}
            generator = torch.Generator(device='cuda').manual_seed(20261103)
            rejected = 0
            for attempt in range(8):
                frames = (torch.rand(len(songs), device='cuda', generator=generator)
                          * (bank.ends[songs]-128).clamp_min(1)).long()
                collector.reset(songs=songs, frames=frames, perturb=False)
                episodes = collector.state['episode'].clone()
                for _ in range(32):
                    collector.step(zero=True, split='train', perturb=False)
                z = collector.state
                valid = (z['episode'] == episodes) & (z['frame']+64 < bank.ends[z['song']])
                rejected += int((~valid).sum())
                for index, song in enumerate(songs.cpu().tolist()):
                    if bool(valid[index]) and len(picked[song]) < 2:
                        picked[song].append({k: v[index:index+1].detach().cpu().clone() for k,v in z.items()})
                if all(len(rows) == 2 for rows in picked.values()):
                    break
            if any(len(rows) != 2 for rows in picked.values()):
                raise RuntimeError('真实零Adapter起点未覆盖每首训练歌曲两次，不能伪造均衡标定')
            rows = [row for group in picked.values() for row in group]
            torch.save(dict(starts={k: torch.cat([row[k] for row in rows]) for k in rows[0]},
                split='train', policy_version=0, source='real_nominal_zero_adapter',
                checkpoint=cfg['supermimic']['checkpoint'], checkpoint_sha256=control.bundle.metadata['sha256'],
                tasks=cfg['tasks'], seed=20261103, attempts=attempt+1, rejected_episode_starts=rejected),
                output/'fixed_train_starts.pt')
        print(json.dumps(dict(passed=report['passed'],checks=checks),ensure_ascii=False),flush=True)
        if not report['passed']: raise RuntimeError('新物理桥接验证失败，见physics_report.json')


if __name__ == '__main__':
    main()
