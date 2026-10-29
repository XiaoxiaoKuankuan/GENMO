"""正式训练的独立四至八环境真实物理评价子进程。

父进程只传入只读配置和当前Adapter快照，本进程独立构造相同SM/RobotSpec/Newton
物理、GENMO缓存和音乐合同，实际只创建4至8个环境。按相同歌曲初态、xi、无探索
分别执行零Adapter和当前模型，复用现有完整整歌/前缀/失败/接触脚滑评价与配对检查。
只读基线缓存必须来自同一专用环境数量和评价预算；首次切换不会沿用2048环境基线。
不访问训练Replay、优化器或父进程RNG，不产生训练更新；报告写入新的显式运行目录。
"""

import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--num-envs', type=int, default=4)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--adapter-updates', type=int, required=True)
    p.add_argument('--baseline-source')
    p.add_argument('--repeat-candidate', type=int, default=1)
    args = p.parse_args()
    if args.num_envs not in range(4, 9):
        raise ValueError('专用物理评价只允许4至8个实际环境')
    if args.repeat_candidate < 1:
        raise ValueError('重复物理评价次数必须为正')
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    sys.path.insert(0, cfg['supermimic']['repo'])
    from isaaclab.app import add_launcher_args, launch_simulation
    launcher_parser = argparse.ArgumentParser()
    add_launcher_args(launcher_parser)
    launcher = launcher_parser.parse_args(['--visualizer', 'none', '--deterministic'])
    import torch
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from gem.closedloop.reference_adapter.application import build_engine, read_tasks, seed_all
    from gem.closedloop.reference_adapter.vector_bank import VectorBank
    from gem.closedloop.reference_adapter.vector_control import VectorControl
    from gem.closedloop.reference_adapter.vector_collector import VectorCollector
    from gem.closedloop.reference_adapter.reference_constraints import ReferenceProjection
    from gem.closedloop.reference_adapter.precision import PrecisionObjective
    from gem.closedloop.reference_adapter.vector_evaluation import evaluate_physical
    from gem.closedloop.reference_adapter.paired_evaluation import paired_comparison
    from gem.closedloop.reference_adapter.rewards import sm_mdp_contract
    from gem.closedloop.reference_adapter.vector_entry import compile_kinematics
    seed_all(42); torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    engine = build_engine(cfg, 'cuda:0')
    snapshot = torch.load(args.snapshot, map_location='cpu', weights_only=False, mmap=True)
    engine.step.adapter.load_state_dict(snapshot['adapter'])
    bank = VectorBank(read_tasks(cfg['tasks']), engine, 'cuda:0')
    control = VectorControl(engine, bank)
    control.adapter.disable_root_xy = True
    control.disable_root_orientation = not cfg['alternating'].get('root_orientation', True)
    control.soft_joint_limit_cost = False
    control.sm_soft_termination_only = True
    control.proposal_feasibility_weight = cfg['alternating'].get('proposal_feasibility_weight', 0.)
    control.projection = ReferenceProjection(control.bundle.spec, control.adapter.bounds,
        **cfg['alternating']['projection']).cuda()
    control.projection.compile_derivatives = cfg['alternating'].get('compile_so3_derivatives', True)
    if args.baseline_source:
        import importlib.util, types
        path = Path(args.baseline_source)/'gem/closedloop/reference_adapter/so3_projection.py'
        name = 'gem.closedloop.reference_adapter.dedicated_baseline_so3'
        spec = importlib.util.spec_from_file_location(name, path)
        old = importlib.util.module_from_spec(spec); sys.modules[name] = old; spec.loader.exec_module(old)
        control.projection.project_rotation = types.MethodType(lambda self, *a: old.project_rotation_implicit(self, *a), control.projection)
    calibration = json.loads(Path(cfg['alternating']['calibration']).read_text())
    control.precision = PrecisionObjective(calibration['scales'], **cfg['alternating']['precision'])
    compile_kinematics(engine, bank)
    sim_cfg = make_sim('cuda:0', launcher)
    begin = time.perf_counter()
    with launch_simulation(sim_cfg, launcher):
        env = StreamingPhysicalEnv(sim_cfg, args.num_envs, control.bundle)
        collector = VectorCollector(env, control, seed=42)
        for split in ['validation', 'evaluation']:
            scope = 'full_song' if args.max_steps is None else f'prefix_{args.max_steps}'
            key = f'{split}_{scope}'
            zero_path = out/f'{key}_zero_dedicated{args.num_envs}.json'
            if zero_path.exists():
                zero = json.loads(zero_path.read_text())
                if (zero['num_envs'] != args.num_envs or zero['max_steps'] != args.max_steps
                        or zero.get('sm_mdp_contract') != sm_mdp_contract(control.runtime)):
                    raise ValueError('专用零基线环境数、评价预算或奖励/终止标准不一致；请使用新评价目录')
            else:
                seed_all(42); collector.generator.manual_seed(42)
                zero = evaluate_physical(collector, zero=True, split=split, max_steps=args.max_steps)
                zero.update(adapter_updates=0, exploration_noise=0., evaluation_scope=scope,
                            evaluation_backend='dedicated_physics', actual_physics_envs=args.num_envs)
                zero_path.write_text(json.dumps(zero, ensure_ascii=False, indent=2))
            for repeat in range(args.repeat_candidate):
                seed_all(42); collector.generator.manual_seed(42)
                report = evaluate_physical(collector, zero=False, split=split, max_steps=args.max_steps)
                report.update(adapter_updates=args.adapter_updates, exploration_noise=0., evaluation_scope=scope,
                    zero_baseline_key=key, evaluation_backend='dedicated_physics', actual_physics_envs=args.num_envs,
                    repeat_index=repeat)
                report['paired_zero'] = paired_comparison(zero, report)
                report['paired_zero_difference'] = {k: report[k]-zero[k] if report[k] is not None and zero[k] is not None else None
                    for k in ['completion_rate','mean_tracking','mean_full_horizon_tracking','mean_root_position_rmse_m','mean_jerk_rms','mean_beat_f1']}
                suffix = '' if repeat == 0 else f'_repeat{repeat}'
                (out/f'{split}_{args.adapter_updates:06d}{suffix}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out/f'dedicated_timing_{args.adapter_updates:06d}.json').write_text(json.dumps(dict(
        seconds=time.perf_counter()-begin, actual_physics_envs=args.num_envs, training_physics_advanced=False,
        physical_transitions=collector.real_transitions, measured_by_role=collector.real_by_role,
        snapshot=args.snapshot), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
