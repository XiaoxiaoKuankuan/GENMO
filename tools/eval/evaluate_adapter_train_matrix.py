"""用完整歌曲比较一组只读Adapter快照，不产生任何训练更新。

输入清单包含候选名称、冻结配置、checkpoint与预先声明的选择规则；每个进程复用
一个独立Newton仿真和冻结SuperMimic Actor，逐个加载Adapter权重。先执行两次零
Adapter，再从相同歌曲初态、物理参数及50Hz时钟执行候选，沿用正式配对评价的
六项Tracking、关节/姿态/根误差、完成率、真实接触脚滑、jerk与音乐指标。
默认只取train；显式--splits可评价随机混合歌曲，并保留每首原始划分、来源和
本轮训练是否见过的身份，不将混合统计描述成纯未见测试，不更改任何训练进程。
每个候选保留原配置的软成本，但排序依据原始名义参考的共同物理指标；归一化、
Actor与优化器均不更新。可选择保存真实轨迹并重复候选，用于最终视频和噪声复核。
输出必须是新目录；已有结果、模型、日志和正在使用的Replay全部保持只读。
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--repeat-candidate', type=int, default=1)
    parser.add_argument('--save-trajectories', action='store_true')
    parser.add_argument('--splits', nargs='+', default=['train'],
                        choices=['train', 'validation', 'audit', 'evaluation'])
    pre, _ = parser.parse_known_args()
    import yaml
    manifest = json.loads(pre.manifest.read_text())
    candidates = manifest['candidates'][pre.worker_index::pre.num_workers]
    if not candidates or pre.repeat_candidate < 1:
        raise ValueError('候选分片和重复次数必须有效')
    cfg = yaml.safe_load(Path(candidates[0]['config']).read_text())
    sys.path.insert(0, cfg['supermimic']['repo'])
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    import torch
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from gem.closedloop.reference_adapter.application import build_engine, read_tasks, seed_all, direct_reference_config
    from gem.closedloop.reference_adapter.vector_bank import VectorBank
    from gem.closedloop.reference_adapter.vector_control import VectorControl
    from gem.closedloop.reference_adapter.vector_collector import VectorCollector
    from gem.closedloop.reference_adapter.precision import PrecisionObjective
    from gem.closedloop.reference_adapter.vector_entry import compile_kinematics
    from gem.closedloop.reference_adapter.paired_evaluation import evaluate_fixed, paired_comparison, tensor_digest

    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    seed_all(42)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = direct_reference_config(cfg)
    cfg['observation_noise'] = False
    engine = build_engine(cfg, 'cuda:0')
    tasks = [t for t in read_tasks(cfg['tasks']) if t['split'] in args.splits]
    expected = manifest.get('expected_song_count', len(tasks))
    if len(tasks) != expected or len({t['song_id'] for t in tasks}) != expected:
        raise ValueError('实际任务必须覆盖声明数量的不同歌曲，不能用重复回放凑数量')
    if len({t.get('global_music_group_id', t['song_id']) for t in tasks}) != expected:
        raise ValueError('同一全局音乐组的不同裁剪不能计成独立歌曲')
    bank = VectorBank(tasks, engine, 'cuda:0')
    # selected仅为一次批量物理运行的索引；原任务及报告中的split保持不变。
    bank.by_split['selected'] = torch.arange(len(tasks), device=bank.device)
    control = VectorControl(engine, bank)
    control.adapter.disable_root_xy = True
    control.soft_joint_limit_cost = False
    control.sm_soft_termination_only = True
    calibration = json.loads(Path(cfg['alternating']['calibration']).read_text())
    actor_before = control.bundle.frozen_snapshot()
    run = dict(input_manifest=manifest, candidates=candidates, command=sys.argv,
               started_unix=time.time(), optimizer_steps=0, task_ids=[t['id'] for t in tasks],
               actual_physics_envs=len(tasks), splits=args.splits, seed=42,
               source_files={name: tensor_digest(torch.tensor(list((ROOT/name).read_bytes()), dtype=torch.uint8))
                             for name in ['gem/closedloop/reference_adapter/vector_control.py',
                                          'gem/closedloop/reference_adapter/precision.py']})
    (output/'manifest.json').write_text(json.dumps(run, ensure_ascii=False, indent=2))
    (output/'compile.json').write_text(json.dumps(compile_kinematics(engine, bank), indent=2))
    sim_cfg = make_sim('cuda:0', args)
    with launch_simulation(sim_cfg, args):
        env = StreamingPhysicalEnv(sim_cfg, len(tasks), control.bundle)
        collector = VectorCollector(env, control, seed=42)

        def evaluate(name, zero):
            seed_all(42)
            collector.generator.manual_seed(42)
            report = evaluate_fixed(collector, zero=zero, split='selected',
                trajectory_dir=output/(name+'_trajectories') if args.save_trajectories else None)
            by_id = {task['id']: task for task in tasks}
            for episode in report['episodes']:
                task = by_id[episode['task']]
                episode.update(split=task['split'], source=task.get('source'),
                    dataset_split=task.get('dataset_split'),
                    global_music_group_id=task.get('global_music_group_id', task['song_id']),
                    adapter_training_exact_task_seen=task.get('adapter_training_exact_task_seen'),
                    adapter_training_seen=task.get('adapter_training_seen', task['split']=='train'))
            report.update(model_label=name, training_updated=False)
            (output/(name+'.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps(dict(event='completed_evaluation', name=name,
                completion=report['completion_rate'], tracking=report['mean_full_horizon_tracking'],
                root=report['mean_root_position_rmse_m'], seconds=report['wall_seconds']), ensure_ascii=False), flush=True)
            return report

        control.disable_root_orientation = not cfg['alternating'].get('root_orientation', True)
        control.precision = PrecisionObjective(calibration['scales'], **cfg['alternating']['precision'])
        baseline = evaluate('zero', True)
        repeat = evaluate('zero_repeat', True)
        (output/'zero_repeat_comparison.json').write_text(json.dumps(paired_comparison(baseline, repeat), indent=2))
        for candidate in candidates:
            saved = torch.load(candidate['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
            candidate_cfg = yaml.safe_load(Path(candidate['config']).read_text())
            if (candidate_cfg['tasks'] != cfg['tasks'] or candidate_cfg.get('sm_mdp') != cfg.get('sm_mdp')
                    or candidate_cfg['supermimic']['checkpoint'] != cfg['supermimic']['checkpoint']):
                raise ValueError('候选名义数据、物理奖励或冻结Actor不同，不能共用本次配对基线')
            control.adapter.load_state_dict(saved['adapter'])
            control.disable_root_orientation = not candidate_cfg['alternating'].get('root_orientation', True)
            control.precision = PrecisionObjective(calibration['scales'], **candidate_cfg['alternating']['precision'])
            identity = tensor_digest(torch.cat([v.flatten() for v in saved['adapter'].values()]))
            updates = saved['state']['accepted_updates']
            del saved
            for repeat_index in range(args.repeat_candidate):
                name = candidate['label'] + (f'_repeat{repeat_index}' if repeat_index else '')
                report = evaluate(name, False)
                report.update(adapter_sha256=identity, adapter_updates=updates, checkpoint=candidate['checkpoint'],
                              paired_zero=paired_comparison(baseline, report), repeat_index=repeat_index)
                (output/(name+'.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2))
            control.bundle.assert_unchanged(actor_before)
        (output/'completed.json').write_text(json.dumps(dict(ended_unix=time.time(),
            real_transitions=collector.real_transitions, measured_transitions=collector.real_by_role,
            optimizer_steps=0, actor_unchanged=True), indent=2))


if __name__ == '__main__':
    main()
