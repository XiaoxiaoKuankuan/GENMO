#!/usr/bin/env python3
"""在联合训练任务上采集零 Adapter 真实转移，固定速度与执行误差量纲。

仅读取新联合Stage1名义缓存，以50:50音乐/文本采样调用冻结Actor和真实Newton
物理；不更新任何网络，不访问validation/evaluation做标定。逐模态记录实际转移
数量与误差分布，计算每关节速度RMS作为固定s0、真实跟踪误差P75作为精度标尺。
输出独立music_baseline.pt、calibration.json、零策略Replay和runtime.yaml；保留当前
H64、WM H8、direct及奖励权重，不运行梯度调参，也不将标定结果当作物理收益。
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=1024)
    parser.add_argument('--num-envs', type=int, default=64)
    preliminary, _ = parser.parse_known_args()
    cfg = OmegaConf.to_container(OmegaConf.load(preliminary.config), resolve=True)
    sys.path.insert(0, cfg['supermimic']['repo'])
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 8 or args.num_envs < 2 or args.num_envs % 2:
        raise ValueError('零标定至少8步且使用正偶数环境')
    import torch
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from supermimic.bridge.bundle import atomic_save
    from gem.closedloop.reference_adapter.application import build_engine, read_tasks, seed_all
    from gem.closedloop.reference_adapter.vector_bank import VectorBank
    from gem.closedloop.reference_adapter.vector_control import VectorControl
    from gem.closedloop.reference_adapter.vector_collector import VectorCollector
    from gem.closedloop.reference_adapter.vector_replay import VectorReplay
    from gem.closedloop.reference_adapter.precision import errors, calibrate, NAMES, REFERENCE_NAMES, REFERENCE_FLOORS
    seed_all(cfg['training']['seed'])
    engine = build_engine(cfg, 'cuda:0', calibrating=True)
    tasks = [task for task in read_tasks(cfg['tasks']) if task['split'] == 'train']
    bank = VectorBank(tasks, engine, 'cuda:0')
    if set(bank.task_modalities.tolist()) != {0, 1}:
        raise ValueError('联合零标定必须同时有music/text训练任务')
    control = VectorControl(engine, bank)
    control.adapter.disable_root_xy = True
    control.soft_joint_limit_cost = False
    control.sm_soft_termination_only = True
    control.disable_root_orientation = not cfg['alternating'].get('root_orientation', True)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    replay = VectorReplay(args.steps*args.num_envs, args.num_envs, 'cuda:0', compute_mc=False,
                          task_modalities=bank.task_modalities)
    values = {kind: {name: [] for name in (*NAMES, 'qd')} for kind in (0, 1)}
    sim_cfg = make_sim('cuda:0', args)
    with launch_simulation(sim_cfg, args):
        collector = VectorCollector(StreamingPhysicalEnv(sim_cfg, args.num_envs, engine.step.bundle), control, seed=42)
        collector.uniform_start = True
        collector.reset(split='train', perturb=False)
        for index in range(args.steps):
            with torch.no_grad():
                row = collector.step(replay, zero=True, split='train', perturb=False)
                z, end = row['start'], row['end']
                nominal = bank.ref(z['song'], z['frame']+1)
                measured = errors(control, z, end['state'], nominal, row['raw'])
                for kind in (0, 1):
                    mask = bank.task_modalities[z['song']] == kind
                    for key in NAMES:
                        values[kind][key].append(measured[key][mask].cpu())
                    values[kind]['qd'].append(end['state'][mask, 34:].cpu())
            if (index+1) % 64 == 0:
                print(json.dumps(dict(zero_calibration_steps=index+1, real_transitions=collector.real_transitions)), flush=True)
    grouped = {kind: {name: torch.cat(chunks) for name, chunks in fields.items()} for kind, fields in values.items()}
    if any(not len(grouped[kind]['qd']) for kind in (0, 1)):
        raise ValueError('真实采集未覆盖两个模态，不写伪标定结果')
    # 两模态RMS等权；P75使用相同数量真实样本，长歌不会压过短文本动作。
    s0 = (sum(grouped[kind]['qd'].square().mean(0) for kind in (0, 1))/2).sqrt().clamp_min(.05)
    count = min(len(grouped[kind]['qd']) for kind in (0, 1))
    balanced = {name: torch.cat([grouped[kind][name][torch.linspace(0, len(grouped[kind][name])-1, count).long()]
                                for kind in (0, 1)]) for name in NAMES}
    calibration = dict(source='zero_adapter_real_training_episodes', split='train', scales=calibrate(balanced),
        modality_counts={('music' if k == 0 else 'text'): len(grouped[k]['qd']) for k in (0, 1)},
        tasks=[dict(id=t['id'], condition_kind=t['condition_kind'], contract=c)
               for t, c in zip(tasks, bank.task_contracts)], optimizer_updates=0,
        evidence='actual_physics_calibration_not_policy_improvement')
    reference = {kind: {name: [] for name in REFERENCE_NAMES} for kind in (0, 1)}
    for song in bank.by_split['train'].tolist():
        frames = torch.arange(int(bank.ends[song])+1, device=bank.device)
        ref = bank.ref(torch.full_like(frames, song), frames)
        fields = dict(joint_velocity=ref.qd, joint_acceleration=ref.qd.diff(dim=0)/.02,
            root_velocity=ref.root_w, root_acceleration=ref.root_w.diff(dim=0)/.02,
            joint_overlap=ref.q.diff(dim=0), root_overlap=ref.root_w*.02)
        kind = int(bank.task_modalities[song])
        for name, value in fields.items():
            reference[kind][name].append(value.detach().abs().flatten().cpu())
    soft_scales = {}
    for name in REFERENCE_NAMES:
        values = [torch.cat(reference[kind][name]) for kind in (0, 1)]
        count = min(map(len, values))
        equal = torch.cat([v[torch.linspace(0, len(v)-1, count).long()] for v in values])
        soft_scales[name] = max(float(torch.quantile(equal, .75)), REFERENCE_FLOORS[name])
    calibration.update(soft_reference_scales=soft_scales,
        soft_reference_scale_source='new_frozen_multimodal_stage1_train_cache_equal_modality_P75',
        reward_weights='inherited_fixed_configuration_not_retuned')
    atomic_save(dict(source='zero_adapter_real_training_episodes', s0=s0,
        joint_names=tuple(engine.step.bundle.spec.joint_names), tasks=calibration['tasks'],
        modality_counts=calibration['modality_counts']), output/'music_baseline.pt')
    atomic_save(dict(replay=replay.state_dict(), source='zero_adapter_real_training_episodes',
        tasks=tasks, config=cfg), output/'zero_replay.pt')
    (output/'calibration.json').write_text(json.dumps(calibration, ensure_ascii=False, indent=2, allow_nan=False))
    cfg['music_baseline'] = str(output/'music_baseline.pt')
    cfg['alternating']['calibration'] = str(output/'calibration.json')
    cfg['alternating']['precision']['soft_reference_scales'] = soft_scales
    cfg['alternating'].pop('gradient_calibration', None)
    OmegaConf.save(OmegaConf.create(cfg), output/'runtime.yaml')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
