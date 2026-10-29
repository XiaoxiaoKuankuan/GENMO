"""用真实训练Replay标定根XY和软连续性权重，并生成八组独立正式配置。

只读既有checkpoint中的零Adapter真实采集起点，按64首train歌曲等量抽样，
所有组使用原始预训练WM/零Adapter；不加载旧退化策略，不拟合validation/evaluation。
根XY权重按零策略下参数梯度相对原SM六项Tracking的10%/30%标定。连续性在零
残差处理论梯度为零，因此使用预先声明的微小非零网络偏置探针，六项各自平衡后
将合计梯度定为Tracking的5%/15%。探针仅用于标定，完成即恢复零Adapter。
物理量纲标尺取训练名义运动P75和数值下限，记录抽样身份、固定探针和完整梯度。
这些只是目标尺度校准，不能证明WM方向或长期物理改善。八组配置固定为
H32/H64 × 根辅助弱/强 × 连续性弱/强，Critic全部关闭，10000更新，不因评价退化退出。
"""

import argparse
import copy
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--per-song', type=int, default=2)
    args = parser.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, cfg['supermimic']['repo'])
    import torch
    from gem.closedloop.reference_adapter.application import build_engine, read_tasks, seed_all
    from gem.closedloop.reference_adapter.vector_bank import VectorBank
    from gem.closedloop.reference_adapter.vector_control import VectorControl
    from gem.closedloop.reference_adapter.precision import PrecisionObjective, REFERENCE_NAMES, REFERENCE_FLOORS
    from gem.closedloop.reference_adapter.objective_diagnostics import reward_gradients
    from gem.closedloop.reference_adapter.update_diagnostics import parameter_identity
    from supermimic.networks import Critic
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    seed_all(42)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out/'configs').mkdir()
    engine = build_engine(cfg, 'cuda:0')
    bank = VectorBank(read_tasks(cfg['tasks']), engine, 'cuda:0')
    control = VectorControl(engine, bank)
    control.adapter.disable_root_xy = True
    control.soft_joint_limit_cost = False
    control.sm_soft_termination_only = True
    critic = Critic(2835).cuda().requires_grad_(False)
    calibration = json.loads(Path(cfg['alternating']['calibration']).read_text())
    precision_weight = cfg['alternating']['precision']['weight']
    frozen = control.bundle.frozen_snapshot()
    initial_adapter = copy.deepcopy(control.adapter.state_dict())
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    replay = ck['replay']; data = replay['data']
    songs, frames = data['z_song'], data['z_frame']
    allowed = (data['version'] == 0) & (frames >= 10) & (frames+64 < bank.ends.cpu()[songs])
    allowed &= data['vector_time'] >= max(0, replay['total_vector_steps']-replay['size'])
    rows, columns = [], []
    generator = torch.Generator().manual_seed(42)
    for song in bank.by_split['train'].cpu().tolist():
        pairs = (allowed & (songs == song)).nonzero()
        if len(pairs) < args.per_song:
            raise ValueError(f'train歌曲{song}没有足够零策略连续支持H64的真实起点')
        chosen = pairs[torch.randperm(len(pairs), generator=generator)[:args.per_song]]
        rows.append(chosen[:, 0]); columns.append(chosen[:, 1])
    rows, columns = torch.cat(rows), torch.cat(columns)
    starts = {k: data['z_'+k][rows, columns].cuda() for k in replay['dynamic_keys']}
    torch.save(dict(starts={k:v.cpu() for k,v in starts.items()}, checkpoint=args.checkpoint,
        replay_rows=rows, replay_columns=columns, policy_version=0, split='train'), out/'fixed_train_starts.pt')
    del ck, replay, data
    # 固定单位尺度来自名义训练数据；不将这些数值解释为硬件运动极限。
    reference_values = {k: [] for k in REFERENCE_NAMES}
    for song in bank.by_split['train'].tolist():
        f = torch.arange(int(bank.ends[song])+1, device=bank.device)
        ref = bank.ref(torch.full_like(f, song), f)
        fields = dict(joint_velocity=ref.qd, joint_acceleration=ref.qd.diff(dim=0)/.02,
            root_velocity=ref.root_w, root_acceleration=ref.root_w.diff(dim=0)/.02,
            joint_overlap=ref.q.diff(dim=0), root_overlap=ref.root_w*.02)
        for name, value in fields.items():
            reference_values[name].append(value.abs().flatten())
    scales = {name: max(float(torch.quantile(torch.cat(value), .75)), REFERENCE_FLOORS[name])
              for name, value in reference_values.items()}
    del reference_values
    metadata = dict(checkpoint=args.checkpoint, starts=len(starts['state']), songs=len(bank.by_split['train']),
        split='train', soft_scales=scales, precision_scales=calibration['scales'],
        root_scale_m=.2, root_target_ratios=[.1,.3], continuity_target_ratios=[.05,.15],
        adapter_sha256=parameter_identity(control.adapter), wm_sha256=parameter_identity(engine.dynamics),
        nominal_tasks=cfg['tasks'], probe_latent_std=.02, seed=42, horizons={},
        boundary='training_gradient_scale_not_physical_direction_certificate',
        initialization='original_pretrained_WM_and_zero_Adapter_no_old_policy_weights')
    for h in (32, 64):
        control.adapter.load_state_dict(initial_adapter)
        control.precision = PrecisionObjective(calibration['scales'], weight=precision_weight, root_xy_weight=1.)
        zero_report, zero_grads = reward_gradients(control, engine.dynamics, critic, starts, h)
        sm_norm = zero_grads['sm_tracking'].norm()
        root_norm = zero_grads['root_non_saturating'].norm()
        if not bool(torch.isfinite(sm_norm+root_norm)) or float(min(sm_norm, root_norm)) < 1e-10:
            raise ValueError('零策略根/Tracking梯度非有限或不可辨识，不能伪造标定权重')
        root_base = float(sm_norm/root_norm)
        del zero_grads
        with torch.no_grad():
            probe = torch.randn(control.adapter.net[-1].bias.shape, generator=generator)*.02
            control.adapter.net[-1].bias.copy_(probe.cuda())
        control.precision = PrecisionObjective(calibration['scales'], weight=precision_weight, root_xy_weight=1.,
            soft_reference_weights=dict.fromkeys(REFERENCE_NAMES, 1.), soft_reference_scales=scales)
        probe_report, probe_grads = reward_gradients(control, engine.dynamics, critic, starts, h)
        names = ['reference_soft_'+name for name in REFERENCE_NAMES]
        norms = {name: probe_grads[name].norm() for name in names}
        if any(not bool(torch.isfinite(n)) or float(n) < 1e-10 for n in norms.values()):
            raise ValueError('非零探针没有可辨识连续性梯度，不能用无穷权重补偿')
        equal = {name: 1/float(norms[name]) for name in names}
        combined = sum(probe_grads[name]*equal[name] for name in names)
        soft_base = float(probe_grads['sm_tracking'].norm()/combined.norm())
        metadata['horizons'][str(h)] = dict(zero=zero_report, nonzero_probe=probe_report,
            root_unit_ratio=root_base, soft_unit_weights={name.removeprefix('reference_soft_'): equal[name]*soft_base for name in names},
            probe_bias=probe.tolist())
        del probe_grads, combined
        control.adapter.load_state_dict(initial_adapter)
        for i, root_ratio in enumerate((.1, .3)):
            for j, soft_ratio in enumerate((.05, .15)):
                gpu = (0 if h == 32 else 4)+i*2+j
                new = copy.deepcopy(cfg)
                a = new['alternating']; v = new['vector']; t = new['training']
                a.update(critic_enabled=False, root_xy=False, root_orientation=True, action_noise=0., exploration_fraction=0.,
                    objective_version='sm_music_absolute_xy_goal_continuity_v9',
                    stage_budget='10000_Adam_updates_metrics_never_stop_training',
                    experiment_label=f'H{h}_rootgrad{root_ratio:g}_softgrad{soft_ratio:g}_seed42',
                    objective_gradient_every=100, objective_gradient_samples=32,
                    gradient_calibration=str(out/'calibration.json'),
                    evaluation_updates=[0,10,20,50,100,200,500]+list(range(1000,10001,1000)))
                a['precision'] = dict(weight=precision_weight, root_xy_scale=.2, root_xy_weight=root_ratio*root_base,
                    soft_reference_weights={name.removeprefix('reference_soft_'): soft_ratio*equal[name]*soft_base for name in names},
                    soft_reference_scales=scales)
                t.update(seed=42, adapter_lr=1e-5, wm_lr=3e-5, actor_batch=1024, backward_microbatch=256,
                         wm_batch=2048, wm_fit_window=8, wm_updates_per_iteration=10)
                v.update(num_envs=2048, target_accepted_updates=10000, checkpoint_every=200, sampling_seed=42)
                new['bptt_steps'] = h
                (out/'configs'/f'gpu{gpu}.yaml').write_text('# 训练集梯度标定的独立根XY/软连续性对照；不恢复硬投影，不按评价退化停止。\n'+yaml.safe_dump(new, allow_unicode=True, sort_keys=False))
        (out/'calibration.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
        print(json.dumps(dict(horizon=h, root_unit_ratio=root_base, scales=scales, zero=zero_report,
                             nonzero=probe_report), ensure_ascii=False), flush=True)
    control.bundle.assert_unchanged(frozen)
    print('八组标定完成；Adapter恢复零初始化，未执行优化器更新或物理验收。', flush=True)


if __name__ == '__main__':
    main()
