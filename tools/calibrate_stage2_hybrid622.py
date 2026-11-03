"""为指定hybrid622/26k基座标定共同损失，并生成GPU0–5的六组正式训练配置。

输入必须是最新名义物理采集的真实零Adapter训练起点；每首训练歌曲两个起点，
含已实际执行的历史、参考计划和音乐状态。只构造可微WM轨迹与逐项参数梯度，
不执行任何Adam、不读取开发/评价指标、不导入旧Adapter或旧世界模型权重。

共同root和精度权重按H32/H64中较敏感的一档确定：根XY辅助梯度最多为原Tracking
的50%，四项精度合计最多20%；所有组共享权重以隔离时域因素。六项参考连续性用
固定微小非零偏置探针标定，H32合计为Tracking的5%，H64使用同样系数并报告实际
比例。探针完成后恢复原始零Adapter。额外jerk/足端惩罚为零，避免本轮混入另一因素。

六组为H32无Critic、H64无Critic、H32有Critic，各使用42/43两个种子；Critic独立
零初始化、实际TD预热、想象lambda-return与真实TD锚定，价值上界由实际奖励确定。
本标定仅规定可追溯的目标尺度，不证明WM梯度方向正确或整歌质量已提升。
"""

import argparse
import copy
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--starts', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), cfg['supermimic']['repo']]
    import torch
    from supermimic.networks import Critic
    from gem.closedloop.reference_adapter.application import build_engine, read_tasks, seed_all
    from gem.closedloop.reference_adapter.features import critic_dimension
    from gem.closedloop.reference_adapter.vector_bank import VectorBank
    from gem.closedloop.reference_adapter.vector_control import VectorControl
    from gem.closedloop.reference_adapter.precision import PrecisionObjective, NAMES, REFERENCE_NAMES, REFERENCE_FLOORS
    from gem.closedloop.reference_adapter.objective_diagnostics import reward_gradients
    from gem.closedloop.reference_adapter.update_diagnostics import parameter_identity

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
    critic = Critic(critic_dimension(control.adapter)).cuda().requires_grad_(False)
    initial = copy.deepcopy(control.adapter.state_dict())
    frozen = control.bundle.frozen_snapshot()
    payload = torch.load(args.starts, map_location='cpu', weights_only=False)
    if payload['split'] != 'train' or payload['policy_version'] != 0:
        raise ValueError('只能标定真实零Adapter训练起点')
    starts = {k: v.cuda() for k,v in payload['starts'].items()}
    songs, counts = starts['song'].unique(return_counts=True)
    if not torch.equal(songs.sort().values, bank.by_split['train'].sort().values) or not bool((counts == 2).all()):
        raise ValueError('必须每首训练歌曲恰好两个真实起点')
    calibration = json.loads(Path(cfg['alternating']['calibration']).read_text())
    values = {k: [] for k in REFERENCE_NAMES}
    for song in bank.by_split['train'].tolist():
        frames = torch.arange(int(bank.ends[song])+1, device=bank.device)
        ref = bank.ref(torch.full_like(frames, song), frames)
        fields = dict(joint_velocity=ref.qd, joint_acceleration=ref.qd.diff(dim=0)/.02,
            root_velocity=ref.root_w, root_acceleration=ref.root_w.diff(dim=0)/.02,
            joint_overlap=ref.q.diff(dim=0), root_overlap=ref.root_w*.02)
        for name, value in fields.items():
            values[name].append(value.abs().flatten())
    scales = {name: max(float(torch.quantile(torch.cat(value), .75)), REFERENCE_FLOORS[name])
              for name,value in values.items()}
    del values
    report = dict(checkpoint=cfg['supermimic']['checkpoint'], starts=args.starts,
        source_metadata={k:v for k,v in payload.items() if k != 'starts'},
        samples=len(starts['state']), songs=len(songs), precision_scales=calibration['scales'],
        soft_reference_scales=scales, root_gradient_max_ratio=.5, precision_gradient_max_ratio=.2,
        continuity_H32_gradient_ratio=.05, horizons={},
        adapter_sha256=parameter_identity(control.adapter), wm_sha256=parameter_identity(engine.dynamics),
        evidence='train_only_gradient_scale_not_physical_quality', optimizer_updates=0)
    generator = torch.Generator().manual_seed(20261103)
    probe = torch.randn(control.adapter.net[-1].bias.shape, generator=generator)*.02
    report['probe_bias'] = probe.tolist()
    for horizon in (32,64):
        control.adapter.load_state_dict(initial)
        control.precision = PrecisionObjective(calibration['scales'], weight=1., root_xy_weight=1.)
        zero, grads = reward_gradients(control, engine.dynamics, critic, starts, horizon)
        sm = grads['sm_tracking'].norm()
        root = grads['root_non_saturating'].norm()
        precision = sum(grads['precision_'+name] for name in NAMES).norm()
        if not all(bool(torch.isfinite(x)) and float(x) > 1e-10 for x in (sm,root,precision)):
            raise ValueError('根/精度梯度非有限或不可辨识，不能生成有效权重')
        info = dict(zero=zero, root_unit_weight=float(sm/root), precision_unit_weight=float(sm/precision))
        del grads
        with torch.no_grad():
            control.adapter.net[-1].bias.copy_(probe.cuda())
        control.precision = PrecisionObjective(calibration['scales'], weight=1., root_xy_weight=1.,
            soft_reference_weights=dict.fromkeys(REFERENCE_NAMES,1.), soft_reference_scales=scales)
        probe_report, grads = reward_gradients(control, engine.dynamics, critic, starts, horizon)
        names = ['reference_soft_'+name for name in REFERENCE_NAMES]
        norms = {name: grads[name].norm() for name in names}
        if not all(bool(torch.isfinite(x)) and float(x) > 1e-10 for x in norms.values()):
            raise ValueError('连续性探针梯度不可辨识')
        equal = {name: 1/float(norms[name]) for name in names}
        combined = sum(grads[name]*equal[name] for name in names)
        ratio = float(grads['sm_tracking'].norm()/combined.norm())
        info.update(probe=probe_report, continuity_unit_weights={name.removeprefix('reference_soft_'):equal[name]*ratio for name in names})
        if horizon == 32:
            soft = {name:.05*weight for name,weight in info['continuity_unit_weights'].items()}
        info['actual_shared_continuity_ratio'] = float(sum(grads['reference_soft_'+name]*weight for name,weight in soft.items()).norm()/grads['sm_tracking'].norm())
        del grads, combined
        report['horizons'][str(horizon)] = info
        print(json.dumps(dict(horizon=horizon, root_unit_weight=info['root_unit_weight'],
            precision_unit_weight=info['precision_unit_weight'], continuity_ratio=info['actual_shared_continuity_ratio'])),flush=True)
    control.adapter.load_state_dict(initial)
    control.bundle.assert_unchanged(frozen)
    root_weight = .5*min(x['root_unit_weight'] for x in report['horizons'].values())
    precision_weight = .2*min(x['precision_unit_weight'] for x in report['horizons'].values())
    report['shared_weights'] = dict(root_xy_weight=root_weight, weight=precision_weight, soft_reference_weights=soft)
    report['actual_gradient_ratios'] = {h:dict(root=root_weight/v['root_unit_weight'],
        precision=precision_weight/v['precision_unit_weight']) for h,v in report['horizons'].items()}
    for gpu,(horizon,enabled,seed) in enumerate([(32,False,42),(32,False,43),(64,False,42),(64,False,43),(32,True,42),(32,True,43)]):
        new = copy.deepcopy(cfg)
        a,t,v = new['alternating'],new['training'],new['vector']
        a.update(critic_enabled=enabled, critic_epochs=3, critic_minibatches=4, critic_real_td_weight=1.,
            wm_only_iterations=100, objective_version='hybrid622_26k_leg0_calibrated_horizon_critic_v13',
            experiment_label=f'hybrid622_26k_H{horizon}_critic{int(enabled)}_seed{seed}_30k',
            coefficient_provenance='new_26k_WM_real_train_starts_gradient_calibration',
            gradient_calibration=str(out/'calibration.json'),
            full_song_from_update=50, evaluation_updates=[0,50,100,200,500,1000,2000,3000,5000,7500,10000,15000,20000,25000,30000])
        a['precision'] = dict(weight=precision_weight, root_xy_weight=root_weight, root_xy_scale=.2,
            soft_reference_weights=soft, soft_reference_scales=scales, root_tracking_bonus_weight=0.,
            execution_jerk_weight=0., stance_foot_weight=0.)
        t.update(seed=seed, adapter_lr=1e-5, adapter_lr_milestones=[100,500,2000,10000], adapter_lr_gamma=.5,
            critic_lr=3e-5, critic_batch=2048)
        v.update(sampling_seed=seed, target_accepted_updates=30000, checkpoint_every=1000)
        new['bptt_steps'] = horizon
        (out/'configs'/f'gpu{gpu}.yaml').write_text('# 新26k基座；三种算法各两个种子，共同训练集标定损失；从零Adapter独立训练。\n'+yaml.safe_dump(new,allow_unicode=True,sort_keys=False))
    (out/'calibration.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(configs=6,adapter_restored=parameter_identity(control.adapter)==report['adapter_sha256'],weights=report['shared_weights'])),flush=True)


if __name__ == '__main__':
    main()
