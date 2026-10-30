"""为四组双种子正式对照准备固定训练标尺、真实代理检查和八份配置。

读取既有64首train的零Adapter真实轨迹，固定执行jerk的P75标尺；读取既有真实
Replay导出的128个因果起点，不加载旧Adapter或拟合过的WM。用64个独立物理环境
检查零Adapter及微小非零Adapter下，名义支撑窗口与Newton真实接触的对应关系。
代理没有足够接触精确率时只关闭可选足端训练项，正式jerk/根位置对照仍然生成；
不放宽物理约束，不将代理检查当成WM方向证书，也不启动Adapter优化。

随后用原预训练WM、零Adapter、固定train起点的H32完整BPTT标定小权重：执行
jerk和足端项各自参数梯度范数初始为原SM Tracking的10%。只读梯度分析不执行
Adam。四组为原根奖励/增加根奖励 × 执行稳定性关闭/开启，各用seed42和43；
其它超参数相同，名义缓存和SM六项报告合同相同，评价下降不触发停训。

所有输出写入明确的新准备目录，不能覆盖旧目录。标定、真实物理检查的转移数
与正式训练更新分开记录。最终evaluation歌曲不用于权重标定或代理可靠性选择。
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--starts', required=True)
    parser.add_argument('--zero-trajectories', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--physical-steps', type=int, default=128)
    preliminary, _ = parser.parse_known_args()
    import yaml
    cfg = yaml.safe_load(Path(preliminary.config).read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, cfg['supermimic']['repo'])
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    import numpy as np
    import torch
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from supermimic.networks import Critic
    from gem.closedloop.reference_adapter.application import build_engine, read_tasks, seed_all, direct_reference_config
    from gem.closedloop.reference_adapter.vector_bank import VectorBank
    from gem.closedloop.reference_adapter.vector_control import VectorControl, detach_state, select_state
    from gem.closedloop.reference_adapter.vector_collector import VectorCollector
    from gem.closedloop.reference_adapter.precision import PrecisionObjective
    from gem.closedloop.reference_adapter.alternating_learning import imagine
    from gem.closedloop.reference_adapter.dynamics import frozen_parameters
    from gem.closedloop.reference_adapter.update_diagnostics import parameter_identity
    from gem.closedloop.reference_adapter.vector_entry import compile_kinematics

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'configs').mkdir()
    seed_all(42); torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
    cfg = direct_reference_config(cfg)
    engine = build_engine(cfg, 'cuda:0')
    bank = VectorBank(read_tasks(cfg['tasks']), engine, 'cuda:0')
    control = VectorControl(engine, bank)
    control.adapter.disable_root_xy = True
    control.soft_joint_limit_cost = False
    control.sm_soft_termination_only = True
    original_adapter = copy.deepcopy(control.adapter.state_dict())
    frozen = control.bundle.frozen_snapshot()
    scales = json.loads(Path(cfg['alternating']['calibration']).read_text())['scales']
    precision = copy.deepcopy(cfg['alternating']['precision'])
    jerk_samples, trajectory_songs = [], []
    for path in sorted(Path(args.zero_trajectories).glob('*.npz')):
        manifest = json.loads(path.with_suffix('.json').read_text())
        task = manifest['task']
        if task['split'] != 'train' or not manifest['zero_adapter']:
            raise ValueError('jerk标尺只能读取train零Adapter真实轨迹')
        with np.load(path) as trajectory:
            qd = trajectory['state'][:, 34:]
            jerk_samples.append(np.abs(np.diff(qd, n=2, axis=0)/.02**2).reshape(-1))
        trajectory_songs.append(task['id'])
    if len(trajectory_songs) != 64 or len(set(trajectory_songs)) != 64:
        raise ValueError('需要完整且不重复的64首train轨迹')
    jerk_scale = max(100., float(np.quantile(np.concatenate(jerk_samples), .75)))
    del jerk_samples
    precision.update(root_tracking_bonus_weight=0., execution_jerk_weight=1., execution_jerk_scale=jerk_scale,
                     stance_foot_weight=1., stance_foot_scale=.1)
    control.precision = PrecisionObjective(scales, **precision)
    control.precision.prepare_stance(control)
    saved = torch.load(args.starts, map_location='cpu', weights_only=False)
    starts = {k:v.cuda() for k,v in saved['starts'].items()}
    n = control.runtime.mdp.num_joints
    # 原真实历史未加观测噪声，按权威MDP尺度还原上一个实际速度，绝不读未来标签。
    starts['execution_previous_qd'] = starts['history'][:, -2, n:2*n]/control.runtime.mdp.obs_scale[n:2*n]
    starts['execution_jerk_valid'] = starts['frame'] > starts['start_frame']
    if not bool(torch.isin(starts['song'], bank.by_split['train']).all()):
        raise ValueError('真实起点混入非train歌曲')
    torch.save(dict(starts={k:v.cpu() for k,v in starts.items()}, source=args.starts,
                    boundary='real_train_causal_history_no_future_state'), out/'fixed_train_starts.pt')
    compile_kinematics(engine, bank)
    metadata = dict(created_unix=time.time(), source_starts=args.starts, source_trajectories=args.zero_trajectories,
        training_songs=trajectory_songs, jerk_scale=jerk_scale, foot_scale_mps=.1,
        adapter_initial=parameter_identity(control.adapter), wm_initial=parameter_identity(engine.dynamics),
        physical_num_envs=64, physical_steps=args.physical_steps, physical_checks={},
        gradient_ratio_targets=dict(execution_jerk=.1, stance_foot_velocity=.1),
        boundary='calibration_and_proxy_check_not_formal_updates_or_WM_direction_certificate')
    selected = torch.tensor([(starts['song'] == song).nonzero()[0,0]
                             for song in bank.by_split['train']], device=bank.device)
    physical_starts = select_state(starts, selected)
    sim_cfg = make_sim('cuda:0', args)
    with launch_simulation(sim_cfg, args):
        env = StreamingPhysicalEnv(sim_cfg, 64, control.bundle)
        collector = VectorCollector(env, control, seed=42)
        for mode in ('zero', 'small_nonzero'):
            control.adapter.load_state_dict(original_adapter)
            if mode == 'small_nonzero':
                with torch.no_grad():
                    control.adapter.net[-1].bias.copy_(torch.randn(control.adapter.net[-1].bias.shape,
                        generator=torch.Generator().manual_seed(881)).cuda()*.02)
            collector.reset(songs=physical_starts['song'], frames=physical_starts['frame'],
                initial=physical_starts['state'], joint_bias=physical_starts['xi'][:,1:22], perturb=False)
            for key in ('action','history','heading','left','anchor','plan','plan_valid','music',
                        'execution_previous_qd','execution_jerk_valid'):
                collector.state[key] = physical_starts[key].clone()
            env.a_prev[:] = collector.state['action']
            counters = torch.zeros(6,device=bank.device)
            for step in range(args.physical_steps):
                row = collector.step(zero=mode=='zero', split='train', perturb=False,contact_count=64)
                if step < 5:
                    continue
                actual = row['contacts'][1]
                predicted = row['raw']['stance_weight'] > 0
                gate = row['raw']['stance_weight']
                counters += torch.stack(((predicted & actual).sum(),(predicted & ~actual).sum(),
                    (~predicted & actual).sum(),(~predicted & ~actual).sum(),
                    (gate*actual).sum(),gate.sum())).float()
            tp,fp,fn,tn,weighted_true,mass = counters.cpu().tolist()
            metadata['physical_checks'][mode] = dict(tp=tp,fp=fp,fn=fn,tn=tn,
                precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
                weighted_precision=weighted_true/max(mass,1e-8),
                actual_transitions=64*args.physical_steps)
            print(json.dumps(dict(physical_proxy=mode,metrics=metadata['physical_checks'][mode])),flush=True)
        metadata['calibration_physical_transitions'] = collector.real_transitions
    control.adapter.load_state_dict(original_adapter)
    metadata['foot_proxy_usable'] = all(x['precision'] >= .85 and x['recall'] >= .02
                                       for x in metadata['physical_checks'].values())
    critic = Critic(2835).cuda().requires_grad_(False)
    parameters = tuple(control.adapter.parameters())
    names = ('sm_tracking','execution_jerk','stance_foot_velocity')
    gradients = {name:[torch.zeros_like(p) for p in parameters] for name in names}
    with engine.dynamics.adapter_phase(), frozen_parameters(critic):
        for offset in range(0,len(starts['state']),32):
            z = detach_state(select_state(starts,slice(offset,offset+32)))
            result = imagine(control,engine.dynamics,critic,z,32,terminal_value=False,gamma=.99)
            terms = dict(sm_tracking=result['tracking'],**{k:result['terms'][k] for k in names[1:]})
            for i,name in enumerate(names):
                values = torch.autograd.grad(terms[name].sum()/len(starts['state']),parameters,
                    retain_graph=i<len(names)-1,allow_unused=True)
                for total,value in zip(gradients[name],values):
                    if value is not None: total.add_(value.detach())
    norms = {k:float(sum(x.square().sum() for x in v).sqrt()) for k,v in gradients.items()}
    if not all(np.isfinite(v) and v>1e-10 for v in norms.values()):
        raise FloatingPointError(f'目标梯度非有限或不可辨识：{norms}')
    jerk_weight = .1*norms['sm_tracking']/norms['execution_jerk']
    foot_weight = .1*norms['sm_tracking']/norms['stance_foot_velocity'] if metadata['foot_proxy_usable'] else 0.
    metadata.update(gradient_norms=norms,execution_jerk_weight=jerk_weight,stance_foot_weight=foot_weight,
        foot_status='independent_train_physics_proxy_checked' if foot_weight else
                    'disabled_proxy_precision_insufficient_real_contact_slip_still_evaluated')
    for gpu in range(8):
        group,seed = gpu//2,42+gpu%2
        config = copy.deepcopy(cfg)
        config['bptt_steps']=32
        config['training']['seed']=seed
        config['vector'].update(sampling_seed=seed,target_accepted_updates=10000)
        s=config['alternating']
        s.update(critic_enabled=False,action_noise=0.,exploration_fraction=0.,
            stage_budget='10000_Adam_updates_metrics_never_stop_training',
            experiment_label=f'H32_rootbonus{int(group in (1,3))}_execution{int(group in (2,3))}_seed{seed}',
            evaluation_updates=[0,5,10,20,50,100,200,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000],
            execution_stability_calibration=str(out/'calibration.json'))
        s['precision'].update(root_tracking_bonus_weight=1.2 if group in (1,3) else 0.,
            execution_jerk_weight=jerk_weight if group in (2,3) else 0.,execution_jerk_scale=jerk_scale,
            stance_foot_weight=foot_weight if group in (2,3) else 0.,stance_foot_scale=.1)
        config=direct_reference_config(config)
        (out/f'configs/gpu{gpu}.yaml').write_text('# 四组双种子正式对照；网络和参考合同相同，评价不早停。\n'+
            yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
    control.bundle.assert_unchanged(frozen)
    (out/'calibration.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print(json.dumps(metadata,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
