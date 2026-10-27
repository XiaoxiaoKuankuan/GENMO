"""使用真实困难复位状态独立审计已训练Critic，不改Adapter、Critic或任何优化器。

读取指定完整checkpoint，恢复其实际XY屏蔽、参考约束、价值归一化与概率策略，
在训练/验证各歌曲20%到80%的合法50Hz时刻复位。温和及较难的关节、姿态、速度
扰动只作用于真实仿真初态，失败由原SM终止实测决定；绝不制造失败标签。
回报完整采到真实终止，单独保留每个复位片段的起点、长度、失败和MC，不称完整歌曲。
训练/验证分别保存，验证数据不参与拟合；这个审计不给已完成的策略倒算训练收益。
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    preliminary, _ = parser.parse_known_args()
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(OmegaConf.load(preliminary.config).supermimic.repo))
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    import torch
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from supermimic.mdp import unflatten_state, flatten_state
    from supermimic.rotations import so3_exp
    from gem.closedloop.reference_adapter.application import read_config, build_engine, read_tasks, seed_all
    from gem.closedloop.reference_adapter.vector_bank import VectorBank
    from gem.closedloop.reference_adapter.vector_control import VectorControl
    from gem.closedloop.reference_adapter.vector_collector import VectorCollector
    from gem.closedloop.reference_adapter.vector_learning import GroupNormalizedCritic, validate_critic
    from gem.closedloop.reference_adapter.real_returns import complete_real_mc, dataset_to, coverage
    from gem.closedloop.reference_adapter.vector_entry import json_write, compile_kinematics
    from gem.closedloop.reference_adapter.update_diagnostics import parameter_identity
    cfg = read_config(args.config)
    seed_all(20261027); torch.set_num_threads(1)
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    ck = torch.load(args.checkpoint, weights_only=False, map_location='cpu', mmap=True)
    engine = build_engine(cfg, 'cuda:0')
    for k, v in engine.step.bundle.state_dict().items():
        if not torch.equal(v.cpu(), ck['frozen_bundle'][k]):
            raise ValueError('独立审计冻结资产不匹配')
    engine.step.adapter.load_state_dict(ck['adapter'])
    tasks = read_tasks(cfg['tasks']); bank = VectorBank(tasks, engine, 'cuda:0')
    control = VectorControl(engine, bank)
    repair = ck['config']['repair']
    control.adapter.disable_root_xy = repair['root_xy_disabled']
    control.constraint_beta = 1. if repair['variant'] == 'A' else repair['constraint_beta']
    control.root_cost_weight = repair['root_position_cost']; control.velocity_cost_weight = repair['reference_velocity_cost']
    control.relative_critic_features = True; control.soft_imagination_continuation = False
    critic = GroupNormalizedCritic(2835, control.spec).cuda()
    critic.load_state_dict(ck['critic']); critic.requires_grad_(False)
    adapter_id, critic_id = parameter_identity(control.adapter), parameter_identity(critic)
    json_write(out/'manifest.json', dict(checkpoint=str(Path(args.checkpoint).resolve()), accepted_updates=ck['state']['accepted_updates'],
        adapter_sha256=adapter_id, critic_sha256=critic_id, optimizer_updates=0, seed=20261027,
        joint_perturbation_rad=.30, orientation_rotvec_component_rad=.45, joint_velocity_perturbation=1.5,
        root_velocity_perturbation=.75, start_fractions=[.2+.6*i/7 for i in range(8)],
        labels='actual_physics_until_true_terminal_no_synthetic_failure', datasets_separate=True))
    compile_kinematics(engine, bank)

    class DifficultCollector(VectorCollector):
        def starting_states(self, songs, frames, *, perturb=False):
            x = super().starting_states(songs, frames, perturb=False)
            if not perturb:
                return x
            p, R, v, w, q, qd = unflatten_state(x)
            noise = lambda width: 2*self.random((len(x), width))-1
            q = (q+.30*noise(21)).clamp(self.control.runtime.kin.q_lo, self.control.runtime.kin.q_hi)
            R = R@so3_exp(.45*noise(3))
            return flatten_state((p, R, v+.75*noise(3), w, q, qd+1.5*noise(21)))

    sim_cfg = make_sim('cuda:0', args)
    datasets, episodes, reports = {}, {}, {}
    with launch_simulation(sim_cfg, args), torch.no_grad():
        env = StreamingPhysicalEnv(sim_cfg, 64, engine.step.bundle)
        collector = DifficultCollector(env, control, seed=20261027)
        for split in ('train', 'validation'):
            selected = bank.by_split[split].repeat(8)
            fractions = torch.linspace(.2, .8, 8, device=bank.device).repeat_interleave(len(bank.by_split[split]))
            frames = (fractions*bank.ends[selected]).long()
            data, meta = complete_real_mc(collector, split=split, version=ck['state']['accepted_updates'],
                stride=4, repetitions=8, start_frames=frames, latent_std=repair['latent_std'] if repair['variant'] == 'C' else 0.)
            datasets[split], episodes[split] = data, meta
            json_write(out/f'coverage_{split}.json', coverage(data, meta, tasks))
            reports[split] = validate_critic(critic, dataset_to(data, 'cuda:0'), datasets['train']['y'].mean(),
                cfg['training']['critic_quality'], len(datasets['train']['y']), tasks)
            json_write(out/f'calibration_{split}.json', reports[split])
        assert parameter_identity(control.adapter) == adapter_id and parameter_identity(critic) == critic_id
        torch.save(dict(datasets=datasets, episodes=episodes, origin='independent_real_reset_segments',
            checkpoint=args.checkpoint, real_transitions=collector.real_transitions), out/'real_returns.pt')
        json_write(out/'result.json', dict(reports=reports, real_transitions=collector.real_transitions,
            actor_critic_unchanged=True, optimizer_updates=0, status='audit_complete_not_training'))


if __name__ == '__main__':
    main()
