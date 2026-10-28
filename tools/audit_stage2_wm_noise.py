"""只读冻结模型的真实动作噪声分布对照，不执行任何优化器更新。

在同一批 64 首 train 歌曲、同初态/历史/物理条件上，比较零 Adapter 与指定已有
Adapter 在 0、0.01、0.03、90% 0.01 加 10% 0.1 噪声下的真实 H32 连续窗口。
噪声实际送入物理求解器，Replay 数据和历史都保存执行动作。原始预训练 WM 与
已有失败 checkpoint 的独立 WM 在完全相同真实动作序列上递归预测 H1/H8/H16/H32，
报告状态/速度误差、归一化漂移和有限幅度参考响应；无噪声 baseline 额外重复一次
报告真实可重复性。真实终止后数据不纳入有效窗口，不跨 reset 拼接。这里不声称
短片段结果是整歌收益，不做拟合，不以 validation/evaluation 歌曲训练任何网络。
输出采用唯一新目录，保存真实窗口供后续有限监督分析，绝不覆盖旧 checkpoint。
"""

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--critic-checkpoint', help='可单独读取原启用 Critic 的失败现场，只读且不进入控制')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=7261)
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
    from supermimic.bridge.bundle import atomic_save
    from supermimic.mdp import unflatten_state
    from supermimic.networks import Critic
    from gem.closedloop.reference_adapter.application import build_engine, read_tasks, seed_all
    from gem.closedloop.reference_adapter.vector_bank import VectorBank
    from gem.closedloop.reference_adapter.vector_control import VectorControl
    from gem.closedloop.reference_adapter.vector_collector import VectorCollector
    from gem.closedloop.reference_adapter.reference_constraints import ReferenceProjection
    from gem.closedloop.reference_adapter.precision import PrecisionObjective
    from gem.closedloop.reference_adapter.vector_entry import json_write, compile_kinematics
    from gem.closedloop.reference_adapter.wm_distribution import evaluate_executed_action_windows, paired_response_noise

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed); torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    engine = build_engine(cfg, 'cuda:0')
    original = copy.deepcopy(engine.dynamics)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    engine.step.adapter.load_state_dict(checkpoint['adapter'])
    engine.dynamics.load_state_dict(checkpoint['dynamics'])
    failed_critic = Critic(2835).cuda().eval()
    critic_checkpoint = (torch.load(args.critic_checkpoint, map_location='cpu', weights_only=False, mmap=True)
                         if args.critic_checkpoint else checkpoint)
    failed_critic.load_state_dict(critic_checkpoint['critic'])
    failed_critic.requires_grad_(False)
    bank = VectorBank(read_tasks(cfg['tasks']), engine, 'cuda:0')
    control = VectorControl(engine, bank)
    control.adapter.disable_root_xy = True
    settings = cfg['alternating']
    control.projection = ReferenceProjection(control.bundle.spec, control.adapter.bounds,
                                            **settings['projection']).cuda()
    calibration = json.loads(Path(settings['calibration']).read_text())
    control.precision = PrecisionObjective(calibration['scales'], **settings['precision'])
    control.soft_joint_limit_cost = False
    control.relative_critic_features = True
    for model in (engine.step.adapter, engine.dynamics, original):
        model.requires_grad_(False)
    frozen = control.bundle.frozen_snapshot()
    compile_kinematics(engine, bank)
    models = dict(original=original, failed_checkpoint=engine.dynamics)
    songs = bank.by_split['train']
    if len(songs) != 64:
        raise ValueError('本对照明确使用现有 64 首 train 歌曲')
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    frames = (torch.rand(64, device='cuda', generator=generator)*(bank.ends[songs]-64).clamp_min(1)).long()
    base_noise = torch.randn(32, 64, 21, device='cuda', generator=generator)
    mixture = torch.full((32, 64, 1), .01, device='cuda')
    mixed_mask = torch.rand((32, 64, 1), device='cuda', generator=generator) < .1
    mixture[mixed_mask] = .1
    noise = dict(zero=base_noise*0, small01=base_noise*.01, small03=base_noise*.03,
                 mixture_std0p01_10percent_std0p1=base_noise*mixture)
    report = dict(config=args.config, checkpoint=args.checkpoint, critic_checkpoint=args.critic_checkpoint or args.checkpoint,
        seed=args.seed, split='train',
        fitting=False, optimizer_updates=0, initial_frames=frames.cpu().tolist(),
        songs=[bank.tasks[int(i)]['song_id'] if hasattr(bank, 'tasks') else int(i) for i in songs],
        scopes='real H32 train distribution diagnostic; not independent full-song evaluation', rows=[])
    json_write(output/'manifest.json', report)
    sim_cfg = make_sim('cuda:0', args)
    start_time = time.time()
    with launch_simulation(sim_cfg, args):
        env = StreamingPhysicalEnv(sim_cfg, 64, engine.step.bundle)
        collector = VectorCollector(env, control, seed=args.seed)
        initial = collector.starting_states(songs, frames, perturb=False)

        @torch.no_grad()
        def collect(label, perturbation, zero):
            seed_all(args.seed); collector.generator.manual_seed(args.seed)
            collector.reset(songs=songs, frames=frames, initial=initial, perturb=False)
            if 'saved_critic_unclipped' not in report:
                values = failed_critic(control.critic_features(collector.state))
                report['saved_critic_unclipped'] = dict(mean=float(values.mean()), max=float(values.max()),
                    min=float(values.min()), fraction_above_proven_upper_10=float((values > 10).float().mean()),
                    norm_count=float(failed_critic.norm.count), real_MC_calibrated=False,
                    evidence='unchanged_saved_Critic_on_fixed_real_reset_states_no_bootstrap_label')
            state, target, actions, physics, mask, rewards, reference_delta, policy_delta = [], [], [], [], [], [], [], []
            active = torch.ones(64, device='cuda', dtype=torch.bool)
            for t in range(32):
                row = collector.step(zero=zero, auto_reset=False, split='train', perturb=False,
                                     action_offset=perturbation[t])
                state.append(row['start']['state']); target.append(row['end']['state'])
                actions.append(row['proposal']['action']); physics.append(row['start']['xi'])
                mask.append(active.clone()); rewards.append(row['reward'])
                reference_delta.append(row['proposal']['residual'].square().mean((1, 2)).sqrt())
                policy_delta.append((row['proposal']['action']-perturbation[t]-row['proposal']['zero_action']).square().mean(-1).sqrt())
                active &= ~row['done']
            window = dict(origin='real', **{key: torch.stack(value, 1) for key, value in
                dict(state=state, next_state=target, action=actions, xi=physics, mask=mask, reward=rewards,
                     residual_rms=reference_delta, policy_action_delta_rms=policy_delta).items()})
            row_report = dict(label=label, zero_adapter=zero, valid_h32=int(window['mask'].all(1).sum()),
                actual_noise_rms=float(perturbation.square().mean().sqrt()),
                mean_policy_action_change=float(window['policy_action_delta_rms'][window['mask']].mean()),
                mean_reference_rms=float(window['residual_rms'][window['mask']].mean()), models={})
            for name, model in models.items():
                row_report['models'][name] = evaluate_executed_action_windows(model, window, kin=control.runtime.kin)
            atomic_save({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in window.items()}, output/f'{label}.pt')
            report['rows'].append(row_report)
            json_write(output/'report.json', report, replace=True)
            print(json.dumps(dict(completed=label, valid_h32=row_report['valid_h32'], elapsed=time.time()-start_time)), flush=True)
            return window

        all_windows = {}
        for label, perturbation in noise.items():
            baseline = collect(label+'_zero_adapter', perturbation, True)
            candidate = collect(label+'_saved_adapter', perturbation, False)
            all_windows[label] = (baseline, candidate)
        repeated = collect('zero_repeat', noise['zero'], True)
        baseline = all_windows['zero'][0]
        valid = baseline['mask'] & repeated['mask']
        report['zero_repeat_max_state_abs_error'] = float((baseline['next_state']-repeated['next_state'])[valid].abs().max())
        report['zero_noise_response_vs_repeat'] = paired_response_noise(baseline, all_windows['zero'][1], repeated)
        report['reference_response'] = {}
        for label, (base, candidate) in all_windows.items():
            directions = dict(interpretation='finite_amplitude_response_not_smooth_contact_derivative',
                initial_state_max_abs_difference=float((base['state'][:, 0]-candidate['state'][:, 0]).abs().max()))
            for name, model in models.items():
                pb, pc = base['state'][:, 0], candidate['state'][:, 0]
                rows = {}
                for t in range(32):
                    pb = model.step(pb, base['action'][:, t], base['xi'][:, t])
                    pc = model.step(pc, candidate['action'][:, t], candidate['xi'][:, t])
                    if t+1 not in (1, 8, 16, 32):
                        continue
                    keep = base['mask'][:, :t+1].all(1)&candidate['mask'][:, :t+1].all(1)
                    if not bool(keep.any()):
                        rows[str(t+1)] = dict(valid_pairs=0); continue
                    real = model.model.state_error(unflatten_state(candidate['next_state'][keep, t]),
                                                  unflatten_state(base['next_state'][keep, t]))
                    predicted = model.model.state_error(unflatten_state(pc[keep]), unflatten_state(pb[keep]))
                    scale = torch.maximum(model.model.out_norm.std(), model.error_floor)
                    rn, pn = real/scale, predicted/scale
                    cosine = (rn*pn).sum(-1)/(rn.norm(dim=-1)*pn.norm(dim=-1)).clamp_min(1e-9)
                    rows[str(t+1)] = dict(valid_pairs=int(keep.sum()), normalized_response_rmse=float((pn-rn).square().mean().sqrt()),
                        normalized_direction_cosine_mean=float(cosine.mean()),
                        actual_action_difference_rms=float((candidate['action'][keep, :t+1]-base['action'][keep, :t+1]).square().mean().sqrt()),
                        real_qd_response_rms=float(real[:, -21:].square().mean().sqrt()),
                        predicted_qd_response_rms=float(predicted[:, -21:].square().mean().sqrt()))
                directions[name] = rows
            report['reference_response'][label] = directions
        report.update(real_transitions=collector.real_transitions, wall_seconds=time.time()-start_time,
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        control.bundle.assert_unchanged(frozen)
        json_write(output/'report.json', report, replace=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
