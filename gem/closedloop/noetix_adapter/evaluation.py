"""对新noetix Adapter和零Adapter做同初态的固定音乐整曲物理对照。

只加载指定底座Actor、Adapter及显式验证动作库，不创建优化器或更新WM/Critic。
每首歌曲占一个独立环境，关闭探索、观测噪声、复位噪声及域随机化；两次运行
重新建立环境以复现相同初态与执行器状态。只执行真实音乐区间，不把名义缓存
额外的未来支持帧计入整歌长度。回报按整首计划长度归一化，失败后的未执行部分
计零；记录物理终止、提前截断与完整音乐结束，短测截断不能算完成。

结果用于固定开发歌曲的模型选择，不声称未见音乐泛化或真机效果。训练入口每
指定间隔单独调用本模块，保存零策略和候选的逐歌证据，不影响训练环境和RNG。
"""

import json
from pathlib import Path

import torch

from .model import ReferenceLibrary, ReferencePolicy
from .data import digest


@torch.no_grad()
def evaluate(checkpoint, library_path, output, max_steps=None):
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.robots import get_robot_spec
    from supermimic.training.checkpoints import atomic_json_save

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    settings = saved['settings']
    packed = torch.load(library_path, map_location='cpu', weights_only=False)
    actor, config = load_policy_from_checkpoint(settings['checkpoint'], 'cuda:0')
    model = ReferencePolicy(actor, get_robot_spec('bumi4340').joint_names).to('cuda:0').eval()
    ec = config['env']
    n = len(packed['names'])
    planned = torch.tensor([row['music_end_frame'] for row in packed['provenance']], device='cuda:0')
    results = {}
    for mode in ('zero_adapter', 'adapter'):
        if mode == 'adapter':
            model.net.load_state_dict(saved['adapter'])
        args = SuperMimicEnvCfgArgs(npz_files=[str(library_path)], robot='bumi4340',
            num_envs=n, compact_library=True, history_length=actor.cfg.history,
            proprio_anchor_ori=actor.cfg.proprio_anchor_ori, future_frames=10,
            episode_length_s=1e5, action_noise_std=0., obs_noise=False, reset_noise=False,
            domain_randomization=False, sampling_mode='start', record_transitions=False,
            seed=42, anchor_obs=ec['anchor_obs'], goal_features=ec['goal_features'],
            goal_root_height=ec['goal_root_height'], self_collision=ec['self_collision'],
            self_collision_weight=ec['self_collision_weight'], self_collision_scale=ec['self_collision_scale'],
            joint_limit_weight=ec['joint_limit_weight'], action_rate_weight=ec['action_rate_weight'],
            survival_weight=ec.get('survival_weight'), joint_velocity_weight=ec.get('joint_velocity_weight'))
        env = make_env(args, device='cuda:0')
        try:
            command = env.command_manager.get_term('motion')
            lib = ReferenceLibrary(command.library, packed['music'])
            command.forced_frame[:] = command.library.clip_start
            obs, _ = env.reset()
            active = torch.ones(n, dtype=torch.bool, device='cuda:0')
            reward_sum = torch.zeros(n, device='cuda:0')
            steps = torch.zeros(n, dtype=torch.long, device='cuda:0')
            failed = torch.zeros_like(active)
            truncated = torch.zeros_like(active)
            terms = list(env.reward_manager.active_terms)
            term_sum = torch.zeros(n, len(terms), device='cuda:0')
            limit = min(int(planned.max()), max_steps) if max_steps else int(planned.max())
            for _ in range(limit):
                goal = torch.cat((lib.goal_window(command.frame, 10), obs['goal'][:, -6:]), -1)
                action = model(obs['policy'], goal)
                obs, reward, term, timeout, _ = env.step(action)
                reward_sum += reward * active
                term_sum += env.reward_manager.term_rewards * active[:, None]
                steps += active.long()
                failed |= term & active
                truncated |= timeout & active & (steps < planned)
                active &= ~(term | timeout | (steps >= planned))
                if not bool(active.any()):
                    break
            rows = []
            for i, name in enumerate(packed['names']):
                count, length = int(steps[i]), int(planned[i])
                task = packed['provenance'][i]['task']
                rows.append(dict(name=name, steps=count, planned_steps=length,
                    adapter_training_seen=task.get('adapter_training_seen'),
                    source_split=task.get('dataset_split', task.get('split')),
                    global_music_group_id=task.get('global_music_group_id'),
                    completed=count >= length and not bool(failed[i] | truncated[i]),
                    physical_failure=bool(failed[i]), truncated=bool(truncated[i] | active[i]),
                    normalized_full_return=float(reward_sum[i]) / (.1 * length),
                    terms={key: float(term_sum[i, j]) / max(count, 1) for j, key in enumerate(terms)}))
            results[mode] = dict(clips=rows, completed=sum(r['completed'] for r in rows),
                failed=sum(r['physical_failure'] for r in rows),
                mean_full_return=sum(r['normalized_full_return'] for r in rows) / n)
        finally:
            env.close()
    protocol = (json.loads(Path(settings['evaluation_protocol']).read_text())
                if settings.get('evaluation_protocol') else {})
    report = dict(adapter_updates=saved['adapter_updates'], checkpoint=str(checkpoint),
        checkpoint_sha256=settings['checkpoint_sha256'], library=str(library_path),
        library_sha256=digest(library_path), requested_songs=protocol.get('expected_songs', n),
        evaluable_songs=n, reference_failures=protocol.get('reference_failures', []),
        selection_seed=protocol.get('selection_seed'),
        songs=n, max_steps=max_steps, scope='fixed_development_music_physical_pair_not_unseen_test', results=results)
    atomic_json_save(report, Path(output))
    print(json.dumps({key: {k: v for k, v in value.items() if k != 'clips'}
                      for key, value in results.items()}), flush=True)
    return report
