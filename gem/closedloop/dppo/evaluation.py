"""第十步已加载上层策略的独立、可重算闭环评估。

调用方必须显式传入已经加载目标 checkpoint 的 policy，本模块不重新加载 Stage1
权重、不创建或调用优化器，也不把评估转移放入训练 Buffer。评估计划从完整
FullMusicCatalog 的 val/test manifest 池确定性选取不同配对样本，再与显式 seed 做
笛卡尔积；支持 all。每次从歌曲开头或清单中记录的中心切点运行，默认完整剩余
音乐；若调用者指定有限时长，则明确记录行政截断，不把两秒短对照当作全量评估。

每首歌保存逐上层决策及逐 50Hz 控制区间的实际奖励、奖励分项、原始跟踪/稳定误差、
活动度、延迟、前缀长度、执行失败和参考拒绝。聚合只从这些已落盘 episode 记录
计算，包含全体、来源、seed 和来源×seed 统计。基础设施故障单独记 invalid 并
停止本批评估，不能计为策略失败。详细物理数组和完整去噪链仍由执行 journal 与
原始采样文件保留，评估 JSON 只引用它们，避免再次复制所有大数组。

评估前后核验 Actor 参数及全部 buffers、已有梯度和显式传入的其他冻结网络；
恢复模型模式、Python/NumPy/Torch/CUDA 随机状态及环境的评估设置，不消费训练
音乐采样器。报告只证明这些清单上的执行结果，不声称 val/test 未曾进入预训练。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from gem.closedloop.evaluation_music import load_music_features
from gem.closedloop.frozen_actor import stable_noise_seed

VERSION = 'genmo.closedloop.stage10.evaluation.v1'


def _plain(value):
    if isinstance(value, torch.Tensor):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Evaluation evidence cannot contain NaN or infinity')
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _publish_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(_plain(value), stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        # 硬链接原子发布且拒绝覆盖既有评估证据。
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_evaluation_tasks(catalog, *, split='val', eval_count='all', seeds=(1729, 2718),
                           selection_seed=42, start_mode='beginning', episode_seconds=None):
    """构建可保存的评估计划；eval_count 是不同配对样本数，实际 episode 数还要乘 seed 数。"""
    if split not in {'val', 'test'}:
        raise ValueError('Independent evaluation requires explicit val or test split')
    if start_mode not in {'beginning', 'center'}:
        raise ValueError('Evaluation start_mode must be beginning or center')
    if episode_seconds is not None and (isinstance(episode_seconds, bool) or not math.isfinite(float(episode_seconds)) or float(episode_seconds) <= 0):
        raise ValueError('Evaluation episode_seconds must be positive or None for full music')
    if start_mode == 'center' and episode_seconds is None:
        raise ValueError('Center evaluation requires an explicit finite evaluation window')
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0 for seed in seeds):
        raise ValueError('Evaluation requires distinct nonnegative integer seeds')
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, (int, np.integer)):
        raise ValueError('Evaluation selection_seed must be an integer')
    samples_by_source = catalog.samples[split]
    pool, source_positions = [], {}
    for source in sorted(samples_by_source):
        source_positions[source] = []
        for sample in sorted(samples_by_source[source], key=lambda value: str(value['row']['sample_id'])):
            if sample.get('split', sample['row'].get('split')) != split or sample['row'].get('split') != split or sample['dataset'] != source:
                raise ValueError('Evaluation catalog sample differs from its declared split/source')
            source_positions[source].append(len(pool))
            pool.append(copy.deepcopy(sample))
    if not pool:
        raise ValueError('Evaluation pool is empty')
    identities = [(sample['dataset'], str(sample['row']['sample_id'])) for sample in pool]
    if len(set(identities)) != len(identities):
        raise ValueError('Evaluation pool contains duplicate sample identities')
    if eval_count == 'all':
        indices = list(range(len(pool)))
    else:
        if isinstance(eval_count, bool) or not isinstance(eval_count, int) or not 1 <= eval_count <= len(pool):
            raise ValueError('eval_count must be all or a positive count no greater than the complete pool')
        rng = np.random.default_rng(int(selection_seed))
        nonempty = [positions for positions in source_positions.values() if positions]
        indices = [int(rng.choice(positions)) for positions in nonempty] if eval_count >= len(nonempty) else []
        rest = np.asarray([index for index in range(len(pool)) if index not in set(indices)], dtype=np.int64)
        indices.extend(int(index) for index in rng.choice(rest, size=eval_count-len(indices), replace=False))
        indices.sort()
    tasks = []
    for sample_index in indices:
        sample = pool[sample_index]
        frames = int(sample['row']['num_frames'])
        if frames <= 0 or sample['row'].get('fps') != 30:
            raise ValueError('Evaluation music must contain a positive 30Hz frame count')
        start = 0 if start_mode == 'beginning' else max(0, (frames-int(float(episode_seconds)*30))//2)
        for seed in seeds:
            identity = {'dataset': sample['dataset'], 'sample_id': str(sample['row']['sample_id']),
                        'split': split, 'seed': int(seed), 'music_start_frame': start}
            tasks.append({'task_id': _digest(identity)[:24], **identity, 'sample': copy.deepcopy(sample),
                          'full_music_frames': frames, 'full_music_seconds': frames/30.,
                          'remaining_music_seconds': (frames-start)/30.,
                          'noise_index_start': stable_noise_seed(int(seed), _digest(identity))})
    plan = {'version': VERSION, 'split': split, 'data_root': str(Path(catalog.data_root).resolve()),
            'catalog_identity': _plain(catalog.identity), 'complete_pool_count': len(pool),
            'complete_pool_counts_by_source': {source: len(values) for source, values in samples_by_source.items()},
            'complete_pool_identity_sha256': _digest(pool), 'requested_eval_count': eval_count,
            'selection_unit': 'paired_manifest_sample_not_necessarily_unique_song',
            'complete_unique_group_count': len({sample['group_id'] for sample in pool}),
            'selected_unique_group_count': len({pool[index]['group_id'] for index in indices}),
            'selected_unique_audio_count': len({pool[index]['row']['source_audio_sha256'] for index in indices}),
            'selected_sample_count': len(indices), 'task_count': len(tasks), 'seeds': [int(seed) for seed in seeds],
            'selection_seed': int(selection_seed), 'selection_rule': 'all_or_seeded_without_replacement_with_source_coverage_when_possible',
            'start_mode': start_mode, 'episode_seconds': episode_seconds, 'tasks': tasks,
            'pretraining_unseen_claim': False, 'training_buffer_allowed': False}
    plan['plan_sha256'] = _digest(plan)
    return plan


def _model_fingerprint(module, *, gradients=False):
    digest = hashlib.sha256()
    items = [('gradient', name, parameter.grad) for name, parameter in module.named_parameters()] if gradients else (
        [('parameter', name, parameter) for name, parameter in module.named_parameters()] +
        [('buffer', name, buffer) for name, buffer in module.named_buffers()])
    for kind, name, value in items:
        digest.update((kind + ':' + name).encode())
        if value is None:
            digest.update(b'None')
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


def _capture_rng(*, local_cuda_only=False):
    # torchrun 的每个 rank 只保存自己绑定设备，避免 get_rng_state_all 初始化其他 GPU。
    device = torch.cuda.current_device() if local_cuda_only and torch.cuda.is_initialized() else None
    cuda = None
    if torch.cuda.is_initialized():
        cuda = torch.cuda.get_rng_state(device).clone() if local_cuda_only else torch.cuda.get_rng_state_all()
    return {'python': random.getstate(), 'numpy': copy.deepcopy(np.random.get_state()),
            'torch': torch.get_rng_state().clone(), 'cuda': cuda, 'cuda_device': device,
            'local_cuda_only': bool(local_cuda_only)}


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        if state.get('local_cuda_only', False):
            torch.cuda.set_rng_state(state['cuda'], device=state['cuda_device'])
        else:
            torch.cuda.set_rng_state_all(state['cuda'])


def _field(result, key, default=None):
    return result.get(key, default) if isinstance(result, dict) else getattr(result, key, default)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(float(value)):
        raise ValueError(f'Invalid finite evaluation scalar: {name}')
    return float(value)


def _decision_evidence(result, index):
    if not _field(result, 'transition_valid', True):
        raise ValueError('Invalid execution cannot be treated as a valid evaluation episode')
    metadata = _field(result, 'metadata')
    if not isinstance(metadata, dict):
        raise ValueError('Evaluation transition requires executed reward metadata')
    rewards = _plain(_field(result, 'rewards'))
    count = _field(result, 'executed_control_steps')
    if type(count) is not int or count < 0 or not isinstance(rewards, list) or len(rewards) != count:
        raise ValueError('Evaluation reward count differs from executed controls')
    physics = _field(result, 'executed_physics_steps', count*4)
    if type(physics) is not int or physics != count*4:
        raise ValueError('Evaluation physical substeps do not match actual controls')
    details = metadata.get('reward_details')
    if not isinstance(details, list) or len(details) != count:
        raise ValueError('Evaluation requires one reward detail per actual control interval')
    controls = []
    for reward, detail in zip(rewards, details):
        if detail.get('transition_valid') is not True:
            raise ValueError('Invalid reward detail cannot enter evaluation statistics')
        if detail.get('version') != 'stage9.execution_reward.v2' or detail.get('activity', {}).get('valid') is not True:
            raise ValueError('Evaluation requires current reward version and valid paired activity')
        if not math.isclose(_number(detail['dt_s'], 'dt'), .02, abs_tol=1e-12):
            raise ValueError('Evaluation control intervals must be 50Hz')
        if controls and int(detail['tick']) != controls[-1]['tick'] + 12:
            raise ValueError('Evaluation control rewards are not contiguous')
        components = {}
        for name, item in detail['components'].items():
            components[name] = {key: _plain(item.get(key)) for key in
                                ('score', 'gate', 'activity_gate', 'weight', 'weighted_rate', 'integrated_reward', 'valid')}
            if components[name]['gate'] is None:
                components[name]['gate'] = item.get('activity_gate', 1.)
            if components[name]['integrated_reward'] is None:
                components[name]['integrated_reward'] = _number(item['weighted_rate'], name)*_number(detail['dt_s'], 'dt')
            for key in ('score', 'gate', 'weight', 'weighted_rate', 'integrated_reward'):
                _number(components[name][key], f'{name}.{key}')
        continuous = _number(detail['reward'], 'continuous reward')
        actual = _number(reward, 'actual reward')
        if not math.isclose(continuous, sum(item['integrated_reward'] for item in components.values()), abs_tol=1e-9, rel_tol=1e-9):
            raise ValueError('Evaluation component integral does not match recorded reward')
        raw_errors = {}
        for group in ('track', 'stable'):
            for key, value in detail['components'].get(group, {}).get('raw', {}).items():
                if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                    raw_errors[key] = _number(value, key)
        activity = {key: _plain(detail.get('activity', {}).get(key)) for key in
                    ('valid', 'actual_activity_rad_s', 'target_activity_rad_s', 'gate', 'intensity_score',
                     'window_count', 'window_complete', 'window_begin_tick', 'window_end_tick')}
        controls.append({'tick': int(detail['tick']), 'dt_s': _number(detail['dt_s'], 'dt'),
                         'reward': actual, 'continuous_reward': continuous, 'event_reward': actual-continuous,
                         'components': components, 'raw_errors': raw_errors, 'activity': activity})
    zero_step_event = _number(metadata.get('event_reward', 0.), 'zero step event')
    event_penalty = _number(metadata.get('event_penalty_total', zero_step_event), 'event penalty')
    total = sum(_number(value, 'reward') for value in rewards) + zero_step_event
    if not math.isclose(total, sum(row['continuous_reward'] for row in controls)+event_penalty, abs_tol=1e-8, rel_tol=1e-9):
        raise ValueError('Evaluation event penalties are missing or counted more than once')
    generated = metadata.get('generated', {})
    prefix = generated.get('prefix_frames')
    if prefix is None:
        context = _field(result, 'context', {})
        mask = context.get('known_qpos30_mask') if isinstance(context, dict) else None
        if mask is not None:
            prefix = int(torch.as_tensor(mask).any(-1).sum())
    if type(prefix) is not int or not 0 <= prefix < 120:
        raise ValueError('Evaluation requires actual known prefix frame count')
    terminal_snapshot = metadata.get('terminal_snapshot') or {}
    physical_failure = bool(terminal_snapshot.get('terminated', False))
    reason = _field(result, 'reason')
    terminated, truncated = bool(_field(result, 'terminated')), bool(_field(result, 'truncated'))
    if terminated and reason not in (None, 'music_end', 'task_complete'):
        physical_failure = True
    return {'decision_index': index, 'identity': _plain(_field(result, 'identity', {})),
            'executed_control_steps': count, 'executed_physics_steps': int(_field(result, 'executed_physics_steps', count*4)),
            'reward_sum': total, 'continuous_reward_sum': sum(row['continuous_reward'] for row in controls),
            'event_penalty_total': event_penalty, 'zero_step_event_reward': zero_step_event,
            'terminated': terminated, 'truncated': truncated, 'reason': reason, 'physical_failure': physical_failure,
            'rejection': _plain(metadata.get('rejection')), 'latency_seconds': _number(metadata['latency_seconds'], 'latency'),
            'commit_seconds': _number(metadata.get('commit_seconds', 0.), 'commit latency'), 'prefix_frames': prefix,
            'events': _plain(metadata.get('events', [])), 'raw_sample_path': str(metadata.get('raw_sample_path', '')),
            'controls': controls}


def _stats(values):
    if not values:
        return {'count': 0, 'mean': None, 'min': None, 'max': None, 'p50': None, 'p95': None}
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite evaluation aggregate input')
    return {'count': len(values), 'mean': float(array.mean()), 'min': float(array.min()), 'max': float(array.max()),
            'p50': float(np.quantile(array, .5)), 'p95': float(np.quantile(array, .95))}


def _aggregate_group(episodes):
    valid = [episode for episode in episodes if episode.get('transition_valid') is True]
    decisions = [decision for episode in valid for decision in episode['decisions']]
    controls = [row for decision in decisions for row in decision['controls']]
    components = sorted({name for row in controls for name in row['components']})
    error_names = sorted({name for row in controls for name in row['raw_errors']})
    duration = sum(row['dt_s'] for row in controls)
    total_reward = sum(episode['reward_sum'] for episode in valid)
    rejections = [item for item in decisions if item['rejection']]
    failures = sum(bool(episode['physical_failure']) for episode in valid)
    return {'episode_count': len(episodes), 'valid_episode_count': len(valid),
            'infrastructure_failure_count': len(episodes)-len(valid), 'physical_failure_count': failures,
            'physical_failure_rate': failures/len(valid) if valid else None,
            'music_completion_count': sum(episode['end_reason'] in ('music_end', 'task_complete') for episode in valid),
            'administrative_truncation_count': sum(bool(episode['truncated']) for episode in valid),
            'decision_count': len(decisions), 'executed_control_steps': len(controls),
            'executed_physics_steps': sum(item['executed_physics_steps'] for item in decisions),
            'executed_seconds': duration, 'reward_sum': total_reward,
            'reward_per_executed_second': total_reward/duration if duration else None,
            'episode_reward': _stats([episode['reward_sum'] for episode in valid]),
            'rejection_count': len(rejections), 'rejection_rate_per_decision': len(rejections)/len(decisions) if decisions else None,
            'finite_invalid_reference_count': sum(bool(item['rejection'].get('policy_penalty', False)) for item in rejections),
            'latency_seconds': _stats([item['latency_seconds'] for item in decisions]),
            'prefix_frames': _stats([item['prefix_frames'] for item in decisions]),
            'prefix_over_18_fraction': sum(item['prefix_frames']>18 for item in decisions)/len(decisions) if decisions else None,
            'decision_missed_count': sum(event.get('kind')=='decision_missed' for item in decisions for event in item['events']),
            'components': {name: {'integrated_reward_sum': sum(row['components'][name]['integrated_reward'] for row in controls if name in row['components']),
                                  'score': _stats([row['components'][name]['score'] for row in controls if name in row['components']])} for name in components},
            'raw_errors': {name: _stats([row['raw_errors'][name] for row in controls if name in row['raw_errors']]) for name in error_names},
            'activity': {name: _stats([row['activity'][name] for row in controls if row['activity'].get('valid') and row['activity'].get(name) is not None])
                         for name in ('actual_activity_rad_s', 'target_activity_rad_s', 'gate', 'intensity_score')}}


def aggregate_evaluation(episodes):
    """只从逐 episode 证据重算所有统计；可由独立 CPU 审计器调用。"""
    episodes = list(episodes)
    sources = sorted({episode['dataset'] for episode in episodes})
    seeds = sorted({episode['seed'] for episode in episodes})
    return {'overall': _aggregate_group(episodes),
            'by_source': {source: _aggregate_group([episode for episode in episodes if episode['dataset']==source]) for source in sources},
            'by_seed': {str(seed): _aggregate_group([episode for episode in episodes if episode['seed']==seed]) for seed in seeds},
            'by_source_seed': {source: {str(seed): _aggregate_group([episode for episode in episodes if episode['dataset']==source and episode['seed']==seed])
                                        for seed in seeds} for source in sources}}


def evaluate_policy(env, policy, task_plan, output_dir, *, catalog=None, episode_seconds=None,
                    deterministic=False, actor_identity=None, frozen_modules=None, progress=None,
                    local_cuda_only=False):
    """只评估显式 policy，返回可重算报告；故障时先写报告再抛错，绝不开始优化。"""
    if env.policy is not policy:
        raise ValueError('Evaluation environment must use the explicitly supplied loaded Actor policy')
    plan = copy.deepcopy(task_plan)
    if plan.get('version') != VERSION or plan.get('split') not in ('val', 'test') or plan.get('training_buffer_allowed') is not False:
        raise ValueError('Invalid independent evaluation task plan')
    declared = plan.pop('plan_sha256', None)
    if declared != _digest(plan):
        raise ValueError('Evaluation task plan identity mismatch')
    plan['plan_sha256'] = declared
    if catalog is not None and _plain(catalog.identity) != plan['catalog_identity']:
        raise ValueError('Evaluation catalog differs from planned manifest identity')
    limit = plan.get('episode_seconds') if episode_seconds is None else episode_seconds
    if limit is not None and (isinstance(limit, bool) or not math.isfinite(float(limit)) or float(limit) <= 0):
        raise ValueError('Evaluation episode_seconds must be positive or None')
    if plan['start_mode'] == 'center' and limit != plan['episode_seconds']:
        raise ValueError('Center evaluation window cannot change after start frames were fixed')
    output = Path(output_dir)
    _publish_json(output/'selection.json', plan)
    modules = {'actor': policy.actor, **(frozen_modules or {})}
    if modules['actor'] is not policy.actor:
        raise ValueError('Frozen module mapping cannot replace the evaluated Actor')
    before = {name: _model_fingerprint(module) for name, module in modules.items()}
    gradient_before = _model_fingerprint(policy.actor, gradients=True)
    modes = {name: {child_name: child.training for child_name, child in module.named_modules()} for name, module in modules.items()}
    rng = _capture_rng(local_cuda_only=local_cuda_only)
    original_limit = env.config['stage9']['episode_seconds']
    original_noise_index = env.comparison_noise_index
    episodes, manifests = [], []
    error = None
    try:
        for module in modules.values():
            module.eval()
        for index, task in enumerate(plan['tasks']):
            if task['sample']['row'].get('split') != plan['split']:
                raise ValueError('Evaluation task contains a training or different split sample')
            if progress:
                progress({'event': 'evaluation_episode_start', 'episode_index': index, 'task_id': task['task_id'],
                          'dataset': task['dataset'], 'seed': task['seed']})
            episode = {'version': VERSION, 'task_id': task['task_id'], 'episode_index': index,
                       'dataset': task['dataset'], 'sample_id': task['sample_id'], 'split': plan['split'], 'seed': task['seed'],
                       'music_start_frame': task['music_start_frame'], 'full_music_seconds': task['full_music_seconds'],
                       'remaining_music_seconds': task['remaining_music_seconds'], 'episode_seconds_limit': limit,
                       'noise_index_start': task['noise_index_start'], 'decisions': [], 'transition_valid': False,
                       'physical_failure': False, 'truncated': False, 'end_reason': None, 'reward_sum': 0.}
            try:
                music = catalog.load_music(task['sample']) if catalog is not None else load_music_features(plan['data_root'], task['sample'])
                env.config['stage9']['episode_seconds'] = float(task['remaining_music_seconds']) if limit is None else float(limit)
                env.comparison_noise_index = int(task['noise_index_start'])
                with torch.no_grad():
                    env.reset_task(task['sample'], music, seed=task['seed'], phase='evaluation', music_start_frame=task['music_start_frame'])
                    maximum_decisions = math.ceil(min(task['remaining_music_seconds'], float(limit) if limit is not None else task['remaining_music_seconds'])/.02)+2
                    for decision_index in range(maximum_decisions):
                        result = env.step(deterministic=deterministic)
                        evidence = _decision_evidence(result, decision_index)
                        episode['decisions'].append(evidence)
                        episode['reward_sum'] += evidence['reward_sum']
                        if evidence['terminated'] or evidence['truncated']:
                            episode.update(transition_valid=True, physical_failure=evidence['physical_failure'],
                                           truncated=evidence['truncated'], end_reason=evidence['reason'])
                            break
                        if evidence['executed_control_steps'] <= 0:
                            raise RuntimeError('Evaluation made no progress without a terminal boundary')
                    else:
                        raise RuntimeError('Evaluation exceeded finite decision guard without termination')
            except Exception as caught:
                episode.update(transition_valid=False, error={'type': type(caught).__name__, 'message': str(caught)},
                               end_reason='infrastructure_failure')
                error = caught
            episode_path = output/'episodes'/f'{index:06d}.json'
            _publish_json(episode_path, episode)
            episodes.append(episode)
            manifests.append({'path': str(episode_path.relative_to(output)), 'sha256': hashlib.sha256(episode_path.read_bytes()).hexdigest(),
                              'task_id': task['task_id'], 'transition_valid': episode['transition_valid']})
            if progress:
                progress({'event': 'evaluation_episode_end', 'episode_index': index, 'task_id': task['task_id'],
                          'transition_valid': episode['transition_valid'], 'reward_sum': episode['reward_sum']})
            if error is not None:
                break
    except Exception as caught:
        error = caught
    finally:
        after = {name: _model_fingerprint(module) for name, module in modules.items()}
        gradient_after = _model_fingerprint(policy.actor, gradients=True)
        env.config['stage9']['episode_seconds'] = original_limit
        env.comparison_noise_index = original_noise_index
        for name, module in modules.items():
            for child_name, child in module.named_modules():
                child.training = modes[name][child_name]
        _restore_rng(rng)
    unchanged = {name: before[name] == after[name] for name in modules}
    if not all(unchanged.values()) or gradient_before != gradient_after:
        error = RuntimeError('Evaluation changed model parameters/buffers or Actor gradients')
    report = {'version': VERSION, 'status': 'failed' if error is not None else 'passed',
              'actor_identity': _plain(actor_identity or {}), 'plan_sha256': declared, 'selection_path': 'selection.json',
              'rng_scope': 'current_cuda_device' if local_cuda_only else 'all_initialized_cuda_devices',
              'split': plan['split'], 'requested_task_count': len(plan['tasks']), 'completed_episode_count': len(episodes),
              'episode_manifests': manifests, 'aggregate': aggregate_evaluation(episodes),
              'network_fingerprints_before': before, 'network_fingerprints_after': after, 'networks_unchanged': unchanged,
              'actor_gradients_unchanged': gradient_before == gradient_after, 'rng_restored': True,
              'updates_performed': False, 'training_buffer_used': False, 'deterministic': bool(deterministic),
              'episode_seconds_limit': limit, 'pretraining_unseen_claim': False,
              'execution_evidence': 'separate acknowledged execution journal and raw sample paths',
              'error': None if error is None else {'type': type(error).__name__, 'message': str(error)}}
    _publish_json(output/'report.json', report)
    if error is not None:
        error.evaluation_report = report
        raise error
    return report
