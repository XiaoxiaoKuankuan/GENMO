"""用已有连续 Newton sidecar 只读核验冻结世界模型，不启动仿真或 GENMO。

数据流：核验原 train 来源清单、三类索引合同和不可变分片 SHA，读取真实 state55、
实际执行 action21 与固定 xi25；一步预测覆盖所有有下一真实状态的转移，多步预测
从每首轨迹均匀选取相同锚点，连续使用缓存中的实际动作，比较 H1/8/16/32 末端误差。
末步缺少下一状态的动作明确排除，reset 前后绝不拼接；真实终止前的困难轨迹保留。

仅调用原 SuperMimic bridge 和原误差/旋转算法，FP32/no_grad，SM/WM/normalizer
全部冻结。默认 CPU、有界线程和批量，独立输出目录；不修改训练配置、缓存、权重、
训练随机数或旧认证范围。输出逐来源和逐锚点证据，以及按来源、运动类型汇总的误差。
这是记录动作下的动力学预测精度，不代表学生闭环控制、梯度方向或完整舞蹈质量通过。
"""

from collections import Counter, defaultdict
import copy
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from .contracts import SIDECAR_SCHEMA, digest, source_key


def paired_transitions(episode):
    """严格识别 s[t], a[t], s[t+1]，用 previous_action 交叉核对真实时间对应。"""
    state, action = episode['state'], episode['action']
    if episode['reset_count'] != 1 or episode['state_origin'] != 'continuous_newton_observation_not_data_reference':
        raise ValueError('仅接受单次reset的连续真实Newton轨迹')
    if episode['frame_origin_tick'] != 0:
        raise ValueError('当前离线合同必须从明确的episode时间零点开始')
    if state.ndim != 2 or state.shape[1] != 55 or action.ndim != 2 or action.shape[1] != 21:
        raise ValueError('真实状态/动作维度不是state55/action21')
    if len(state) not in (len(action), len(action) + 1):
        raise ValueError('state/action长度不满足连续采集合同')
    if episode['control_steps'] != len(action) or tuple(episode['xi'].shape) != (25,):
        raise ValueError('动作数量或xi合同不一致')
    if any(not torch.isfinite(x).all() for x in (state, action, episode['xi'])):
        raise ValueError('真实缓存含非有限张量，不能删除困难样本后继续报告')
    n = min(len(action), max(0, len(state) - 1))
    if episode['previous_action'].shape != (len(state), 21):
        raise ValueError('previous_action没有逐真实状态保存')
    if not torch.equal(episode['previous_action'][1:n+1], action[:n]):
        raise ValueError('previous_action不匹配前一步实际动作，禁止错位评价')
    return n


def matched_starts(pair_count, horizons, anchors):
    """各H采用同一首内相同起点；短轨迹只报告确有真实终点的较短H。"""
    available = [h for h in horizons if h <= pair_count]
    if not available:
        return [], []
    maximum = max(available)
    count = min(anchors, pair_count - maximum + 1)
    starts = np.unique(np.linspace(0, pair_count - maximum, count).round().astype(int)).tolist()
    return starts, available


def error_rows(predicted, truth):
    from gem.closedloop.mb_control.audit_metrics import component_errors
    components = component_errors(predicted, truth, None)
    result = {k: v.double().square().mean(-1) for k, v in components.items()}
    # 角度RMSE之外额外给最差关节误差，避免21个关节平均掩盖单个关节的大偏差。
    result['max_joint_position_rad'] = components['joint_position_rad'].double().abs().amax(-1).square()
    return result


def pack_errors(values, count):
    good = torch.ones(count, dtype=torch.bool)
    for value in values.values():
        good &= torch.isfinite(value.cpu())
    stats = {}
    for key, value in values.items():
        value = value.cpu().double()[good]
        stats[key] = ({'mse_sum': float(value.sum()), 'rmse': float(value.mean().sqrt()),
                       'p95_point_error': float(torch.quantile(value.sqrt(), .95)),
                       'max_point_error': float(value.sqrt().max())} if len(value) else None)
    return {'count': count, 'finite_count': int(good.sum()), 'nonfinite_count': int((~good).sum()),
            'metrics': stats}


@torch.no_grad()
def evaluate_episode(episode, bundle, *, batch_size, anchors, horizons):
    """真实动作逐步推进WM；每个锚点只在起点读取一次真实状态，不中途回填GT。"""
    from .requests import motion_tags
    n = paired_transitions(episode)
    device = next(bundle.world_model.parameters()).device
    state, action, xi = episode['state'], episode['action'], episode['xi']
    all_errors = defaultdict(list)
    for begin in range(0, n, batch_size):
        end = min(n, begin + batch_size)
        prediction = bundle.step_model(state[begin:end].to(device), action[begin:end].to(device),
                                       xi.to(device)[None].expand(end-begin, -1))
        for key, value in error_rows(prediction, state[begin+1:end+1].to(device)).items():
            all_errors[key].append(value.cpu())
    scores = {}
    if n:
        scores['one_step_all'] = pack_errors({k: torch.cat(v) for k, v in all_errors.items()}, n)
    starts, available = matched_starts(n, horizons, anchors)
    anchor_rows = []
    if starts:
        ix = torch.tensor(starts)
        prediction = state[ix].to(device)
        actions = torch.stack([action[s:s+max(available)] for s in starts]).to(device)
        truth = torch.stack([state[s+1:s+max(available)+1] for s in starts]).to(device)
        xi_batch = xi.to(device)[None].expand(len(starts), -1)
        alive = torch.ones(len(starts), dtype=torch.bool, device=device)
        tags = [motion_tags(state[s:s+max(available)+1], bundle.spec.joint_names,
                            episode['contacts'][s:s+max(available)]) for s in starts]
        for h in range(1, max(available) + 1):
            # 非有限轨迹继续记失败，但不让它参与后续前向影响其它轨迹的数值核验。
            next_prediction = torch.full_like(prediction, float('nan'))
            if alive.any():
                next_prediction[alive] = bundle.step_model(prediction[alive], actions[alive, h-1], xi_batch[alive])
            prediction = next_prediction
            alive &= torch.isfinite(prediction).all(-1)
            if h not in available:
                continue
            errors = error_rows(prediction, truth[:, h-1])
            scores[f'rollout_endpoint_h{h}'] = pack_errors(errors, len(starts))
            errors = {k: v.cpu().tolist() for k, v in errors.items()}
            for index, start in enumerate(starts):
                finite = all(np.isfinite(v[index]) for v in errors.values())
                anchor_rows.append({'start_control_step': start, 'start_tick': start*12,
                    'horizon': h, 'seconds': h*.02, 'motion_tags': tags[index], 'finite': finite,
                    'squared_errors': {k: v[index] if np.isfinite(v[index]) else None for k, v in errors.items()}})
    return {'source': episode['source_key'], 'dataset': episode['dataset_id'],
            'termination_reason': episode['termination_reason'], 'stored_states': len(state),
            'stored_actions': len(action), 'paired_transitions': n,
            'unpaired_final_actions': len(action)-n, 'horizons_available': available,
            'scores': scores, 'anchors': anchor_rows}


def aggregate(records):
    """同时报告所有时刻加权RMSE与各来源等权均值，保留非有限失败分母。"""
    groups = defaultdict(lambda: defaultdict(list))
    tags = defaultdict(lambda: defaultdict(list))
    for record in records:
        for group in ('all', 'dataset/'+record['dataset']):
            for name, score in record['scores'].items():
                groups[group][name].append(score)
        for row in record['anchors']:
            for tag in row['motion_tags']:
                tags[tag][str(row['horizon'])].append(row)
    result = {}
    for group, modes in groups.items():
        result[group] = {}
        for mode, scores in modes.items():
            finite_count = sum(s['finite_count'] for s in scores)
            metrics = {}
            for name in scores[0]['metrics']:
                values = [s['metrics'][name] for s in scores if s['metrics'][name] is not None]
                metrics[name] = ({'pooled_rmse': (sum(v['mse_sum'] for v in values)/finite_count)**.5,
                    'source_mean_rmse': float(np.mean([v['rmse'] for v in values])),
                    'source_p95_rmse': float(np.quantile([v['rmse'] for v in values], .95)),
                    'maximum_point_error': max(v['max_point_error'] for v in values)} if values else None)
            result[group][mode] = {'sources': len(scores), 'count': sum(s['count'] for s in scores),
                'nonfinite_count': sum(s['nonfinite_count'] for s in scores), 'metrics': metrics}
    motion = {}
    for tag, modes in tags.items():
        motion[tag] = {}
        for h, rows in modes.items():
            valid = [r for r in rows if r['finite']]
            motion[tag][h] = {'count': len(rows), 'nonfinite_count': len(rows)-len(valid),
                'rmse': {k: float(np.mean([r['squared_errors'][k] for r in valid])**.5)
                         for k in valid[0]['squared_errors']} if valid else {}}
    return {'groups': result, 'motion_tags_overlapping': motion}


def source_inventory(cfg, bundle):
    """核对完整train清单与sidecar来源，不从source存在推断整曲成功或H32合法。"""
    from .data import SourceCatalog
    catalog_cfg = copy.deepcopy(cfg)
    catalog_cfg['data']['loading'] = {}
    catalog = SourceCatalog(catalog_cfg, 'train')
    expected = {source_key({'dataset_id': d.dataset_name, 'split': 'train',
                            'sample_id': str(d.rows[i]['sample_id'])}) for d, i in catalog.sources()}
    directories = cfg['data']['sidecar']
    directories = [directories] if isinstance(directories, str) else directories
    sources, shards, indices = {}, {}, []
    assets = None
    for directory in directories:
        directory = Path(directory).resolve()
        index_path = directory/'index.json'
        index = json.loads(index_path.read_text())
        if index['schema'] != SIDECAR_SCHEMA or index['split'] != 'train':
            raise ValueError('必须使用现有v2 train sidecar')
        if index['sample_contract'] != cfg['data']['sample_contract']:
            raise ValueError('sidecar样本时间合同不同')
        if assets is not None and index['assets'] != assets:
            raise ValueError('不同目录的物理资产合同不一致')
        assets = index['assets']
        if assets['sm_wm_sha256'] != bundle.metadata['sha256'] or assets['robot'] != bundle.metadata['robot']:
            raise ValueError('当前冻结SM/WM或执行器与真实缓存不一致')
        if assets['time'] != {'clock': 600, 'source': 30, 'control': 50, 'physics': 200}:
            raise ValueError('sidecar时间合同不匹配')
        indices.append({'path': str(index_path), 'sha256': digest(index_path), 'sources': len(index['sources'])})
        for name, identity in index['shards'].items():
            shards[str(directory/name)] = identity
        for key, entry in index['sources'].items():
            if key in sources:
                raise ValueError('重复source，禁止重复计数')
            sources[key] = (str(directory/entry['shard']), int(entry['episode']))
    if set(sources) != expected:
        raise ValueError(f'原train与sidecar来源不同: missing={len(expected-set(sources))}, extra={len(set(sources)-expected)}')
    return sources, shards, indices


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def run_audit(cfg, output, *, device='cpu', threads=8, batch_size=512, anchors=8,
              horizons=(1, 8, 16, 32), max_sources=None):
    from supermimic.bridge.bundle import load_bundle
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    if device != 'cpu':
        raise ValueError('此并行训练审计入口固定CPU，避免占用正式训练GPU')
    started = time.perf_counter()
    bundle = load_bundle(cfg['supermimic']['checkpoint'], device,
                         robot_asset_dir=cfg['supermimic']['robot_asset_dir'])
    frozen = bundle.frozen_snapshot()
    sources, shards, indices = source_inventory(cfg, bundle)
    selected = sorted(sources)
    if max_sources is not None:
        by_dataset = defaultdict(list)
        for key in selected:
            by_dataset[json.loads(key)[0]].append(key)
        selected = [key for row in itertools.zip_longest(*by_dataset.values())
                    for key in row if key is not None][:max_sources]
    manifest = {'schema': 'genmo.joint_physics.cached_wm_accuracy.v1',
        'checkpoint': bundle.metadata['source'], 'checkpoint_sha256': bundle.metadata['sha256'],
        'available_sources': len(sources), 'selected_sources': len(selected), 'indices': indices,
        'device': device, 'precision': 'fp32', 'threads': threads, 'batch_size': batch_size,
        'anchors_per_source': anchors, 'horizons': list(horizons),
        'sampling': 'one_step_all_pairs; evenly_spaced_source_anchors_shared_across_available_horizons',
        'evidence': 'recorded_real_actions_open_loop_prediction; no_newton_no_genmo_no_training',
        'evaluation_split': 'existing_train_sidecar; not_claimed_as_unseen_wm_training_data',
        'terminal_policy': 'retain_real_terminal_transition_if_stored; exclude_missing_next_state; no_reset_stitch',
        'units': 'meters,radians,meters_per_second,radians_per_second; root_xy_rmse_per_axis',
        'not_proven': ['current_student_closed_loop_accuracy', 'physical_gradient_direction', 'free_generation_quality']}
    write_json(output/'manifest.json', manifest)
    work = defaultdict(list)
    for key in selected:
        shard, index = sources[key]
        work[shard].append((key, index))
    records, verified_bytes = [], 0
    with (output/'sources.jsonl').open('x') as stream:
        for shard, entries in sorted(work.items()):
            if digest(shard) != shards[shard]:
                raise ValueError(f'sidecar分片SHA不同: {shard}')
            verified_bytes += Path(shard).stat().st_size
            payload = torch.load(shard, map_location='cpu', weights_only=False, mmap=True)
            for key, index in entries:
                episode = payload[index]
                if episode['source_key'] != key or episode['split'] != 'train':
                    raise ValueError('分片source或split身份不符')
                record = evaluate_episode(episode, bundle, batch_size=batch_size, anchors=anchors, horizons=horizons)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                stream.flush()
                records.append(record)
                if len(records) % 50 == 0 or len(records) == len(selected):
                    status = {'completed_sources': len(records), 'total_sources': len(selected),
                        'paired_transitions': sum(r['paired_transitions'] for r in records),
                        'wall_seconds': time.perf_counter()-started}
                    write_json(output/'status.json', status)
                    print(json.dumps(status), flush=True)
    bundle.assert_unchanged(frozen)
    if any(digest(i['path']) != i['sha256'] for i in indices):
        raise ValueError('验证期间sidecar索引发生变化，结果不可宣称为固定全集')
    summary = {**manifest, 'completed_sources': len(records), 'frozen_unchanged': True,
        'verified_shard_bytes': verified_bytes, 'wall_seconds': time.perf_counter()-started,
        'paired_transitions': sum(r['paired_transitions'] for r in records),
        'unpaired_final_actions': sum(r['unpaired_final_actions'] for r in records),
        'termination_reasons': dict(Counter(r['termination_reason'] for r in records)),
        'short_sources': [{'source': r['source'], 'pairs': r['paired_transitions'],
                           'available': r['horizons_available']} for r in records if max(horizons) not in r['horizons_available']],
        **aggregate(records)}
    write_json(output/'summary.json', summary)
    print(json.dumps({'complete': True, 'sources': len(records), 'summary': str(output/'summary.json'),
                      'wall_seconds': summary['wall_seconds']}), flush=True)
    return summary
