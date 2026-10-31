#!/usr/bin/env python3
"""从联合 Stage1 白名单发布音乐/文本 Stage2 任务与显式机器人初态。

读取用户指定的联合数据 YAML、已经审计的 split allowlist 和 Stage2 配置；音乐读取
完整30Hz配对序列，文本读取完整qpos28及真实caption区间。按来源稳定选样，一条
母来源只选一次，train/val/test映射为train/validation/evaluation，不伪造独立划分。
初态只使用区间首姿态并按RobotSpec具名关节重排，速度置零；其余标签不写入名义
缓存。输出tasks_prepare.json与runtime.yaml，后续必须由冻结联合Stage1真正生成
名义缓存。工具不训练、不启动物理、不写源数据，也不指定旧s595000模型。
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare_tasks(data, bundle, output, *, train_per_source=8, heldout_per_source=2, seed=42,
                  min_duration_s=2.):
    import numpy as np
    import torch
    from gem.closedloop.stage1_multimodal_dataset import _load_allowlist, make_multimodal_source
    from gem.datasets.pure_motion.bumi_text import caption_intervals
    from tools.data.bumi.prepare_multimodal_stage1 import _record_identities
    allowlist = _load_allowlist(data)
    tasks, used = [], set()
    for original, split in (('train', 'train'), ('val', 'validation'), ('test', 'evaluation')):
        for name, entry in data['datasets'][original].items():
            source = make_multimodal_source(entry, source_name=name, split=original, evaluation=True)
            allowed = set(allowlist['entries'][f'{original}/{name}'])
            indices = [i for i, key in enumerate(source.record_keys) if key in allowed]
            indices.sort(key=lambda i: hashlib.sha256(f'{seed}:{name}:{source.record_keys[i]}'.encode()).hexdigest())
            needed, accepted = (train_per_source if split == 'train' else heldout_per_source), 0
            for index in indices:
                identities = _record_identities(source, index, data.get('source_identity_aliases', {}))
                if used.intersection(identities):
                    continue
                row = source.rows[index]
                text, caption_index = None, None
                if source.modality == 'text':
                    record = source.release.read_record(index)
                    intervals = caption_intervals(record)
                    possible = [i for i, (start, stop) in enumerate(intervals)
                                if (stop-start-1)/30 >= min_duration_s]
                    if not possible:
                        continue
                    caption_index = possible[0]
                    begin, end = intervals[caption_index]
                    text = record['captions'][caption_index]
                    sequence = source.reader.load_aligned_sequence(row, record=record)
                else:
                    sequence = source.reader.load_aligned_sequence(row)
                    begin, end = 0, len(sequence['qpos'])
                duration = (end-begin-1)*20
                if duration < round(min_duration_s*600):
                    continue
                qpos = sequence['qpos'][begin]
                names = tuple(source.kinematics.joint_order)
                if set(names) != set(bundle.spec.joint_names):
                    raise ValueError('联合源动作与冻结Actor关节名称不一致')
                initial = torch.zeros(55)
                initial[:7] = qpos[:7]
                initial[13:34] = qpos[7:][[names.index(n) for n in bundle.spec.joint_names]]
                initial_path = output/f'initial_{len(tasks):04d}.npy'
                np.save(initial_path, initial.numpy())
                group = hashlib.sha256('\n'.join(identities).encode()).hexdigest()
                task = dict(id=f'{split}_{name}_{len(tasks):04d}', source_group_id=group, song_id=group,
                    source_identity_tokens=identities, condition_kind=source.modality, split=split, seed=seed,
                    initial_state=str(initial_path), duration_ticks=duration, duration_s=duration/600,
                    source=name, source_record=source.record_keys[index], source_manifest=source.data_identity['manifest'],
                    source_interval_frames=[begin, end], initial_semantics='source_interval_first_pose_zero_velocity',
                    joint_names=list(bundle.spec.joint_names))
                if text is not None:
                    task.update(text=text, caption_index=caption_index)
                else:
                    task.update(music_features=str(sequence['music_path']), music_start_tick=0)
                tasks.append(task); used.update(identities); accepted += 1
                if accepted == needed:
                    break
            if accepted != needed:
                raise ValueError(f'{original}/{name}只找到{accepted}/{needed}条独立且时长合格的白名单动作')
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config', required=True)
    parser.add_argument('--config', required=True, help='绑定联合Stage1 checkpoint、统计和冻结SuperMimic的Stage2配置')
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-per-source', type=int, default=8)
    parser.add_argument('--heldout-per-source', type=int, default=2)
    parser.add_argument('--min-duration-s', type=float, default=2.)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    from omegaconf import OmegaConf
    data = OmegaConf.to_container(OmegaConf.load(args.data_config), resolve=True)
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    sys.path.insert(0, cfg['supermimic']['repo'])
    from supermimic.bridge.bundle import load_bundle
    from gem.closedloop.reference_adapter.application import read_tasks, require_file
    if min(args.train_per_source, args.heldout_per_source) < 1 or args.min_duration_s < 1.3:
        raise ValueError('每来源任务数必须为正，H64任务最短时长不得低于1.3秒')
    for name in ('checkpoint', 'stats', 'kinematics'):
        require_file(cfg['genmo'][name], 'genmo.'+name)
    bundle = load_bundle(cfg['supermimic']['checkpoint'], 'cpu', robot_asset_dir=cfg['supermimic']['robot_asset_dir'])
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    tasks = prepare_tasks(data, bundle, output, train_per_source=args.train_per_source,
        heldout_per_source=args.heldout_per_source, seed=args.seed, min_duration_s=args.min_duration_s)
    path = output/'tasks_prepare.json'
    path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2))
    read_tasks(path, preparing=True)
    cfg['tasks'] = str(path)
    cfg['task_sampling'] = {'music': .5, 'text': .5}
    OmegaConf.save(OmegaConf.create(cfg), output/'runtime.yaml')
    (output/'manifest.json').write_text(json.dumps(dict(data_config=str(Path(args.data_config).resolve()),
        split_allowlist=data['split_allowlist'], tasks=len(tasks), physics_run=False,
        nominal_cache_prepared=False, source_motion_used_only_for_initial_pose=True), ensure_ascii=False, indent=2))
    print(json.dumps(dict(tasks=str(path), config=str(output/'runtime.yaml'), count=len(tasks)), ensure_ascii=False))


if __name__ == '__main__':
    main()
