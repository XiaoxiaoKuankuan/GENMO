"""准备随机128首Adapter动力学评价的独立歌曲清单，不执行训练或伪造参考。

默认从当前Adapter训练歌曲随机抽32首，再从其余授权数据集歌曲随机抽96首。
按音乐组和实际音频SHA双重去重，保留原数据划分及本轮训练见过/未见过标记。
已有完整GENMO缓存只读复用；新歌仅从原动作首帧构造零速度物理初态，未来参考
必须由现有prepare_stage2a_cache冻结GENMO流程实际生成。输出随机种子、候选池
规模、全部抽样清单和互斥缓存分片；生成失败不得换歌，也不得减少正式统计分母。
所有产物写入新目录，不修改正式训练配置、模型、数据或已有缓存。
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import sys


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def sample_unique(pool, count, rng, occupied_groups, occupied_audio):
    """先按歌组等概率打乱，再组内抽样；真实音频去重，不按动作条数加权。"""
    groups = collections.defaultdict(list)
    for task in pool:
        if task['song_id'] not in occupied_groups:
            groups[task['song_id']].append(task)
    order = sorted(groups)
    rng.shuffle(order)
    chosen = []
    for group in order:
        rows = sorted(groups[group], key=lambda task: task['id'])
        rng.shuffle(rows)
        for task in rows:
            audio_hash = digest(task['audio_path'])
            if audio_hash in occupied_audio:
                continue
            chosen.append(task | {'audio_sha256': audio_hash})
            occupied_groups.add(group)
            occupied_audio.add(audio_hash)
            break
        if len(chosen) == count:
            return chosen
    raise ValueError(f'独立歌曲不足：需要{count}，实际只有{len(chosen)}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--songs', type=int, default=128)
    p.add_argument('--seen-songs', type=int, default=32)
    p.add_argument('--seed', type=int, default=20261031)
    p.add_argument('--shards', type=int, default=8)
    args = p.parse_args()
    if not 0 < args.seen_songs < args.songs or args.shards < 1:
        p.error('需要非空的训练已见与未见歌曲，以及正数缓存分片')
    import numpy as np
    import torch
    import yaml
    cfg = yaml.safe_load(args.config.read_text())
    sys.path.insert(0, cfg['supermimic']['repo'])
    from supermimic.bridge.bundle import load_bundle
    old = json.loads(Path(cfg['tasks']).read_text())
    seen = [task for task in old if task['split']=='train']
    seen_groups = {task['song_id'] for task in seen}
    seen_audio = {task['audio_sha256'] for task in seen}
    rng = random.Random(args.seed)
    chosen = sample_unique(seen, args.seen_songs, rng, set(), set())
    chosen = [task | dict(adapter_training_seen=True, dataset_split='train') for task in chosen]
    pool = []
    for source in ('AIST++', 'AIOZ-GDANCE', 'FineDance', 'Mine'):
        for original_split, split in [('train','train'), ('val','validation'), ('test','evaluation')]:
            manifest = args.dataset/source/'manifests'/f'{original_split}.jsonl'
            if not manifest.exists():
                continue
            for line in manifest.read_text().splitlines():
                row = json.loads(line)
                group = source+':'+str(row.get('music_group_id', row['audio_key']))
                if group in seen_groups or not row.get('quality_accepted') or row['num_frames'] < 240:
                    continue
                paths = {key: str((args.dataset/source/row[value]).resolve()) for key, value in
                         [('audio_path','audio_path'), ('music_features','music_feature_path'),
                          ('source_motion','motion_path')]}
                if not all(Path(path).is_file() for path in paths.values()):
                    continue
                pool.append(dict(id=source+'_'+row['sample_id'], song_id=group, split=split,
                    dataset_split=original_split, source=source, source_manifest=str(manifest),
                    original_frames=row['num_frames'], seed=42, music_start_tick=0,
                    adapter_training_seen=False, **paths))
    new = sample_unique(pool, args.songs-args.seen_songs, rng, set(seen_groups), set(seen_audio))
    bundle = load_bundle(cfg['supermimic']['checkpoint'], 'cpu',
                         robot_asset_dir=cfg['supermimic']['robot_asset_dir'])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for index, task in enumerate(new):
        motion = torch.load(task['source_motion'], map_location='cpu', weights_only=False)
        music = torch.load(task['music_features'], map_location='cpu', weights_only=True)
        qpos = torch.as_tensor(motion['qpos']).float()
        names = tuple(motion['joint_names'])
        if (motion.get('quaternion_convention') != 'wxyz' or qpos.shape[1:] != (28,)
                or music.shape != (len(qpos),35) or len(names)!=21
                or set(names)!=set(bundle.spec.joint_names)):
            raise ValueError(f'来源状态/关节/音乐合同不匹配：{task["id"]}')
        initial = torch.zeros(55)
        initial[:7] = qpos[0,:7]
        initial[13:34] = qpos[0,7:][[names.index(name) for name in bundle.spec.joint_names]]
        path = output/f'initial_{index:03d}.npy'
        np.save(path, initial.numpy(), allow_pickle=False)
        task.update(initial_state=str(path), music_sha256=digest(task['music_features']),
            initial_semantics='first_source_pose_zero_velocity_explicit_reset',
            state_order='root_p3_quat_wxyz4_root_v3_root_w3_RobotSpec_q21_qd21',
            nominal_requirement='must_generate_with_frozen_GENMO_no_future_dataset_motion')
    all_tasks = chosen+new
    rng.shuffle(all_tasks)
    def write(name, value):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    write('selected_tasks.json', all_tasks)
    write('existing_tasks.json', chosen)
    write('new_tasks.json', new)
    for index in range(args.shards):
        write(f'shard_{index:02d}_tasks.json', new[index::args.shards])
    write('selection.json', dict(seed=args.seed, requested=args.songs,
        unique_songs=len({task['song_id'] for task in all_tasks}),
        unique_audio=len({task['audio_sha256'] for task in all_tasks}),
        adapter_seen=len(chosen), adapter_unseen=len(new),
        eligible_seen_groups=len(seen_groups), eligible_unseen_groups=len({t['song_id'] for t in pool}),
        source_counts=dict(collections.Counter(task['source'] for task in all_tasks)),
        split_counts=dict(collections.Counter(task['split'] for task in all_tasks)),
        protocol=f'stratified_random_song_groups_{len(chosen)}_adapter_seen_{len(new)}_adapter_unseen',
        source_config=str(args.config), generation_completed=False,
        failure_policy='keep_selected_128_no_replacement_no_silent_exclusion'))
    print(json.dumps(dict(output=str(output), selected=len(all_tasks), need_generation=len(new)), ensure_ascii=False))


if __name__ == '__main__':
    main()
