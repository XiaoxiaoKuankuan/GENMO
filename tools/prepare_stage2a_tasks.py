"""扩展Stage2-A训练音乐的可追溯任务清单，不用数据集动作替代GENMO名义参考。

只读取服务器已有授权音乐数据的训练manifest，按歌曲组、音频SHA、已有隔离划分
去重。数据集动作仅提供首帧显式物理复位，速度为零并按RobotSpec名称映射关节；
未来目标必须由另一个冻结GENMO真实生成命令产出。输出只创建新目录和分片配置，
保留现有20首的资产、划分、种子和时钟，不覆盖缓存。清单准备不等于生成成功。
"""

from pathlib import Path
import argparse
import json
import sys
import random
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from omegaconf import OmegaConf
    import numpy as np
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-songs', type=int, default=64)
    parser.add_argument('--shards', type=int, default=6)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    sys.path.insert(0, cfg['supermimic']['repo'])
    from supermimic.bridge.bundle import load_bundle, file_digest
    bundle = load_bundle(cfg['supermimic']['checkpoint'], robot_asset_dir=cfg['supermimic']['robot_asset_dir'])
    old = json.loads(Path(cfg['tasks']).read_text())
    output, dataset = Path(args.output).resolve(), Path(args.dataset).resolve()
    output.mkdir(parents=True, exist_ok=False)
    occupied_audio = {t['audio_sha256'] for t in old}
    occupied_motion = {str(Path(t['source_motion']).resolve()) for t in old}
    occupied_groups = set()
    manifests = {}
    for source in ('AIST++', 'AIOZ-GDANCE', 'FineDance', 'Mine'):
        rows = []
        for split in ('train', 'val', 'test'):
            manifest = dataset/source/'manifests'/f'{split}.jsonl'
            if not manifest.exists():
                continue
            for line in manifest.read_text().splitlines():
                row = json.loads(line)
                group = source+':'+str(row.get('music_group_id', row['audio_key']))
                if str((dataset/source/row['motion_path']).resolve()) in occupied_motion:
                    occupied_groups.add(group)
                if split == 'train' and row.get('quality_accepted') and row['num_frames'] >= 240:
                    rows.append((row, str(manifest), group))
        random.Random(42).shuffle(rows)
        manifests[source] = rows
    need = args.train_songs-sum(t['split'] == 'train' for t in old)
    chosen, cursor = [], {s: 0 for s in manifests}
    while len(chosen) < need:
        progressed = False
        for source, rows in manifests.items():
            while cursor[source] < len(rows) and len(chosen) < need:
                row, manifest, group = rows[cursor[source]]; cursor[source] += 1
                audio = dataset/source/row['audio_path']
                music = dataset/source/row['music_feature_path']
                motion = dataset/source/row['motion_path']
                if group in occupied_groups or not all(p.is_file() for p in (audio, music, motion)):
                    continue
                digest = file_digest(audio)
                if digest in occupied_audio:
                    continue
                source_motion = torch.load(motion, weights_only=False, map_location='cpu')
                if source_motion.get('quaternion_convention') != 'wxyz':
                    raise ValueError('必须显式确认来源四元数合同')
                order = [source_motion['joint_names'].index(j) for j in bundle.spec.joint_names]
                first = source_motion['qpos'][0].float()
                state = torch.cat((first[:7], torch.zeros(6), first[7:][order], torch.zeros(21)))
                initial = output/f'initial_{len(chosen):04d}.npy'
                np.save(initial, state.numpy(), allow_pickle=False)
                task = dict(id=source+'_'+row['sample_id'], song_id=group, split='train', seed=42,
                    initial_state=str(initial), music_features=str(music), audio_path=str(audio),
                    audio_sha256=digest, music_sha256=file_digest(music), music_start_tick=0,
                    source_manifest=manifest, source_motion=str(motion), original_frames=row['num_frames'], source=source,
                    initial_semantics='first_source_pose_zero_velocity_explicit_reset',
                    state_order='root_p3_quat_wxyz4_root_v3_root_w3_RobotSpec_q21_qd21',
                    nominal_requirement='must_generate_with_frozen_GENMO_no_future_dataset_motion')
                chosen.append(task); occupied_groups.add(group); occupied_audio.add(digest)
                progressed = True
                break
        if not progressed:
            raise RuntimeError(f'已有授权独立歌曲不足：新增{len(chosen)}，还需{need}')
    (output/'new_tasks.json').write_text(json.dumps(chosen, ensure_ascii=False, indent=2))
    (output/'existing_tasks.json').write_text(json.dumps(old, ensure_ascii=False, indent=2))
    for shard in range(args.shards):
        path = output/f'shard_{shard:02d}_tasks.json'
        path.write_text(json.dumps(chosen[shard::args.shards], ensure_ascii=False, indent=2))
        config = dict(cfg, tasks=str(path))
        OmegaConf.save(OmegaConf.create(config), output/f'shard_{shard:02d}.yaml')
    (output/'selection.json').write_text(json.dumps(dict(existing=len(old), added=len(chosen),
        total_train=args.train_songs, seed=42, split_policy='preserve_existing_holdouts_group_and_audio_disjoint',
        generation_completed=False, output=str(output)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
