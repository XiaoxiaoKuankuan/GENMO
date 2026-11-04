"""将已有GENMO名义缓存转换为新noetix训练代码使用的50Hz紧凑动作库。

只读取tasks声明的冻结GENMO缓存和EDGE35音乐；按具名关节映射到用户指定的
BUMI4340模型，位置/关节线性插值、旋转SLERP，不外推。保留原始train/validation/
audit/evaluation划分和每首数据来源哈希；只用train库训练。缓存中已生成的歌曲尾部
支持点也保留，另记真实音乐结束帧供整歌评测使用，不能把支持尾部算成音乐完成度。
不复制旧机器人的FK/PD数据，新环境按指定源码的机器人重新计算身体运动学。
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(tasks_path, destination):
    from supermimic.robots import get_robot_spec
    from supermimic.training.provenance import robot_identity
    from supermimic.data.motion_convert import _so3_derivative

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    tasks = json.loads(Path(tasks_path).read_text())
    robot = get_robot_spec('bumi4340')
    groups = {}
    for task in tasks:
        cache = torch.load(task['nominal_reference'], map_location='cpu', weights_only=False)
        if cache['source'] != 'frozen_genmo_accepted_reference' or cache['start_tick'] != 0:
            raise ValueError('必须使用零起点的真实GENMO名义缓存')
        contract = cache['contract']
        for key, path in (('music_sha256', task['music_features']), ('initial_sha256', task['initial_state'])):
            if contract[key] != digest(path):
                raise ValueError(f'{task["id"]}: {key}不匹配')
        if set(cache['joint_names']) != set(robot.joint_names):
            raise ValueError('缓存关节与指定BUMI4340不匹配')
        qpos = cache['world_qpos'].squeeze(0).numpy().astype(np.float64)
        order = [cache['joint_names'].index(name) for name in robot.joint_names]
        times = np.arange((len(qpos) - 1) * 50 // 30 + 1) / 50
        source_times = np.arange(len(qpos)) / 30
        p = np.stack([np.interp(times, source_times, qpos[:, i]) for i in range(3)], -1)
        q = np.stack([np.interp(times, source_times, qpos[:, 7 + i]) for i in order], -1)
        quat = Slerp(source_times, Rotation.from_quat(qpos[:, [4, 5, 6, 3]]))(times).as_quat()[:, [3, 0, 1, 2]]
        ref = np.concatenate((p, quat, q, np.gradient(p, .02, axis=0),
                              _so3_derivative(quat, .02), np.gradient(q, .02, axis=0)), -1).astype(np.float32)
        music_path = Path(task['music_features'])
        audio = (np.load(music_path, allow_pickle=False) if music_path.suffix == '.npy'
                 else torch.load(music_path, map_location='cpu', weights_only=True).numpy())
        audio = audio.reshape(-1, 35)
        ticks = np.arange(len(times)) * 12
        idx = ((ticks - task.get('music_start_tick', 0)) // 20)[:, None] + np.arange(16)[None]
        valid = (idx >= 0) & (idx < len(audio))
        preview = audio[idx.clip(0, len(audio)-1)] * valid[..., None]
        relative = (task.get('music_start_tick', 0) + idx*20 - ticks[:, None]) / 600
        features = np.concatenate((preview.reshape(len(times), -1), valid.astype(np.float32), relative), -1).astype(np.float32)
        if not np.isfinite(ref).all() or not np.isfinite(features).all():
            raise ValueError('动作或音乐存在非有限值')
        row = dict(task=task, source_sha256=digest(task['nominal_reference']),
                   music_end_frame=(task.get('music_start_tick', 0)+(len(audio)-1)*20)//12,
                   frames=len(times))
        if row['music_end_frame'] + 10 >= len(times):
            raise ValueError('真实名义尾部不够十帧；禁止外推或默默换歌')
        groups.setdefault(task['split'], []).append((ref, features, row))
    summary = dict(tasks=str(tasks_path), tasks_sha256=digest(tasks_path), splits={})
    for split, rows in groups.items():
        lengths = [len(row[0]) for row in rows]
        pack = dict(robot='bumi4340', joint_names=robot.joint_names, fps=50,
                    robot_identity=robot_identity(robot),
                    ref=torch.from_numpy(np.concatenate([row[0] for row in rows])),
                    music=torch.from_numpy(np.concatenate([row[1] for row in rows])),
                    names=[row[2]['task']['id'] for row in rows],
                    starts=np.cumsum([0, *lengths[:-1]]).tolist(), lengths=lengths,
                    provenance=[row[2] for row in rows])
        path = destination / f'{split}.pt'
        torch.save(pack, path)
        summary['splits'][split] = dict(path=str(path), songs=len(rows), frames=sum(lengths), sha256=digest(path))
    (destination / 'manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
