"""固定训练库的CPU监督特征缓存，与Newton/WM离线sidecar严格分开。

只读原音乐动作、stats、FK，按0.1秒候选起点批量计算原codec局部anchor表示；绝不切片
全曲anchor下的qpos30冒充局部表示。保存全曲因果proprio、原接触/地面、窗口表示和固定
GT根坐标身体几何，训练通过mmap读取。没有GENMO、SM或WM推理，没有Newton环境。
文件按source身份命名，绑定原始SHA、manifest、stats及FK；索引原子发布，预取不修改文件。
"""

import hashlib
import json
import os
from pathlib import Path

import torch

from gem.closedloop.stage1_dataset import BumiClosedLoopStage1Dataset, _validate_qpos
from gem.robots.bumi.endecoder import BumiEndecoder
from .contracts import digest, source_key
from .config import ROOT

SCHEMA = 'genmo.joint_supervision_cache.local_anchor.v1'


def identity(cfg, catalog):
    return {'schema': SCHEMA, 'catalog': catalog.identity, 'sample_contract': cfg['data']['sample_contract'],
            'stats': digest(cfg['stage1']['endecoder']['stats_path']),
            'fk': digest(ROOT/cfg['stage1']['endecoder']['kinematics_path']), 'frames': 60, 'stride': 3}


@torch.no_grad()
def prepare_sequence(dataset, row, endecoder, chunk_size=256):
    reader = getattr(dataset.reader, 'reader', dataset.reader)
    sequence = reader.load_aligned_sequence(dataset.rows[row])
    sequence['_joint_contact'] = BumiClosedLoopStage1Dataset._contact_timeline(dataset, sequence)
    sequence['_joint_ground'] = BumiClosedLoopStage1Dataset._ground_supervision(dataset, sequence)
    sequence['_joint_proprio'] = dataset.proprio_builder.source_observations(_validate_qpos(sequence['qpos']))
    qpos = sequence['qpos']
    starts = torch.arange(0, len(qpos), 3)
    features, bodies = [], []
    for begin in range(0, len(starts), chunk_size):
        grid = starts[begin:begin+chunk_size, None] + torch.arange(61)
        encoded, _ = dataset.codec.encode_condition_features(qpos[grid.clamp_max(len(qpos)-1)])
        valid = (grid[:, :60] < len(qpos))[..., None].expand(-1, -1, 30).clone()
        valid[..., :2] &= (grid[:, :60]+1 < len(qpos))[..., None]
        target = torch.where(valid, encoded[:, :60], 0.)
        features.append(target)
        neutral = target.new_zeros(30); neutral[3] = neutral[7] = 1
        neutral[9:] = endecoder.kinematics.default_qpos[7:]
        decoded = endecoder.decode(endecoder.normalize(torch.where(valid, target, neutral)))
        poses = endecoder.compose_qpos(decoded)
        positions = endecoder.kinematics.forward_body_positions(poses)
        bodies.append(endecoder.codec.body_positions_in_root_frame(poses[..., :3], poses[..., 3:7], positions[..., 1:, :]))
    sequence['_joint_windows'] = torch.cat(features)
    sequence['_joint_target_body'] = torch.cat(bodies)
    return sequence


def write_source(directory, dataset, row, endecoder, contract):
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    key = source_key({'dataset_id': dataset.dataset_name, 'split': dataset.split,
                      'sample_id': str(dataset.rows[row]['sample_id'])})
    name = hashlib.sha256(key.encode()).hexdigest()+'.pt'
    path = Path(directory)/name
    if path.exists():
        payload = torch.load(path, weights_only=False, mmap=True, map_location='cpu')
        if payload['contract_sha256'] != contract_sha or payload['source'] != key:
            raise ValueError('监督缓存已有文件合同不匹配')
    else:
        sequence = prepare_sequence(dataset, row, endecoder)
        payload = {'contract_sha256': contract_sha, 'source': key, 'motion_sha256': digest(sequence['motion_path']),
                   'music_sha256': digest(sequence['music_path']), 'sequence': sequence}
        temporary = path.with_suffix(f'.partial-{os.getpid()}')
        try:
            with temporary.open('xb') as f:
                torch.save(payload, f)
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return key, {'file': name, 'sha256': digest(path), 'bytes': path.stat().st_size,
                 'motion_sha256': payload['motion_sha256'], 'music_sha256': payload['music_sha256'],
                 'frames': len(payload['sequence']['qpos']), 'windows': len(payload['sequence']['_joint_windows'])}


def read_manifest(directory, cfg, catalog):
    manifest = json.loads((Path(directory)/'index.json').read_text())
    if manifest['contract'] != identity(cfg, catalog):
        raise ValueError('CPU监督缓存原数据、stats、FK或样本合同不匹配')
    return manifest
