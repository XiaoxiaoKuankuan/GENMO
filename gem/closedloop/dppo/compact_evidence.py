"""完整歌曲评估的单份列式控制证据与紧凑JSON索引。

逐50Hz控制奖励、八项跟踪、活动度、关节边界和事件保持无损，只在每任务结束时
按已有列式协议打包，使用标准库快速压缩保存一次。JSON只含决策摘要和文件引用；
读取时验证路径、大小和SHA再还原，因而仍能独立重算奖励与所有评估指标。
本模块不保存扩散链、不启动数据库、不操作模型，也不参与物理控制热循环。
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import zlib

from gem.runtime.trajectory_blocks import pack_trace, unpack_trace
from .journal_codec import encode_binary, decode_payload

SCHEMA = 'genmo.stage2.evaluation_control_archive.v1'


def write_episode(path, episode, *, publish_json, disk_guard=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {key: value for key, value in episode.items() if key != 'decisions'}
    decisions, blocks = [], []
    for decision in episode.get('decisions', []):
        controls = decision['controls']
        decisions.append({key: value for key, value in decision.items() if key != 'controls'})
        # 合法上层区间不超过25控制步；显式分块也支持诊断中的更长片段。
        blocks.append([pack_trace(controls[i:i+25]) for i in range(0, len(controls), 25)])
    payload = zlib.compress(encode_binary(dict(schema=SCHEMA, controls=blocks)), level=1)
    if disk_guard is not None:disk_guard.check(len(payload)+65536)
    target = path.with_suffix('.controls.bin.z')
    descriptor, temporary = tempfile.mkstemp(prefix='.'+target.name, dir=target.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    document.update(decisions=decisions, control_archive=dict(schema=SCHEMA, path=target.name,
        size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
        control_count=sum(len(d['controls']) for d in episode.get('decisions', []))))
    publish_json(path, document)
    if disk_guard is not None:
        disk_guard.account_file(target)
        disk_guard.account_file(path)


def read_episode(path):
    path = Path(path)
    document = json.loads(path.read_text())
    reference = document.get('control_archive')
    if reference is None:
        return document  # 读取既有历史证据；不提供旧训练算法或checkpoint兼容。
    target = (path.parent / reference['path']).resolve(strict=True)
    if reference['schema'] != SCHEMA or not target.is_relative_to(path.parent.resolve()):
        raise ValueError('Evaluation control archive escapes its episode directory')
    payload = target.read_bytes()
    if len(payload) != reference['size_bytes'] or hashlib.sha256(payload).hexdigest() != reference['sha256']:
        raise ValueError('Evaluation control archive size/SHA differs')
    data = decode_payload(zlib.decompress(payload))
    if data['schema'] != SCHEMA or len(data['controls']) != len(document['decisions']):
        raise ValueError('Evaluation control archive decision count differs')
    for decision, blocks in zip(document['decisions'], data['controls']):
        decision['controls'] = [row for block in blocks for row in unpack_trace(block)]
    if sum(len(d['controls']) for d in document['decisions']) != reference['control_count']:
        raise ValueError('Evaluation control archive actual control count differs')
    return document


def summary_only(report):
    """合并结果保留全部任务指标与证据引用，避免重复嵌入同一控制轨迹。"""
    return {key: value for key, value in report.items() if key != 'episodes'}
