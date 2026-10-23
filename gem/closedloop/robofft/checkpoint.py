"""显式 DPPO -> RoboFFT Actor weights-only 初始化，禁止继承旧学习状态。

输入必须是完整、可识别的第二阶段 checkpoint，绑定相同 Stage1/统计量/机器人资产。
仅验证并载入可训练 Actor；冻结前段/末段仍从共同 Stage1 源重建，Critic、两个 Adam、
rollout、概率缓存、MC RNG 和进度全部由新实验重新初始化。源码 SHA 仅在来源记录中
保留，不作为权重导入门禁。普通 --resume 使用 RoboFFT 自身完整 schema 和恢复协议。
"""
from __future__ import annotations

from pathlib import Path
import torch
from . import CHECKPOINT_SCHEMA
from ..dppo.training_checkpoint import SCHEMA, file_sha256
from ..dppo.checkpoint import _validate_model_state


def initialize_actor_weights_only(path, actor, *, assets):
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    previous = payload.get('identity', {})
    if payload.get('schema') not in (SCHEMA, CHECKPOINT_SCHEMA):
        raise ValueError('Explicit weights-only initialization requires a recognized Stage2 checkpoint')
    for key in ('checkpoint', 'stats', 'kinematics'):
        if previous.get('assets', {}).get(key) != assets[key]:
            raise ValueError('Weights-only Stage1/normalization/robot asset mismatch: '+key)
    _validate_model_state(actor, payload['actor'], 'RoboFFT weights-only Actor')
    actor.load_state_dict(payload['actor'], strict=True)
    return dict(kind='explicit_actor_weights_only_new_robofft_experiment',
        path=str(Path(path).resolve()), sha256=file_sha256(path), source_schema=payload['schema'],
        source_iteration=payload.get('state', {}).get('iteration'),
        source_files=previous.get('source_files'), source_version_is_informational=True,
        optimizer_restored=False, critic_restored=False, rollout_reused=False, rng_restored=False)
