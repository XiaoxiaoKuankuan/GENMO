"""四步KL标定证据的运行门禁，不用旧全20步阈值冒充新算法验收。

有限固定rollout试验及同新奖励的真实物理对照必须先落盘。正式学习只接受绑定
当前行为核、奖励、选定soft/hard门槛的SHA证据，pending配置仅可用于采集、评估、
部署延迟标定或Critic预热。这里不猜测阈值、不修改采样，也不读取旧P1优化状态。
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_calibration(limits,kernel,reward):
    binding=limits['calibration']
    if binding.get('status')!='validated' or not binding.get('evidence'):
        raise ValueError('Four-step KL requires real fixed-rollout and physical calibration before Actor learning')
    evidence=binding['evidence'];path=Path(evidence['path'])
    content=path.read_bytes()
    if hashlib.sha256(content).hexdigest()!=evidence['sha256']:raise ValueError('KL calibration evidence SHA differs')
    report=json.loads(content)
    if (report.get('schema')!='genmo.stage2.four_step_kl_calibration.v1' or report.get('status')!='passed' or
            report.get('kernel_sha256')!=digest(kernel) or report.get('reward_sha256')!=digest(reward)):
        raise ValueError('KL calibration belongs to another policy or reward')
    keys=('mean_coordinate','maximum_step_mean_coordinate','p95_coordinate')
    if report.get('hard_limits')!={key:limits[key] for key in keys}:
        raise ValueError('KL thresholds differ from measured calibration')
    if not report.get('fixed_rollout_trials') or not report.get('physical_comparison'):
        raise ValueError('Probability replay alone cannot calibrate a policy update')
    return report
