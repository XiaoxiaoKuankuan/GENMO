"""BUMI 多 checkpoint 排序器的纯数据测试。

测试确认综合分采用候选集合内同量纲归一化、节拍分数方向与其他代价指标相反，并且部署
硬门槛的优先级高于软分数；当软分完全相同，使用更大的 global step 作为稳定次级排序。
测试不运行大模型、不创建评测输出，避免把 smoke/test 产物遗留在训练目录。
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from tools.eval.select_bumi_checkpoints import rank_candidates, validate_demo_report


def _candidate(name: str, step: int, *, violation: float, penetration: float, beat: float):
    return {
        "checkpoint": name,
        "global_step": step,
        "metrics": {
            "joint_limit_violation_rate": violation,
            "foot_penetration_max_m": penetration,
            "root_tilt_max_rad": 0.2,
            "beat_alignment_score": beat,
        },
    }


def test_hard_gate_precedes_soft_checkpoint_score() -> None:
    safe = _candidate("safe.ckpt", 100, violation=0.0, penetration=0.03, beat=0.6)
    unsafe = _candidate("unsafe.ckpt", 200, violation=0.2, penetration=0.0, beat=1.0)
    ranked = rank_candidates(
        [deepcopy(safe), deepcopy(unsafe)],
        max_joint_violation_rate=0.05,
        max_foot_penetration_m=0.08,
        max_root_tilt_rad=1.3,
    )
    assert ranked[0]["checkpoint"] == "safe.ckpt"
    assert ranked[0]["eligible"]
    assert not ranked[1]["eligible"]
    assert "joint_limit_violation_rate" in ranked[1]["hard_gate_reasons"]


def test_equal_metrics_prefer_newer_global_step() -> None:
    first = _candidate("s100.ckpt", 100, violation=0.0, penetration=0.01, beat=0.8)
    second = _candidate("s200.ckpt", 200, violation=0.0, penetration=0.01, beat=0.8)
    ranked = rank_candidates(
        [deepcopy(first), deepcopy(second)],
        max_joint_violation_rate=0.05,
        max_foot_penetration_m=0.08,
        max_root_tilt_rad=1.3,
    )
    assert [item["checkpoint"] for item in ranked] == ["s200.ckpt", "s100.ckpt"]
    assert ranked[0]["score"] == ranked[1]["score"] == 0.0


def test_cached_report_must_match_fixed_evaluation_identity() -> None:
    report = {
        "contract_version": "genmo.bumi_demo_report.v1",
        "checkpoint": {"sha256": "a" * 64},
        "seed": 7,
        "cfg_scale": 2.5,
        "ddim_steps": 20,
        "qpos_shape": [120, 28],
        "normalized_motion_shape": [120, 93],
    }
    validate_demo_report(
        report,
        checkpoint_sha256="a" * 64,
        expected_seed=7,
        cfg_scale=2.5,
        ddim_steps=20,
    )
    report["seed"] = 8
    with pytest.raises(ValueError, match="seed"):
        validate_demo_report(
            report,
            checkpoint_sha256="a" * 64,
            expected_seed=7,
            cfg_scale=2.5,
            ddim_steps=20,
        )
