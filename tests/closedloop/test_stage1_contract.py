"""BUMI closed-loop Stage 1 契约的纯结构回归测试。

测试只构造小型内存张量，核对 qpos30/EDGE35/proprio48 的版本、shape、时间轴和逐坐标
prefix mask 规则；不会读取真实训练数据、生成 target、启动 Isaac/GMT、运行网络或写出
rollout。测试通过只能证明静态 contract 自洽，不能证明 producer 的状态确实来自真实机器人、
不存在未来泄漏，也不能证明 Stage 2、动力学或实机能力。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from gem.closedloop.contracts import (
    CONTACT_DIM,
    GMT_COMMAND_FRAME_DIM,
    GMT_COMMAND_WINDOW_DIM,
    GMT_EXPECTED_JOINT_ORDER,
    GMT_HISTORY_DIM,
    GMT_NOMINAL_DEFAULT_JOINT_POS_RAD,
    GMT_POLICY_DIM,
    MOTION_FPS,
    MOTION_WINDOW_FRAMES,
    MUSIC_FEATURE_DIM,
    PROPRIO_CONTRACT_VERSION,
    PROPRIO_DIM,
    PROPRIO_FIELD_SPECS,
    PROPRIO_FPS,
    PROPRIO_HISTORY_STEPS,
    PROPRIO_SLICES,
    QPOS30_DIM,
    QPOS30_NEXT_SAMPLE_DEPENDENT_FIELDS,
    STAGE1_CONDITION_KEYS,
    STAGE1_CONTRACT_VERSION,
    Stage1ConditionBatch,
    validate_stage1_condition_batch,
)
from gem.robots.bumi.feature_codec import (
    BUMI_FEATURE_DIM,
    BUMI_FEATURE_SLICES,
    BUMI_REPRESENTATION_CONTRACT_VERSION,
)
from gem.runtime.bumi_music_contract import BUMI_ONNX_INPUTS, BUMI_ONNX_OUTPUTS
from gem.utils.music_features import (
    EDGE_BASELINE_FEATURE_NAMES,
    EDGE_FEATURE_DIM,
    EDGE_TARGET_FPS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_CONFIG = REPO_ROOT / "configs/closedloop/stage1_contract_v1.yaml"
CONTRACT_DOC = REPO_ROOT / "docs/closedloop/stage1_contract_v1.md"


def _make_valid_batch(batch_size: int = 2) -> Stage1ConditionBatch:
    decision_time = torch.tensor(
        [10.0 + float(index) for index in range(batch_size)], dtype=torch.float64
    )
    history_offsets = torch.arange(
        PROPRIO_HISTORY_STEPS - 1,
        -1,
        -1,
        dtype=torch.float64,
    ) / float(PROPRIO_FPS)
    future_offsets = torch.arange(MOTION_WINDOW_FRAMES, dtype=torch.float64) / float(MOTION_FPS)
    # v1 不强制 future 的首点等于 decision_time；显式保留一个调度 offset。
    future_start = decision_time + 0.125
    return {
        "music_features": torch.randn(
            batch_size, MOTION_WINDOW_FRAMES, MUSIC_FEATURE_DIM, dtype=torch.float32
        ),
        "music_valid": torch.ones(batch_size, MOTION_WINDOW_FRAMES, dtype=torch.bool),
        "proprio_history": torch.randn(
            batch_size, PROPRIO_HISTORY_STEPS, PROPRIO_DIM, dtype=torch.float32
        ),
        "proprio_history_valid": torch.ones(batch_size, PROPRIO_HISTORY_STEPS, dtype=torch.bool),
        "proprio_history_times": decision_time[:, None] - history_offsets[None, :],
        "known_qpos30": torch.randn(
            batch_size, MOTION_WINDOW_FRAMES, QPOS30_DIM, dtype=torch.float32
        ),
        "known_qpos30_mask": torch.zeros(
            batch_size, MOTION_WINDOW_FRAMES, QPOS30_DIM, dtype=torch.bool
        ),
        "future_valid": torch.ones(batch_size, MOTION_WINDOW_FRAMES, dtype=torch.bool),
        "future_times": future_start[:, None] + future_offsets[None, :],
        "decision_time": decision_time,
    }


def test_authoritative_dimensions_and_slices_are_preserved() -> None:
    assert STAGE1_CONTRACT_VERSION == "genmo.bumi_closedloop.stage1.v1"
    assert PROPRIO_CONTRACT_VERSION == "genmo.bumi_proprio48.v1"
    assert BUMI_REPRESENTATION_CONTRACT_VERSION == "genmo.bumi_motion_features.qpos30.v3"
    assert QPOS30_DIM == BUMI_FEATURE_DIM == 30
    assert dict(BUMI_FEATURE_SLICES) == {
        "root_delta_xy_heading": (0, 2),
        "root_height_offset": (2, 3),
        "root_rot_local": (3, 9),
        "joint_dof": (9, 30),
    }
    assert QPOS30_NEXT_SAMPLE_DEPENDENT_FIELDS == ("root_delta_xy_heading",)

    assert MOTION_FPS == EDGE_TARGET_FPS == 30
    assert MOTION_WINDOW_FRAMES == BUMI_ONNX_INPUTS["music"][1] == 120
    assert MUSIC_FEATURE_DIM == EDGE_FEATURE_DIM == 35
    assert len(EDGE_BASELINE_FEATURE_NAMES) == 35
    assert EDGE_BASELINE_FEATURE_NAMES[0] == "onset_strength"
    assert EDGE_BASELINE_FEATURE_NAMES[1:21] == tuple(f"mfcc_{index:02d}" for index in range(1, 21))
    assert EDGE_BASELINE_FEATURE_NAMES[21:33] == tuple(
        f"chroma_cens_{index:02d}" for index in range(1, 13)
    )
    assert EDGE_BASELINE_FEATURE_NAMES[-2:] == ("onset_peak", "beat_peak")
    assert CONTACT_DIM == BUMI_ONNX_OUTPUTS["pred_foot_contact_logits"][2] == 2


def test_proprio48_layout_and_gmt_joint_contract() -> None:
    assert PROPRIO_DIM == 48
    assert PROPRIO_FPS == 50
    assert PROPRIO_HISTORY_STEPS == 50
    assert dict(PROPRIO_SLICES) == {
        "projected_gravity": (0, 3),
        "base_ang_vel": (3, 6),
        "joint_pos_rel": (6, 27),
        "joint_vel_rel": (27, 48),
    }
    assert tuple((field.name, field.start, field.stop) for field in PROPRIO_FIELD_SPECS) == (
        ("projected_gravity", 0, 3),
        ("base_ang_vel", 3, 6),
        ("joint_pos_rel", 6, 27),
        ("joint_vel_rel", 27, 48),
    )
    assert PROPRIO_FIELD_SPECS[0].unit == "dimensionless_unit_direction"
    assert PROPRIO_FIELD_SPECS[1].unit == "rad/s"
    assert len(GMT_EXPECTED_JOINT_ORDER) == 21
    assert len(set(GMT_EXPECTED_JOINT_ORDER)) == 21
    assert len(GMT_NOMINAL_DEFAULT_JOINT_POS_RAD) == 21
    assert GMT_EXPECTED_JOINT_ORDER[:3] == (
        "l_leg_pitch_joint",
        "r_leg_pitch_joint",
        "waist_yaw_joint",
    )


def test_valid_batch_and_exact_sample_spans() -> None:
    batch = _make_valid_batch()
    validate_stage1_condition_batch(batch)

    history_span = batch["proprio_history_times"][:, -1] - batch["proprio_history_times"][:, 0]
    future_span = batch["future_times"][:, -1] - batch["future_times"][:, 0]
    assert torch.allclose(history_span, torch.full_like(history_span, 49.0 / 50.0))
    assert torch.allclose(future_span, torch.full_like(future_span, 119.0 / 30.0))
    assert not torch.allclose(future_span, torch.full_like(future_span, 4.0))


def test_empty_prefix_and_masked_nonzero_padding_are_valid() -> None:
    batch = _make_valid_batch()
    batch["proprio_history_valid"][:, :7] = False
    batch["proprio_history"][:, :7] = 123.0
    batch["music_valid"][:, -9:] = False
    batch["music_features"][:, -9:] = 456.0
    batch["future_valid"][:, -5:] = False
    batch["known_qpos30"][:, -5:] = 789.0

    assert not bool(batch["known_qpos30_mask"].any())
    validate_stage1_condition_batch(batch)


def test_coordinate_level_prefix_accepts_unknown_boundary_delta() -> None:
    batch = _make_valid_batch(batch_size=1)
    prefix_state_frames = 12
    # 当前帧字段在 0..11 已承诺；delta 仅在同时拥有下一帧的 0..10 可知。
    batch["known_qpos30_mask"][:, :prefix_state_frames, 2:] = True
    batch["known_qpos30_mask"][:, : prefix_state_frames - 1, :2] = True

    validate_stage1_condition_batch(batch)
    assert bool(batch["known_qpos30_mask"][0, prefix_state_frames - 1, 2:].all())
    assert not bool(batch["known_qpos30_mask"][0, prefix_state_frames - 1, :2].any())


def test_root_delta_mask_pair_and_terminal_dependency_fail_closed() -> None:
    batch = _make_valid_batch(batch_size=1)
    batch["known_qpos30_mask"][:, :4, 2:] = True
    batch["known_qpos30_mask"][:, :3, 0] = True
    with pytest.raises(ValueError, match="x/y coordinates"):
        validate_stage1_condition_batch(batch)

    batch = _make_valid_batch(batch_size=1)
    batch["known_qpos30_mask"][:, :, 2:] = True
    batch["known_qpos30_mask"][:, :, :2] = True
    with pytest.raises(ValueError, match=r"last in-window root_delta_xy_heading"):
        validate_stage1_condition_batch(batch)


def test_each_coordinate_known_mask_must_be_a_prefix() -> None:
    batch = _make_valid_batch(batch_size=1)
    batch["known_qpos30_mask"][:, 0, 2] = True
    batch["known_qpos30_mask"][:, 2, 2] = True
    with pytest.raises(ValueError, match="must form a temporal prefix"):
        validate_stage1_condition_batch(batch)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda batch: batch.update(
                {"known_qpos30_mask": torch.zeros(2, 120, dtype=torch.bool)}
            ),
            "known_qpos30_mask must have shape",
        ),
        (
            lambda batch: batch.update({"music_valid": torch.ones(2, 120, dtype=torch.float32)}),
            "music_valid must use torch.bool",
        ),
        (
            lambda batch: batch["future_times"].__setitem__(
                (slice(None), 20), batch["future_times"][:, 19]
            ),
            "future_times must be strictly increasing",
        ),
        (
            lambda batch: batch.update(
                {"proprio_history_times": batch["proprio_history_times"] + 0.02}
            ),
            "valid proprio history samples",
        ),
        (
            lambda batch: batch.update(
                {
                    "future_times": batch["future_times"]
                    - (batch["future_times"][:, :1] - batch["decision_time"][:, None] + 0.02)
                }
            ),
            "first future sample must not be earlier than decision_time",
        ),
        (
            lambda batch: batch["known_qpos30_mask"].__setitem__((slice(None), -1, 2), True),
            "future_valid is false",
        ),
    ],
)
def test_invalid_batches_fail_closed(mutate, message: str) -> None:
    batch = _make_valid_batch()
    if "future_valid is false" in message:
        batch["future_valid"][:, -1] = False
    mutate(batch)
    with pytest.raises((TypeError, ValueError), match=message):
        validate_stage1_condition_batch(batch)


def test_nonfinite_values_are_rejected() -> None:
    batch = _make_valid_batch()
    batch["proprio_history"][0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="proprio_history contains NaN or Inf"):
        validate_stage1_condition_batch(batch)


def test_unrelated_metadata_does_not_replace_required_key_validation() -> None:
    batch = _make_valid_batch()
    runtime_batch = dict(batch)
    runtime_batch["sample_id"] = ["sample-a", "sample-b"]
    validate_stage1_condition_batch(runtime_batch)

    runtime_batch.pop("decision_time")
    with pytest.raises(ValueError, match="missing required keys.*decision_time"):
        validate_stage1_condition_batch(runtime_batch)


def test_condition_type_does_not_claim_stage3_targets() -> None:
    assert set(STAGE1_CONDITION_KEYS) == {
        "music_features",
        "music_valid",
        "proprio_history",
        "proprio_history_valid",
        "proprio_history_times",
        "known_qpos30",
        "known_qpos30_mask",
        "future_valid",
        "future_times",
        "decision_time",
    }
    assert not {"target_qpos30", "target_contact", "target_contact_valid"} & set(
        Stage1ConditionBatch.__annotations__
    )


def test_yaml_and_document_match_python_contract() -> None:
    raw = yaml.safe_load(CONTRACT_CONFIG.read_text(encoding="utf-8"))
    assert raw["contract_version"] == STAGE1_CONTRACT_VERSION
    assert raw["scope"] == "condition_only"
    assert raw["implementation_status"] == "contract_only"
    assert raw["stage1"]["required_condition_fields"] == list(STAGE1_CONDITION_KEYS)
    assert raw["stage1"]["outputs"]["future_motion_qpos30"] == ["B", 120, 30]
    assert raw["stage1"]["outputs"]["future_contact_logits"] == ["B", 120, 2]
    assert raw["stage2"]["implementation_status"] == "not_implemented"
    assert raw["stage2"]["trainable_components"] == ["genmo", "upper_critic"]
    assert raw["stage2"]["frozen_components"] == ["gmt"]
    assert raw["motion"]["feature_dim"] == QPOS30_DIM
    assert raw["motion"]["frames"] == MOTION_WINDOW_FRAMES
    assert raw["motion"]["forbidden_changes"] == {
        "predict_joint_velocity": False,
        "change_bumi_feature_dim": False,
        "create_motion51_representation": False,
        "recompute_51d_stats": False,
        "mutate_legacy_checkpoint_contract": False,
    }
    assert raw["music"]["feature_dim"] == MUSIC_FEATURE_DIM
    assert raw["proprio"]["contract_version"] == PROPRIO_CONTRACT_VERSION
    assert raw["proprio"]["joint_order"] == list(GMT_EXPECTED_JOINT_ORDER)
    assert raw["time"]["future_start_not_before_decision_time"] is True
    assert raw["time"]["future_start_equals_decision_time"] == "not_required"
    assert raw["prefix"]["mask_granularity"] == "per_coordinate"
    assert raw["prefix"]["allow_empty"] is True
    assert raw["prefix"]["nominal_commit_duration_seconds"] is None
    assert "root_position[t+1]" in raw["prefix"]["root_delta_boundary_rule"]
    assert raw["reserved_stage3_targets"]["implementation_status"] == "not_implemented"

    downstream = raw["downstream_gmt"]
    assert downstream["policy"]["shape"] == ["B", GMT_POLICY_DIM]
    assert downstream["history_obs"]["shape"] == ["B", GMT_HISTORY_DIM]
    assert downstream["command_window"]["single_frame_dim"] == GMT_COMMAND_FRAME_DIM
    assert downstream["command_window"]["shape"] == ["B", GMT_COMMAND_WINDOW_DIM]
    assert downstream["policy_fps"] == 50
    assert downstream["physics_fps"] == 200
    assert downstream["physics_steps_per_policy_step"] == 4
    assert raw["replanning"]["status"] == "planned_not_measured"

    document = CONTRACT_DOC.read_text(encoding="utf-8")
    assert "future_times[119] = t0 + 119/30" in document
    assert "不能写成最后采样点位于 `t0+4.0`" in document
    assert "GMT 权重始终冻结" in document
    assert "尚未完成的第 3 步" in document
    assert "尚未完成的第 4 步" in document
