"""BUMI closed-loop Stage 1 的纯条件批次契约。

本模块只固定第一版上层 GENMO 所消费的数据结构、物理语义和可机械检查的因果边界：
音乐与未来 qpos30 共用 30 Hz、120 点时间轴，机器人真实状态使用 50 Hz、48 维历史，
上一轮已发布且短期内不可改写的参考使用现有 physical qpos30 与逐坐标 known mask。

这里刻意不生成训练数据、不构造 target、不修改 qpos30/contact2 表示，也不实现 GMT 的
69/690/1092 维输入、30→50 Hz 重采样、速度派生、通信、Stage 2 或 DPPO。校验器只能
检查张量、时间和 mask 的结构约束；producer 是否确实只使用了 decision_time 之前的真实
状态，仍须由后续数据构造与 provenance 审计证明。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypedDict

import torch

from gem.robots.bumi.feature_codec import (
    BUMI_FEATURE_DIM,
    BUMI_FEATURE_SLICES,
    BUMI_REPRESENTATION_CONTRACT_VERSION,
)
from gem.runtime.bumi_music_contract import BUMI_ONNX_INPUTS, BUMI_ONNX_OUTPUTS
from gem.utils.music_features import EDGE_FEATURE_DIM, EDGE_TARGET_FPS

STAGE1_CONTRACT_VERSION = "genmo.bumi_closedloop.stage1.v1"
PROPRIO_CONTRACT_VERSION = "genmo.bumi_proprio48.v1"

MUSIC_FPS = EDGE_TARGET_FPS
MUSIC_FEATURE_DIM = EDGE_FEATURE_DIM
MOTION_FPS = 30
MOTION_WINDOW_FRAMES = int(BUMI_ONNX_INPUTS["music"][1])
QPOS30_DIM = BUMI_FEATURE_DIM
CONTACT_DIM = int(BUMI_ONNX_OUTPUTS["pred_foot_contact_logits"][2])

PROPRIO_FPS = 50
PROPRIO_HISTORY_STEPS = 50
PROPRIO_DIM = 48

GMT_POLICY_DIM = 69
GMT_HISTORY_STEPS = 10
GMT_HISTORY_DIM = 690
GMT_COMMAND_FRAME_DIM = 52
GMT_COMMAND_PAST_FRAMES = 10
GMT_COMMAND_CURRENT_FRAMES = 1
GMT_COMMAND_FUTURE_FRAMES = 10
GMT_COMMAND_WINDOW_FRAMES = 21
GMT_COMMAND_WINDOW_DIM = 1092

PROPRIO_SLICES: Mapping[str, tuple[int, int]] = MappingProxyType(
    {
        "projected_gravity": (0, 3),
        "base_ang_vel": (3, 6),
        "joint_pos_rel": (6, 27),
        "joint_vel_rel": (27, 48),
    }
)

# qpos30 中只有该字段需要下一采样点；它决定 committed-prefix 边界必须逐坐标标记。
QPOS30_NEXT_SAMPLE_DEPENDENT_FIELDS = ("root_delta_xy_heading",)

# mimic_noetix_bumi3_mha_sonic 的 PhysX native DoF 预期顺序。后续 GMT producer 必须在
# Isaac Lab 启动后将此表与 robot.joint_names 逐项比较；不能把 GENMO 的 MuJoCo qpos 顺序
# 直接用于 proprio48。
GMT_EXPECTED_JOINT_ORDER = (
    "l_leg_pitch_joint",
    "r_leg_pitch_joint",
    "waist_yaw_joint",
    "l_leg_roll_joint",
    "r_leg_roll_joint",
    "l_arm_pitch_joint",
    "r_arm_pitch_joint",
    "l_leg_yaw_joint",
    "r_leg_yaw_joint",
    "l_arm_roll_joint",
    "r_arm_roll_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_arm_yaw_joint",
    "r_arm_yaw_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_elbow_pitch_joint",
    "r_elbow_pitch_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
)

# Bumi_CFG 名义 default pose，已按 GMT_EXPECTED_JOINT_ORDER 排列。GMT 训练任务启动时会对
# active default_joint_pos 的每个环境、每个关节独立加 U[-0.02, 0.02] rad；joint_pos_rel
# 始终相对当时 active default，而不是无条件相对下面这张名义表。
GMT_NOMINAL_DEFAULT_JOINT_POS_RAD = (
    -0.1495,
    -0.1495,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.3,
    -0.3,
    0.3215,
    0.3215,
    0.0,
    0.0,
    -0.1720,
    -0.1720,
    0.0,
    0.0,
    0.0,
    0.0,
)
GMT_DEFAULT_JOINT_VEL_RAD_S = (0.0,) * 21
GMT_DEFAULT_JOINT_POS_STARTUP_NOISE_RAD = (-0.02, 0.02)


@dataclass(frozen=True, slots=True)
class ProprioFieldSpec:
    """单个 proprio48 字段的标准化前物理语义。"""

    name: str
    start: int
    stop: int
    unit: str
    coordinate_frame: str
    raw_semantics: str


PROPRIO_FIELD_SPECS = (
    ProprioFieldSpec(
        name="projected_gravity",
        start=0,
        stop=3,
        unit="dimensionless_unit_direction",
        coordinate_frame="base_link/root_link actor frame",
        raw_semantics=(
            "normalized world gravity direction inverse-rotated by root_link_quat_w; "
            "upright identity orientation is [0, 0, -1]"
        ),
    ),
    ProprioFieldSpec(
        name="base_ang_vel",
        start=3,
        stop=6,
        unit="rad/s",
        coordinate_frame="base_link/root_link actor frame",
        raw_semantics=(
            "root center-of-mass angular velocity in world, inverse-rotated by root_link_quat_w"
        ),
    ),
    ProprioFieldSpec(
        name="joint_pos_rel",
        start=6,
        stop=27,
        unit="rad",
        coordinate_frame="per-joint scalar in GMT PhysX native DoF order",
        raw_semantics="joint_pos - active per-environment default_joint_pos",
    ),
    ProprioFieldSpec(
        name="joint_vel_rel",
        start=27,
        stop=48,
        unit="rad/s",
        coordinate_frame="per-joint scalar in GMT PhysX native DoF order",
        raw_semantics=(
            "joint_vel - default_joint_vel; current task default_joint_vel is zero, "
            "so its value equals joint_vel"
        ),
    ),
)

# 这些范围只记录匹配 GMT 任务当前 policy group 的训练 corruption，不属于 proprio48 的
# 原始物理定义。HistoryObsCfg 不加这些噪声，GENMO 也不得借用 GMT 69D normalizer。
GMT_POLICY_TRAINING_NOISE_UNIFORM_RANGES: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "projected_gravity": (-0.05, 0.05),
        "base_ang_vel": (-0.2, 0.2),
        "joint_pos_rel": (-0.01, 0.01),
        "joint_vel_rel": (-0.5, 0.5),
    }
)


class Stage1ConditionBatch(TypedDict):
    """Stage 1 条件批次；不含尚未实现的监督 target。"""

    music_features: torch.Tensor
    music_valid: torch.Tensor
    proprio_history: torch.Tensor
    proprio_history_valid: torch.Tensor
    proprio_history_times: torch.Tensor
    known_qpos30: torch.Tensor
    known_qpos30_mask: torch.Tensor
    future_valid: torch.Tensor
    future_times: torch.Tensor
    decision_time: torch.Tensor


STAGE1_CONDITION_KEYS = (
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
)


def _require_tensor(batch: Mapping[str, Any], key: str) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{key} must be a torch.Tensor; got {type(value).__name__}")
    return value


def _require_shape(value: torch.Tensor, key: str, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{key} must have shape {expected}; got {tuple(value.shape)}")


def _require_float_finite(value: torch.Tensor, key: str) -> None:
    if not torch.is_floating_point(value):
        raise TypeError(f"{key} must use a floating dtype; got {value.dtype}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{key} contains NaN or Inf")


def _require_bool(value: torch.Tensor, key: str) -> None:
    if value.dtype != torch.bool:
        raise TypeError(f"{key} must use torch.bool; got {value.dtype}")


def _require_regular_time_axis(
    value: torch.Tensor,
    key: str,
    *,
    expected_step_seconds: float,
) -> None:
    delta = value[:, 1:] - value[:, :-1]
    if not bool((delta > 0.0).all()):
        raise ValueError(f"{key} must be strictly increasing")
    expected = torch.full_like(delta, float(expected_step_seconds))
    if not bool(torch.allclose(delta, expected, rtol=2.0e-4, atol=2.0e-6)):
        raise ValueError(f"{key} must advance by {expected_step_seconds:.12g} seconds per sample")


def validate_stage1_condition_batch(batch: Mapping[str, Any]) -> None:
    """Fail closed on Stage 1 条件张量的 shape、时间和 known-mask 约束。

    `known_qpos30` 是 standardization 前的 physical qpos30。padding 槽位可以包含任意有限
    数值；其无效性只能由对应 bool mask 表达，消费者不得把零值本身解释为 padding。
    本函数不填充、不归一化、不修改输入，也不能替代 producer 的因果 provenance 审计。
    """

    missing = sorted(set(STAGE1_CONDITION_KEYS) - set(batch))
    if missing:
        raise ValueError(f"Stage1 condition batch is missing required keys: {missing}")

    values = {key: _require_tensor(batch, key) for key in STAGE1_CONDITION_KEYS}
    decision_time = values["decision_time"]
    if decision_time.ndim != 1 or decision_time.shape[0] <= 0:
        raise ValueError(
            f"decision_time must have shape [B] with B > 0; got {tuple(decision_time.shape)}"
        )
    batch_size = int(decision_time.shape[0])

    expected_shapes = {
        "music_features": (batch_size, MOTION_WINDOW_FRAMES, MUSIC_FEATURE_DIM),
        "music_valid": (batch_size, MOTION_WINDOW_FRAMES),
        "proprio_history": (batch_size, PROPRIO_HISTORY_STEPS, PROPRIO_DIM),
        "proprio_history_valid": (batch_size, PROPRIO_HISTORY_STEPS),
        "proprio_history_times": (batch_size, PROPRIO_HISTORY_STEPS),
        "known_qpos30": (batch_size, MOTION_WINDOW_FRAMES, QPOS30_DIM),
        "known_qpos30_mask": (batch_size, MOTION_WINDOW_FRAMES, QPOS30_DIM),
        "future_valid": (batch_size, MOTION_WINDOW_FRAMES),
        "future_times": (batch_size, MOTION_WINDOW_FRAMES),
        "decision_time": (batch_size,),
    }
    for key, expected in expected_shapes.items():
        _require_shape(values[key], key, expected)

    for key in (
        "music_features",
        "proprio_history",
        "proprio_history_times",
        "known_qpos30",
        "future_times",
        "decision_time",
    ):
        _require_float_finite(values[key], key)
    for key in (
        "music_valid",
        "proprio_history_valid",
        "known_qpos30_mask",
        "future_valid",
    ):
        _require_bool(values[key], key)

    devices = {value.device for value in values.values()}
    if len(devices) != 1:
        raise ValueError(f"all Stage1 condition tensors must share one device; got {devices}")

    history_times = values["proprio_history_times"]
    future_times = values["future_times"]
    _require_regular_time_axis(
        history_times,
        "proprio_history_times",
        expected_step_seconds=1.0 / PROPRIO_FPS,
    )
    _require_regular_time_axis(
        future_times,
        "future_times",
        expected_step_seconds=1.0 / MOTION_FPS,
    )
    if bool((future_times[:, 0] < decision_time - 2.0e-6).any()):
        raise ValueError("the first future sample must not be earlier than decision_time")

    history_valid = values["proprio_history_valid"]
    late_history = history_valid & (history_times > decision_time[:, None] + 2.0e-6)
    if bool(late_history.any()):
        raise ValueError("valid proprio history samples must not be later than decision_time")

    known_mask = values["known_qpos30_mask"]
    future_valid = values["future_valid"]
    if bool((known_mask & ~future_valid[:, :, None]).any()):
        raise ValueError("known_qpos30_mask must be false where future_valid is false")

    # 每个坐标的 known 区域都必须是时间前缀；允许所有坐标全 false，即 prefix 长度为 0。
    if bool(((~known_mask[:, :-1]) & known_mask[:, 1:]).any()):
        raise ValueError("each known_qpos30 coordinate mask must form a temporal prefix")

    delta_start, delta_stop = BUMI_FEATURE_SLICES["root_delta_xy_heading"]
    root_delta_known = known_mask[:, :, delta_start:delta_stop]
    if bool((root_delta_known[:, :, 0] != root_delta_known[:, :, 1]).any()):
        raise ValueError("root_delta_xy_heading x/y coordinates must share one known mask")

    # qpos30 不显式保存绝对 root XY，所以仅凭本 batch 无法证明 delta[t] 的 p[t+1]
    # 来自上一轮已发布计划而不是本轮未来标签。本轮不使用其他 qpos 坐标伪造 provenance；
    # 该因果检查必须由第 3 步 producer 对未截断 source plan 执行。
    if bool(root_delta_known[:, -1].any()):
        raise ValueError(
            "the last in-window root_delta_xy_heading cannot be known without an explicit t+1 sample"
        )


def stage1_contract_summary() -> Mapping[str, Any]:
    """Return an immutable, dependency-light summary for diagnostics and tests."""

    return MappingProxyType(
        {
            "contract_version": STAGE1_CONTRACT_VERSION,
            "proprio_contract_version": PROPRIO_CONTRACT_VERSION,
            "qpos30_contract_version": BUMI_REPRESENTATION_CONTRACT_VERSION,
            "music_shape": (MOTION_WINDOW_FRAMES, MUSIC_FEATURE_DIM),
            "proprio_history_shape": (PROPRIO_HISTORY_STEPS, PROPRIO_DIM),
            "known_qpos30_shape": (MOTION_WINDOW_FRAMES, QPOS30_DIM),
            "contact_shape": (MOTION_WINDOW_FRAMES, CONTACT_DIM),
        }
    )
