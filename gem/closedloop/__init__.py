"""GENMO 闭环规划的版本化公共契约。

当前包只公开 BUMI Stage 1 的条件批次定义与静态常量。它不创建真实 rollout 数据，
不改动现有 qpos30/contact2 训练与 checkpoint 协议，也不实现冻结 GMT、通信、DPPO、
30→50 Hz 重采样或速度派生。后续实现应在这些契约之上新增独立组件，而不是覆盖现有
music-only 基线。
"""

from .contracts import (
    CONTACT_DIM,
    GMT_COMMAND_FRAME_DIM,
    GMT_COMMAND_WINDOW_DIM,
    GMT_EXPECTED_JOINT_ORDER,
    GMT_HISTORY_DIM,
    GMT_POLICY_DIM,
    MOTION_FPS,
    MOTION_WINDOW_FRAMES,
    MUSIC_FEATURE_DIM,
    MUSIC_FPS,
    PROPRIO_CONTRACT_VERSION,
    PROPRIO_DIM,
    PROPRIO_FIELD_SPECS,
    PROPRIO_FPS,
    PROPRIO_HISTORY_STEPS,
    PROPRIO_SLICES,
    QPOS30_DIM,
    STAGE1_CONTRACT_VERSION,
    Stage1ConditionBatch,
    stage1_contract_summary,
    validate_stage1_condition_batch,
)

__all__ = [
    "CONTACT_DIM",
    "GMT_COMMAND_FRAME_DIM",
    "GMT_COMMAND_WINDOW_DIM",
    "GMT_EXPECTED_JOINT_ORDER",
    "GMT_HISTORY_DIM",
    "GMT_POLICY_DIM",
    "MOTION_FPS",
    "MOTION_WINDOW_FRAMES",
    "MUSIC_FEATURE_DIM",
    "MUSIC_FPS",
    "PROPRIO_CONTRACT_VERSION",
    "PROPRIO_DIM",
    "PROPRIO_FIELD_SPECS",
    "PROPRIO_FPS",
    "PROPRIO_HISTORY_STEPS",
    "PROPRIO_SLICES",
    "QPOS30_DIM",
    "STAGE1_CONTRACT_VERSION",
    "Stage1ConditionBatch",
    "stage1_contract_summary",
    "validate_stage1_condition_batch",
]
