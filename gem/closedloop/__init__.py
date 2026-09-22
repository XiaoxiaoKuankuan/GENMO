"""GENMO 闭环规划的版本化公共契约。

当前包公开 BUMI Stage 1 的条件/监督批次契约，以及从完整音乐—qpos 配对数据构造因果
示范 proxy 历史、known prefix 和 qpos30/contact2 target 的独立 Dataset。它不改动现有
music-only Dataset、qpos30/contact2 checkpoint 协议或网络，也不实现冻结 GMT、通信、
Stage 2、DPPO 或执行侧 GMT reference 重采样。
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
    Stage1TrainingBatch,
    stage1_contract_summary,
    validate_stage1_condition_batch,
    validate_stage1_training_batch,
)
from .stage1_dataset import (
    CAUSAL_PROPRIO_CONSTRUCTION_VERSION,
    PREFIX_SOURCE,
    BumiClosedLoopStage1Dataset,
    CausalDemoProprio48Builder,
    collate_stage1_training_samples,
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
    "Stage1TrainingBatch",
    "CAUSAL_PROPRIO_CONSTRUCTION_VERSION",
    "PREFIX_SOURCE",
    "BumiClosedLoopStage1Dataset",
    "CausalDemoProprio48Builder",
    "collate_stage1_training_samples",
    "stage1_contract_summary",
    "validate_stage1_condition_batch",
    "validate_stage1_training_batch",
]
