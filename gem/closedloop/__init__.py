"""GENMO 闭环规划的版本化公共契约。

当前包公开 BUMI Stage 1 的条件/监督批次契约，以及从完整音乐—qpos 配对数据构造因果
示范 proxy 历史、known prefix 和 qpos30/contact2 target 的独立 Dataset。数据集公共
名字按需导入，冻结GMT只读取契约/奖励时不会间接加载Stage1训练的日志和数据依赖。
原from-import和__all__保持兼容；真正使用数据集时仍执行原模块，不隐藏依赖错误。
它不改动现有
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
_DATASET_EXPORTS = frozenset((
    'CAUSAL_PROPRIO_CONSTRUCTION_VERSION', 'PREFIX_SOURCE',
    'BumiClosedLoopStage1Dataset', 'CausalDemoProprio48Builder', 'collate_stage1_training_samples',
))


def __getattr__(name):
    if name in _DATASET_EXPORTS:
        from importlib import import_module
        value = getattr(import_module('.stage1_dataset', __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

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
