"""BUMI 原生 qpos30 表示、运动学、FK 接触标签和足底锁定公共接口。"""

from .contacts import (
    BUMI_CONTACT_CONTRACT_VERSION,
    BumiFootContactTargets,
    derive_bumi_foot_contact,
)
from .feature_codec import (
    BUMI_FEATURE_DIM,
    BUMI_FEATURE_SLICES,
    BUMI_REPRESENTATION_CONTRACT_VERSION,
    BumiMotionFeatureCodec,
)
from .kinematics import BumiKinematics
from .postprocess import (
    BUMI_FOOT_LOCK_CONTRACT_VERSION,
    BumiFootLockResult,
    lock_bumi_foot_contacts,
)

__all__ = [
    "BUMI_FEATURE_DIM",
    "BUMI_FEATURE_SLICES",
    "BUMI_CONTACT_CONTRACT_VERSION",
    "BUMI_FOOT_LOCK_CONTRACT_VERSION",
    "BUMI_REPRESENTATION_CONTRACT_VERSION",
    "BumiFootContactTargets",
    "BumiFootLockResult",
    "BumiKinematics",
    "BumiMotionFeatureCodec",
    "derive_bumi_foot_contact",
    "lock_bumi_foot_contacts",
]
