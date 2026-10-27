"""联合训练测试的原Stage1夹具和只读SM源码定位。

复用既有小型Transformer/合成数据夹具，只用于数值和合同测试；真实资产测试使用配置
中的固定SM checkpoint。优先接受SUPERMIMIC_REPO，不修改任何共享模型或仿真任务。
"""
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[3]
sm_root = os.getenv("SUPERMIMIC_REPO", os.getenv("JOINT_SM_REPO", str(root.parent.parent / "supermimic/SuperMimic")))
sys.path.insert(0, sm_root)
from tests.closedloop.test_stage1_actor import actor_factory  # noqa: F401,E402
