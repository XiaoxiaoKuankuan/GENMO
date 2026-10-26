"""验证分布覆盖、状态支持和独立物理报告比较的纯合同测试。

这些手工记录只验证拒绝逻辑和指标计算，绝不作为真实WM精度、真实物理完成率或
训练收益证据。真实物理报告必须由Newton入口另外生成；本文件不能解锁正式启动。
"""

import pytest
import torch

from gem.closedloop.mb_control.validation import ModelSupport, UnsupportedModelState, state_envelope, validate_coverage
from gem.closedloop.mb_control.formal import paired_quality


def test_training_id_need_not_be_in_independent_validation():
    tasks=[{"id":"train_a","distribution_id":"domain","source":"FineDance"}]
    report={"coverage":{"physics":"nominal_flat_newton_mujoco_warp","tasks":[
        {"id":"val_b","distribution_id":"domain","source":"FineDance","split":"val"}]},"validated_state_envelope":{"present":True}}
    validate_coverage(tasks,report)
    report["coverage"]["tasks"][0]["id"]="train_a"
    with pytest.raises(ValueError):
        validate_coverage(tasks,report)


def test_world_xy_is_invariant_but_unsupported_joint_rejected(tensor_state):
    state=tensor_state.state
    support=ModelSupport(state_envelope(state),device="cpu")
    shifted=state.clone()
    shifted[:,:2]=100.
    support(shifted)
    shifted[:,13]=.1
    with pytest.raises(UnsupportedModelState):
        support(shifted)


def test_physical_degradation_is_not_overridden_by_model_return():
    criteria={"completion_drop":.1,"minimum_duration_ratio":.9,"minimum_activity_ratio":.8,"maximum_beat_drop":.1}
    def report(duration,complete,activity):
        return {"model_return":10000.,"physical_results":[{"task":"a","seed":1,"complete":complete,
            "physical_seconds":duration,"reason":"music_end" if complete else "physical_failure",
            "physical_metrics_mean":{"upper_eight_tracking":.9,"error/anchor_pos":.01,
                "root_xy_error_sq":.009,"error/anchor_ori":.001},
            "motion":{"joint_range_mean_rad":activity,"joint_speed_rms_rad_s":activity,"beat_alignment":.5}}]}
    quality=paired_quality(report(10,True,1),report(5,False,.5),criteria)
    assert quality["degraded"] and quality["model_return_used"] is False
    assert quality["quality_exit_enabled"] is False
