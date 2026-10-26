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


def test_new_audit_does_not_widen_preserved_state_support(tmp_path):
    import json
    from gem.closedloop.mb_control.validation import training_support_envelope
    old={"frozen_sha256":"same","validated_state_envelope":{"schema":"same_coordinates",
        "minimum":[-1.,-2.],"maximum":[1.,2.],"real_state_count":10}}
    path=tmp_path/"prior.json";path.write_text(json.dumps(old))
    new={"frozen_sha256":"same","validated_state_envelope":{"schema":"same_coordinates",
        "minimum":[-3.,-1.],"maximum":[3.,4.],"real_state_count":20}}
    result=training_support_envelope({"imagination":{"state_support_reference_report":str(path)}},new)
    assert result["minimum"]==[-1.,-1.] and result["maximum"]==[1.,2.]


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
    before,after=report(10,True,1),report(10,True,1)
    before["physical_results"][0]["physical_metrics_mean"]["dance_loss"]=.01
    after["physical_results"][0]["physical_metrics_mean"]["dance_loss"]=.02
    quality=paired_quality(before,after,criteria)
    assert not quality["degraded"] and not quality["music_content_selection_eligible"]
    assert quality["quality_exit_enabled"] is False


def test_fixed_replay_checks_delivery_and_wall_clock_does_not_claim_paired_noise():
    from gem.closedloop.mb_control.dance_evaluation import paired_delivery_context
    before={"timing_mode":"fixed_delay_replay","noise_rule":"request_tick_seeded","noise_seed":42,
        "events":[{"request_tick":0,"actual_ready_tick":164,"commit_tick":168,"accepted":True}]}
    same=dict(before,events=[dict(before["events"][0])])
    assert paired_delivery_context(before,same)["exogenous_events_paired"]
    same["events"][0]["commit_tick"]=180
    with pytest.raises(ValueError):paired_delivery_context(before,same)
    before["timing_mode"]=same["timing_mode"]="wall_clock_async"
    same["events"].append({"request_tick":300,"actual_ready_tick":420,"commit_tick":432,"accepted":True})
    result=paired_delivery_context(before,same)
    assert not result["exogenous_events_paired"] and not result["same_request_noise_identity"]
