"""当前奖励中的实际活动度、执行器代价、接触和完整性门禁回归。

八项Tracking、0.7/0.3参考/实际限位和禁用音乐/stable由当前stage2_reward与
physical_reward专门验证；本文件只保留仍在当前实现使用的独立物理分项检查。
旧六项Tracking、独立稳定性、音乐评分及饱和限位公式的专用测试已经删除。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from gem.closedloop.dppo.rewards import DEFAULT_CONFIG, ExecutionReward, resolve_reward_config
from test_data_learning import actual_step, music_fixture, target_activity_fixture


def target(value):
    def fetch(tick):
        return {**target_activity_fixture(tick), "activity_rad_s": value}
    return fetch


def reward(value=1., config=None):
    return ExecutionReward(config, music_features=music_fixture(), target_activity=target(value))


def component(row, name, value=1.):
    result = reward(value).evaluate_step(row)
    assert result["transition_valid"], result["errors"]
    return result["components"][name]








def test_alive_failure_is_zero_without_hiding_terminal_event():
    row = actual_step(speed=1.)
    row["terminated"] = True
    result = reward().evaluate_step(row)
    assert result["transition_valid"]
    assert result["components"]["alive"]["score"] == 0
    assert reward().event_reward("physical_failure") == -5
    row["terminated"], row["truncated"] = False, True
    assert component(row, "alive")["score"] == 1
    row["completed_physics_steps"] = 3
    result = reward().evaluate_step(row)
    assert not result["transition_valid"]
    assert result["components"]["alive"]["score"] == 0






def test_activity_is_actual_velocity_rms_over_exact_matching_causal_window():
    calculator = reward()
    for index in range(30):
        row = actual_step(index, speed=0. if index < 5 else 2.)
        row["reference"]["joint_vel"][:] = 999.
        result = calculator.evaluate_step(row)
        assert result["transition_valid"], result["errors"]
        count = min(index + 1, 25)
        assert result["activity"]["window_count"] == count
        assert result["activity"]["window_complete"] == (count == 25)
        assert result["activity"]["window_begin_tick"] == row["tick"] - 12 * count
    assert result["activity"]["actual_activity_rad_s"] == 2.
    assert result["activity"]["target_activity_rad_s"] == 1.




@pytest.mark.parametrize("bad", ["absent", "window", "nonfinite", "no_source", "invalid"])
def test_missing_or_invalid_target_never_falls_back_to_generated_reference(bad):
    def invalid(tick):
        value = target_activity_fixture(tick)
        if bad == "window":
            value["window_begin_tick"] -= 12
        elif bad == "nonfinite":
            value["activity_rad_s"] = float("nan")
        elif bad == "no_source":
            value.pop("source")
        elif bad == "invalid":
            value["valid"] = False
        return value
    calculator = ExecutionReward(music_features=music_fixture(), target_activity=None if bad == "absent" else invalid)
    row = actual_step(speed=1.)
    row["reference"]["joint_vel"][:] = 1.
    result = calculator.evaluate_step(row)
    assert not result["transition_valid"] and not result["activity"]["valid"]
    # 配对数据仍须完整；活动诊断无效会使转移无效，但不再通过乘零改写跟踪分数。
    tracking=result["components"]["track"]
    assert tracking["gate"]==1.
    assert tracking["weighted_rate"]==pytest.approx(tracking["score"]*tracking["weight"])
    assert result["components"]["music"]["weighted_rate"] == 0


def test_cmd_uses_actual_target_control_dt_and_velocity_limits_not_reference():
    calculator = reward()
    first = actual_step(speed=1.)
    first["joint_position_target"][:] = .7
    value = calculator.evaluate_step(first)["components"]["cmd"]
    assert value["score"] == 0 and not value["valid"] and value["raw"]["first_step"]
    second = actual_step(1, speed=1.)
    second["joint_position_target"][:] = .72
    second["reference"]["joint_pos"][:] = -1.5
    value = calculator.evaluate_step(second)["components"]["cmd"]
    assert value["valid"] and not value["raw"]["first_step"]
    np.testing.assert_allclose(value["raw"]["cmd_rate_rad_s"], np.ones(21))
    assert value["score"] == pytest.approx(.01)
    calculator.reset(target_activity=target(1.))
    assert calculator.evaluate_step(first)["components"]["cmd"]["score"] == 0


@pytest.mark.parametrize("ratio,expected", [(.2, 0.), (.8, 0.), (.9, .25), (1., 1.), (2., 1.)])
def test_pd_torque_free_region_and_saturation_formula(ratio, expected):
    row = actual_step(speed=1.)
    for sample in row["physics_substeps"]:
        sample["physical_diagnostics"]["pd_torque_estimate_nm"][:] = 40 * ratio
    result = component(row, "torque")
    assert result["score"] == pytest.approx(expected)
    assert result["raw"]["max_torque_ratio"] == ratio
    assert "PD_torque_estimate" in result["raw"]["torque_semantics"]


def test_torque_averages_four_substeps_and_preserves_peak_ratio():
    row = actual_step(speed=1.)
    row["physics_substeps"][0]["physical_diagnostics"]["pd_torque_estimate_nm"][0] = 40
    row["physical_diagnostics"]["pd_torque_estimate_nm"][:] = 0.
    result = component(row, "torque")
    assert result["score"] == pytest.approx((.5 / 21 + .5) / 4)
    assert result["raw"]["max_torque_ratio"] == 1.


def test_power_records_each_joint_and_timestamps_without_reward():
    row = actual_step(speed=3.)
    for sample in row["physics_substeps"]:
        sample["physical_diagnostics"]["pd_torque_estimate_nm"][:] = 10.
    result = reward().evaluate_step(row)
    assert result["transition_valid"]
    diagnostic = result["diagnostics"]["power"]
    assert diagnostic["reward_weight"] == 0 and not diagnostic["sampling_synchronized"]
    assert diagnostic["mean_w"] == 30 and diagnostic["mean_sum_w"] == 630 and diagnostic["max_w"] == 30
    for raw, sample in zip(diagnostic["samples"], row["physics_substeps"]):
        assert raw["per_joint_w"] == [30.] * 21
        assert raw["torque_sample_tick"] == sample["physics_tick"] - 3
        assert raw["velocity_sample_tick"] == sample["physics_tick"]
        assert raw["time_s"] == sample["physics_tick"] / 600


def test_false_synchronized_power_claim_is_invalid():
    row = actual_step()
    row["physics_substeps"][0]["physical_diagnostics"]["mechanical_power_pd_estimate"]["sampling_synchronized"] = True
    result = reward().evaluate_step(row)
    assert not result["transition_valid"]
    assert any("synchronization" in error for error in result["errors"])


def test_contact_allows_elbows_and_uses_existing_bad_contact_threshold():
    row = actual_step(speed=1.)
    for sample in row["physics_substeps"]:
        sample["physical_diagnostics"]["net_contact_forces_w_n"][3:, 2] = 200.
    assert component(row, "contact")["score"] == 0
    row["physics_substeps"][0]["physical_diagnostics"]["net_contact_forces_w_n"][0, 2] = 1.01
    assert component(row, "contact")["score"] == pytest.approx(.3 / 4)


def test_foot_impact_is_diagnostic_only_without_uncalibrated_force_penalty():
    row = actual_step(speed=1.)
    for sample in row["physics_substeps"]:
        sample["physical_diagnostics"]["foot_net_contact_forces_w_n"][:, 2] = 10000.
    result = component(row, "contact")
    assert result["score"] == 0 and result["raw"]["impact_reward_weight"] == 0
    assert result["raw"]["samples"][0]["peak_foot_net_force_n"] == 10000.


def test_support_sphere_slide_requires_contact_and_geometric_ground_support():
    row = actual_step(speed=1.)
    for sample in row["physics_substeps"]:
        d = sample["physical_diagnostics"]
        d["foot_support_sphere_tangent_speed_m_s"] = {"left_foot": [.15, 99.], "right_foot": [99., 99.]}
        d["foot_support_sphere_clearance_m"] = {"left_foot": [0., 1.], "right_foot": [1., 1.]}
    result = component(row, "contact")
    assert result["score"] == pytest.approx(.7)
    assert result["raw"]["samples"][0]["supported_foot_count"] == 1
    for sample in row["physics_substeps"]:
        sample["physical_diagnostics"]["foot_net_contact_forces_w_n"][:] = 0.
    assert component(row, "contact")["score"] == 0


def test_link_slide_fallback_is_explicitly_named_proxy():
    row = actual_step(speed=1.)
    for sample in row["physics_substeps"]:
        sample["physical_diagnostics"]["body_link_lin_vel_w"][1:3, 0] = .075
    result = component(row, "contact")
    assert result["score"] == pytest.approx(.7 * .25)
    assert result["raw"]["samples"][0]["slide_velocity_semantics"] == "ankle_link_horizontal_velocity_slide_proxy"






@pytest.mark.parametrize("key", ["joint_vel_rms_rad_s", "root_lin_vel_rms_m_s", "root_ang_vel_rms_rad_s"])
def test_reference_consistency_is_integrity_gate_not_reward(key):
    row = actual_step(speed=1.)
    row["reference_consistency"][key] = 1e-4
    good = reward().evaluate_step(row)
    assert good["transition_valid"] and "consistency" not in good["components"]
    row["reference_consistency"][key] = 1.01e-4
    bad = reward().evaluate_step(row)
    assert not bad["transition_valid"]
    assert bad["reward"] == good["reward"]
    assert bad["diagnostics"]["consistency"]["raw"][key] == 1.01e-4
    assert any("construction error" in error for error in bad["errors"])


@pytest.mark.parametrize("overrides", [
    {"dt": .01}, {"version": "stage9.execution_reward.v1"}, {"failure_penalty": -5},
    {"diagnostics": {"mechanical_power_weight": .1}}, {"diagnostics": {"impact_weight": 1}},
    {"torque": {"free_ratio": 1}}, {"tracking": {"std": {"joint_pos": 0}}},
    {"activity": {"window_s": .51}}, {"tracking": {"weights": {"joint_pos": .9}}},
    {"scales": {"joint_pos_rad": .22}}, {"track_mix": {"joint_pos": .45}}, {"unexpected": 1}])
def test_config_is_validated_before_data_or_physics(overrides):
    with pytest.raises(ValueError):
        resolve_reward_config(overrides)


def test_resolved_config_is_independent_copy_and_all_components_preserve_evidence():
    config = resolve_reward_config({"track_weight": 3.})
    assert config["track_weight"] == 3. and DEFAULT_CONFIG["track_weight"] == 2.5
    result = reward().evaluate_step(actual_step(speed=1.))
    assert set(result["components"]) == {"track", "music", "stable", "alive", "cmd", "torque", "contact", "joint_limit"}
    for item in result["components"].values():
        assert 0 <= item["score"] <= 1
        assert isinstance(item["raw"], dict) and isinstance(item["normalized"], dict)
        assert item["weighted_rate"] == pytest.approx(item["score"] * item["weight"] * item["gate"])
        assert item["integrated_reward"] == pytest.approx(.02 * item["weighted_rate"])


def test_soft_limit_float_rounding_is_clamped_with_explicit_raw_evidence():
    row = actual_step(speed=1.)
    soft = row["physical_diagnostics"]["joint_pos_limits_rad"].copy()
    soft[:, 0] -= 5e-7
    soft[:, 1] += 5e-7
    row["physical_diagnostics"]["soft_joint_pos_limits_rad"] = soft
    result = component(row, "joint_limit")
    assert result["raw"]["numeric_boundary_clamped"]
    np.testing.assert_allclose(result["raw"]["original_safe_limits_rad"], soft)
    np.testing.assert_allclose(result["raw"]["safe_limits_rad"], row["physical_diagnostics"]["joint_pos_limits_rad"])
    soft[0, 0] -= 2e-6
    invalid = reward().evaluate_step(row)
    assert not invalid["transition_valid"]
    assert any("numeric tolerance" in error for error in invalid["errors"])
