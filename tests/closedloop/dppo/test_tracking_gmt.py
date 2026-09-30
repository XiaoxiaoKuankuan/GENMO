"""Stage9 Tracking 与冻结 GMT motion-tracking 诊断契约的 CPU 回归。

本文件检查六项平方误差到 exp(-error/std²) 的解析映射、归一化混合权重、默认关闭
但可显式启用的关节速度项，以及 body/joint 名称、索引、有限数组和同 tick 证据。
这里不实现另一套四元数或参考坐标转换：合成诊断代表 backend 已完成的原 GMT
函数计算，真实四元数跨 ±π、MotionCommand 选择和冻结检查由 GMT producer 测试
直接对照原函数验证。另检查改变 Tracking 时其余 v2 奖励及活动门控保持逐值相同。

测试只运行 CPU，全部 fixture 在内存中构造，不写 checkpoint、数据集或正式日志；
统一 pytest 命令关闭 bytecode/cache，临时输出由调用方在结束时自动删除。
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import yaml

from gem.closedloop.contracts import GMT_EXPECTED_JOINT_ORDER
from gem.closedloop.dppo.rewards import DEFAULT_CONFIG, resolve_reward_config
from test_data_learning import actual_step
from test_rewards_v2 import reward


TERMS = ("anchor_pos", "anchor_ori", "body_pos", "body_ori", "joint_pos", "joint_vel")


def evaluate(row, config=None):
    result = reward(config=config).evaluate_step(row)
    assert result["transition_valid"], result["errors"]
    return result


def test_default_tracking_parameters_preserve_outer_weight_and_gmt_proportions():
    config = resolve_reward_config()
    assert config["track_weight"] == 2.5
    assert config["tracking"]["objective"] == "gmt.motion_tracking.v1"
    assert config["tracking"]["std"] == dict(anchor_pos=.3, anchor_ori=.4, body_pos=.3,
                                             body_ori=.4, joint_pos=.25, joint_vel=1.4)
    assert config["tracking"]["weights"] == pytest.approx(dict(
        anchor_pos=1 / 7, anchor_ori=1 / 7, body_pos=2 / 7,
        body_ori=2 / 7, joint_pos=1 / 7, joint_vel=0.))
    assert "track_mix" not in config
    assert set(config["scales"]) == {"root_height_m", "non_yaw_rad", "slide_m_s"}


@pytest.mark.parametrize("filename", ["stage9_dppo_smoke.yaml", "stage9_dppo_server1.yaml",
                                       "stage10_prepare_server1.yaml", "stage10_formal_server1.yaml"])
def test_training_configurations_share_the_same_explicit_tracking_objective(filename):
    path = Path(__file__).resolve().parents[3] / "configs" / "closedloop" / filename
    stage = "stage10" if filename.startswith("stage10_") else "stage9"
    config = yaml.safe_load(path.read_text())[stage]["reward"]
    assert config["tracking"] == DEFAULT_CONFIG["tracking"]
    assert resolve_reward_config(config)["track_weight"] == 2.5


def test_zero_error_each_tracking_score_and_total_are_one():
    item = evaluate(actual_step(speed=1.))["components"]["track"]
    assert item["score"] == pytest.approx(1.)
    assert set(item["normalized"]["scores"]) == set(TERMS)
    assert all(value == pytest.approx(1.) for value in item["normalized"]["scores"].values())
    assert all(value == 0 for value in item["normalized"]["error_over_std_squared"].values())
    assert item["weighted_rate"] == pytest.approx(2.5)
    assert item["integrated_reward"] == pytest.approx(.05)


@pytest.mark.parametrize("key", TERMS)
def test_squared_error_grows_monotonically_without_squaring_it_twice(key):
    std = DEFAULT_CONFIG["tracking"]["std"][key]
    weight = DEFAULT_CONFIG["tracking"]["weights"][key]
    scores, totals = [], []
    for multiple in (0., .25, 1., 4., 16.):
        row = actual_step(speed=1.)
        row["motion_tracking"]["terms"][key]["error"] = multiple * std ** 2
        item = evaluate(row)["components"]["track"]
        scores.append(item["normalized"]["scores"][key])
        totals.append(item["score"])
        assert scores[-1] == pytest.approx(math.exp(-multiple))
        assert totals[-1] == pytest.approx(1 - weight + weight * math.exp(-multiple))
    assert all(first > second for first, second in zip(scores, scores[1:]))
    if weight:
        assert all(first > second for first, second in zip(totals, totals[1:]))
    else:
        assert totals == pytest.approx([1.] * len(totals))


def test_joint_velocity_optional_weight_can_be_enabled_without_changing_backend_reference():
    row = actual_step(speed=1.)
    row["motion_tracking"]["terms"]["joint_vel"]["error"] = 1.4 ** 2
    off = evaluate(row)["components"]["track"]
    weights = {key: 0. for key in TERMS}
    weights["joint_vel"] = 1.
    on = evaluate(row, {"tracking": {"weights": weights}})["components"]["track"]
    assert off["score"] == pytest.approx(1.)
    assert on["score"] == pytest.approx(math.exp(-1.))
    assert not row["motion_tracking"]["terms"]["joint_vel"]["source_enabled"]
    assert on["normalized"]["weights"]["joint_vel"] == 1.


def test_native_joint_order_and_original_left_then_right_leg_selection():
    row = actual_step(speed=1.)
    diagnostic = row["motion_tracking"]
    assert diagnostic["joint_names"] == list(GMT_EXPECTED_JOINT_ORDER)
    selection = diagnostic["terms"]["joint_pos"]
    assert selection["joint_indices"] == [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20]
    assert selection["joint_names"] == [GMT_EXPECTED_JOINT_ORDER[i] for i in selection["joint_indices"]]
    assert selection["source_term"] == "motion_leg_joint_pos"
    for key in ("body_pos", "body_ori"):
        assert diagnostic["terms"][key]["body_indices"] == list(range(22))
        assert diagnostic["terms"][key]["body_names"] == diagnostic["body_names"]
    evaluate(row)


@pytest.mark.parametrize("bad", [
    "missing", "schema", "coordinate_frame", "quaternion_order", "reference_tick", "control_tick", "native_joint_order",
    "body_order", "joint_selection_names", "body_selection_names", "joint_selection_index",
    "duplicate_body_names", "missing_body_reference", "body_reference_shape",
    "nonfinite_body_reference", "nonfinite_joint_actual", "missing_term", "wrong_function",
    "negative_error", "nonfinite_error",
])
def test_missing_malformed_or_misordered_tracking_evidence_is_invalid(bad):
    row = actual_step(speed=1.)
    diagnostic = row["motion_tracking"]
    if bad == "missing":
        row.pop("motion_tracking")
    elif bad == "schema":
        diagnostic["schema"] = "unknown"
    elif bad == "coordinate_frame":
        diagnostic["coordinate_frame"] = "anchor_local"
    elif bad == "quaternion_order":
        diagnostic["quaternion_order"] = "xyzw"
    elif bad in ("reference_tick", "control_tick"):
        diagnostic[bad] -= 12
    elif bad == "native_joint_order":
        diagnostic["joint_names"] = diagnostic["joint_names"][::-1]
    elif bad == "body_order":
        diagnostic["body_names"] = diagnostic["body_names"][::-1]
    elif bad == "joint_selection_names":
        diagnostic["terms"]["joint_pos"]["joint_names"] = diagnostic["terms"]["joint_pos"]["joint_names"][::-1]
    elif bad == "body_selection_names":
        diagnostic["terms"]["body_ori"]["body_names"] = diagnostic["terms"]["body_ori"]["body_names"][::-1]
    elif bad == "joint_selection_index":
        diagnostic["terms"]["joint_pos"]["joint_indices"][0] = 21
    elif bad == "duplicate_body_names":
        diagnostic["body_names"][1] = diagnostic["body_names"][0]
    elif bad == "missing_body_reference":
        diagnostic["reference"].pop("body_quat_relative_w")
    elif bad == "body_reference_shape":
        diagnostic["reference"]["body_pos_relative_w"] = diagnostic["reference"]["body_pos_relative_w"][:-1]
    elif bad == "nonfinite_body_reference":
        diagnostic["reference"]["body_quat_relative_w"][0, 0] = float("nan")
    elif bad == "nonfinite_joint_actual":
        diagnostic["actual"]["joint_vel"][0] = float("inf")
    elif bad == "missing_term":
        diagnostic["terms"].pop("joint_vel")
    elif bad == "wrong_function":
        diagnostic["terms"]["body_ori"]["function"] = "motion_global_body_orientation_error_exp"
    elif bad == "negative_error":
        diagnostic["terms"]["body_ori"]["error"] = -.01
    elif bad == "nonfinite_error":
        diagnostic["terms"]["body_ori"]["error"] = float("nan")
    result = reward().evaluate_step(row)
    assert not result["transition_valid"]
    assert not result["components"]["track"]["valid"]
    assert result["components"]["track"]["score"] == 0
    assert result["errors"]


def test_replacing_tracking_does_not_change_other_reward_terms_or_activity_gate():
    original = actual_step(speed=.2)
    original["errors"]["root_height_error_m"] = .02
    original["errors"]["non_yaw_orientation_error_rad"] = .03
    changed = copy.deepcopy(original)
    for index, key in enumerate(TERMS):
        changed["motion_tracking"]["terms"][key]["error"] = .1 * (index + 1)
    before, after = evaluate(original), evaluate(changed)
    assert before["activity"] == after["activity"]
    assert before["diagnostics"] == after["diagnostics"]
    for key in set(before["components"]) - {"track"}:
        assert before["components"][key] == after["components"][key]
    difference = after["components"]["track"]["integrated_reward"] - before["components"]["track"]["integrated_reward"]
    assert after["reward"] - before["reward"] == pytest.approx(difference)


@pytest.mark.parametrize("overrides", [
    {"tracking": {"objective": "legacy.rmse"}},
    {"tracking": {"std": {"anchor_ori": float("nan")}}},
    {"tracking": {"std": {"body_pos": -1.}}},
    {"tracking": {"std": {"anchor_pos": 1e-200}}},
    {"tracking": {"std": {"anchor_pos": 1e200}}},
    {"tracking": {"weights": {"joint_vel": -.1}}},
    {"tracking": {"weights": {key: 0. for key in TERMS}}},
    {"tracking": {"weights": {"extra": .1}}},
])
def test_tracking_configuration_rejects_ambiguous_or_invalid_objective(overrides):
    with pytest.raises(ValueError):
        resolve_reward_config(overrides)
