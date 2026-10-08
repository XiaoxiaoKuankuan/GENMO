"""第九步第二版实际执行奖励、活动监督与物理诊断。

本模块只评价已经执行的 50Hz 控制区间。跟踪消费冻结 GMT 原 motion-tracking 六项
平方误差：世界 anchor 位置/完整姿态、原 MotionCommand 对齐后的 body 位置/姿态、
原任务选择的关节位置及可选关节速度。使用 exp(-error/std²)，不再次平方 error，
不重新构造坐标系或用旧 RMSE/末端高度/yaw 混合项回退；名称、选择、时刻和原值
证据随 backend 诊断保存。内部 objective 版本独立标识这次 tracking 语义变化。
稳定性仍直接消费原 backend 的根高与 non-yaw 误差，其余 v2 奖励公式保持不变。
活动门控同时作用于跟踪和音乐；目标活动度来自配对训练动作的同一因果时间窗，
绝不使用 GENMO 生成参考代替监督。音乐节拍复用 BUMI 原有动作节拍和对齐函数。
连续奖励率统一乘 0.02 秒，失败及有限非法参考的事件罚分由调用方单独加入一次。

PD 目标变化按真实生效目标计算；力矩使用未裁剪的隐式 PD torque estimate，在
四个 200Hz 子步分别归一化后平均。机械功率和接触冲击仅做诊断；力矩与速度不同
采样时刻会明确标记，不把代理称为硬件力矩、电功率或能耗。接触允许脚与肘，
优先使用支撑球附近速度；只能获得 link 速度时明确保留 slide_proxy 名称。

所有分数和连续代价有界于 [0,1]，输出同时保存原始量、归一化量、分项积分及最终
奖励。参考位置—速度一致性属于数据完整性门禁，不属于策略奖励；超出容差或
缺少必需证据时标记 transition_valid=false。活动窗口尚不足 0.5 秒时使用实际已有
的因果子窗，并严格要求目标监督使用完全相同的窗口，显式记录 window_complete。
"""
from __future__ import annotations

from collections import deque
import copy
import math

import numpy as np
import torch

from gem.robots.bumi.metrics import _derive_motion_beats, _beat_alignment
from gem.closedloop.contracts import GMT_EXPECTED_JOINT_ORDER
from .performance import profiled


TRACKING_OBJECTIVE = "gmt.motion_tracking.v1"
TRACKING_FUNCTIONS = {
    "anchor_pos": "motion_global_anchor_position_error_exp",
    "anchor_ori": "motion_global_anchor_orientation_error_exp",
    "body_pos": "motion_relative_body_position_error_exp",
    "body_ori": "motion_relative_body_orientation_error_exp",
    "joint_pos": "motion_joint_position_error_exp",
    "joint_vel": "motion_joint_velocity_error_exp",
}
TRACKING_ERROR_FIELDS = {
    "anchor_pos": "anchor_position_error_sq_m2",
    "anchor_ori": "anchor_orientation_error_sq_rad2",
    "body_pos": "body_position_mean_error_sq_m2",
    "body_ori": "body_orientation_mean_error_sq_rad2",
    "joint_pos": "joint_position_mean_error_sq_rad2",
    "joint_vel": "joint_velocity_mean_error_sq_rad2_s2",
}


DEFAULT_CONFIG = {
    "version": "stage9.execution_reward.v2",
    "dt": .02,
    "track_weight": 2.5, "music_weight": 2., "stable_weight": 1., "alive_weight": .5,
    "cmd_penalty_weight": .15, "torque_penalty_weight": .10,
    "contact_penalty_weight": .20, "joint_limit_penalty_weight": .50,
    "failure_penalty": 5., "rejected_plan_penalty": .5,
    "scales": {"root_height_m": .05,
               "non_yaw_rad": .12, "slide_m_s": .15},
    "tracking": {
        "objective": TRACKING_OBJECTIVE,
        "std": {"anchor_pos": .3, "anchor_ori": .4, "body_pos": .3,
                "body_ori": .4, "joint_pos": .25, "joint_vel": 1.4},
        # 对应当前 GMT 启用的 .5/.5/1/1/.5 比例；joint_vel 原任务未启用。
        "weights": {"anchor_pos": 1/7, "anchor_ori": 1/7, "body_pos": 2/7,
                    "body_ori": 2/7, "joint_pos": 1/7, "joint_vel": 0.},
    },
    "stable_mix": {"root_height": .5, "non_yaw": .5},
    "music_mix": {"beat": .7, "intensity": .3},
    "torque": {"free_ratio": .8, "mean_weight": .5, "max_weight": .5},
    "activity": {"window_s": .5, "inactive_target_rad_s": .10, "full_gate_ratio": .5,
                 "epsilon": 1e-8, "intensity_epsilon": .05, "intensity_log_ratio": 2.},
    "music": {"beat_window_s": 1., "feature_fps": 30, "beat_column": 34, "beat_threshold": .5},
    "contact": {"slide_weight": .7, "bad_contact_weight": .3, "support_clearance_m": .025},
    "joint_limit": {"hard_range_margin_fraction": .05, "numeric_tolerance_rad": 1e-6,
                    "actual_weight": .5, "reference_weight": .5},
    "consistency": {"joint_vel_rms_rad_s": 1e-4, "root_lin_vel_rms_m_s": 1e-4,
                    "root_ang_vel_rms_rad_s": 1e-4},
    "diagnostics": {"mechanical_power_weight": 0., "impact_weight": 0.},
    "require_substeps": True,
}


def _array(value, name, shape=None):
    if value is None:
        raise ValueError(f"missing {name}")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} requires shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"nonfinite {name}")
    return result


def _scalar(value, name):
    array = _array(value, name)
    if array.ndim:
        raise ValueError(f"{name} must be scalar")
    return float(array)


def _merge(destination, overrides, prefix="reward"):
    for key, value in overrides.items():
        if key not in destination:
            raise ValueError(f"unknown {prefix} setting {key}")
        if isinstance(destination[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}.{key} must be a mapping")
            _merge(destination[key], value, f"{prefix}.{key}")
        else:
            destination[key] = value


def _score(error, scale):
    normalized = float(error) / float(scale)
    if not math.isfinite(normalized):
        raise ValueError("nonfinite normalized reward error")
    # 指数下溢的极大误差仍具有正确的零分，不改变原始误差或概率。
    return math.exp(-min(abs(normalized), 1e150) ** 2), normalized


class ExecutionReward:
    def __init__(self, config=None, music_features=None, music_start_tick=600, target_activity=None):
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        _merge(self.config, config or {})
        self._validate_config()
        self.music = self._music_array(music_features)
        self.music_start_tick = int(music_start_tick)
        self.target_activity = target_activity
        self._cache_music_beats()
        self.activity_steps = int(round(self.config["activity"]["window_s"] / self.config["dt"]))
        self.beat_steps = int(round(self.config["music"]["beat_window_s"] / self.config["dt"]))
        self.window = deque(maxlen=max(self.activity_steps, self.beat_steps))
        self._last_tick = self._episode = self._previous_target = None

    def _cache_music_beats(self):
        """每次绑定音乐或起点后计算一次，保持原30Hz到50Hz的round映射。"""
        cfg = self.config['music']
        source = (np.empty(0, dtype=np.int64) if self.music is None else
                  np.flatnonzero(self.music[:, cfg['beat_column']] > cfg['beat_threshold']))
        self._beat_ticks = self.music_start_tick + np.rint(source * 20 / 12).astype(np.int64) * 12
        self._beat_ticks.setflags(write=False)

    def seed_previous_target(self, target):
        """奖励开始前承接预热最后一个实际PD目标，不把预热加入音乐活动窗。"""
        if self._last_tick is not None or self._previous_target is not None:
            raise ValueError('previous target can only be seeded before the first reward step')
        self._previous_target = _array(target, 'previous actual submitted joint target', (21,)).copy()

    def _validate_config(self):
        config = self.config
        if config["version"] != "stage9.execution_reward.v2":
            raise ValueError("reward version must be stage9.execution_reward.v2")
        if config["dt"] != .02:
            raise ValueError("Stage9 rewards require dt=0.02 for actual 50Hz controls")
        def finite_nonnegative(value, name, *, positive=False):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
                raise ValueError(f"invalid reward setting {name}")
        for name, value in config.items():
            if name.endswith("weight") or name.endswith("penalty"):
                finite_nonnegative(value, name)
        for group in ("scales", "consistency"):
            for name, value in config[group].items():
                finite_nonnegative(value, f"{group}.{name}", positive=True)
        tracking = config["tracking"]
        if tracking["objective"] != TRACKING_OBJECTIVE:
            raise ValueError(f"tracking.objective must be {TRACKING_OBJECTIVE}")
        for name, value in tracking["std"].items():
            finite_nonnegative(value, f"tracking.std.{name}", positive=True)
            squared = float(value) * float(value)
            if not math.isfinite(squared) or squared <= 0:
                raise ValueError(f"tracking.std.{name} squared must be finite and positive")
        for name, value in tracking["weights"].items():
            finite_nonnegative(value, f"tracking.weights.{name}")
        if not math.isclose(sum(tracking["weights"].values()), 1., abs_tol=1e-12):
            raise ValueError("tracking.weights must sum to one")
        for group in ("stable_mix", "music_mix"):
            for name, value in config[group].items():
                finite_nonnegative(value, f"{group}.{name}")
            if not math.isclose(sum(config[group].values()), 1., abs_tol=1e-12):
                raise ValueError(f"{group} must sum to one")
        for group, keys in (("torque", ("mean_weight", "max_weight")),
                            ("contact", ("slide_weight", "bad_contact_weight")),
                            ("joint_limit", ("actual_weight", "reference_weight"))):
            for key in keys:
                finite_nonnegative(config[group][key], f"{group}.{key}")
            if not math.isclose(sum(config[group][key] for key in keys), 1., abs_tol=1e-12):
                raise ValueError(f"{group} mixing weights must sum to one")
        if not 0 <= config["torque"]["free_ratio"] < 1:
            raise ValueError("torque.free_ratio must be in [0,1)")
        if not 0 < config["joint_limit"]["hard_range_margin_fraction"] < .5:
            raise ValueError("joint limit margin fraction must be in (0,0.5)")
        finite_nonnegative(config["joint_limit"]["numeric_tolerance_rad"], "joint_limit.numeric_tolerance_rad")
        for name, value in config["activity"].items():
            finite_nonnegative(value, f"activity.{name}", positive=name != "inactive_target_rad_s")
        if config["activity"]["intensity_log_ratio"] <= 1:
            raise ValueError("activity.intensity_log_ratio must exceed one")
        for seconds in (config["activity"]["window_s"], config["music"]["beat_window_s"]):
            if not math.isfinite(seconds) or seconds < config["dt"] or not math.isclose(seconds / config["dt"], round(seconds / config["dt"]), abs_tol=1e-9):
                raise ValueError("music/activity window must contain an integer number of controls")
        if config["music"]["feature_fps"] != 30 or config["music"]["beat_column"] != 34:
            raise ValueError("music feature contract requires EDGE35 30Hz beat column 34")
        finite_nonnegative(config["music"]["beat_threshold"], "music.beat_threshold")
        finite_nonnegative(config["contact"]["support_clearance_m"], "contact.support_clearance_m")
        if any(config["diagnostics"].values()):
            raise ValueError("mechanical power and impact are diagnostics only; weights must stay zero")
        if not isinstance(config["require_substeps"], bool):
            raise ValueError("require_substeps must be boolean")

    @staticmethod
    def _music_array(features):
        if features is None:
            return None
        music = _array(features, "music_features").copy()
        if music.ndim != 2 or music.shape[1] != 35:
            raise ValueError("music_features must be EDGE35 [T,35]")
        return music

    def reset(self, *, music_features=None, music_start_tick=None, target_activity=None):
        if music_features is not None:
            self.music = self._music_array(music_features)
        if music_start_tick is not None:
            self.music_start_tick = int(music_start_tick)
        # 换 episode 后不允许无意沿用上一首配对动作监督。
        self.target_activity = target_activity
        self._cache_music_beats()
        self.window.clear()
        self._last_tick = self._episode = self._previous_target = None

    def event_reward(self, reason):
        if reason in {"reference_rejected", "plan_rejected", "rejected"}:
            return -float(self.config["rejected_plan_penalty"])
        if reason in {"physical_failure", "task_failure", "reference_exhausted"}:
            return -float(self.config["failure_penalty"])
        if reason in {None, "music_end", "task_complete", "collection_limit", "rollout_truncated", "infrastructure_failure"}:
            return 0.
        raise ValueError(f"event must use an explicit Stage9 reason category, got {reason}")

    def _activity(self, tick):
        if not callable(self.target_activity):
            raise ValueError("A_target not_available: paired activity supervision is required")
        rows = list(self.window)[-self.activity_steps:]
        expected_count = min(self.activity_steps, (tick - self.music_start_tick) // 12)
        if expected_count <= 0 or len(rows) != expected_count:
            raise ValueError("actual activity history does not cover the matching causal target window")
        speed = np.stack([row[2] for row in rows])
        actual = _scalar(np.sqrt(np.mean(speed ** 2)), "actual activity RMS")
        target = self.target_activity(int(tick))
        if not isinstance(target, dict) or target.get("valid") is not True:
            raise ValueError("A_target not_available or invalid")
        target_value = _scalar(target.get("activity_rad_s"), "A_target.activity_rad_s")
        if target_value < 0:
            raise ValueError("A_target activity must be nonnegative")
        expected_begin = int(rows[0][0] - 12)
        if (target.get("window_count") != len(rows) or target.get("window_begin_tick") != expected_begin
                or target.get("window_end_tick") != tick or target.get("window_complete") != (len(rows) == self.activity_steps)):
            raise ValueError("A_target and actual activity windows differ")
        if not target.get("source"):
            raise ValueError("A_target requires paired-data source identity")
        cfg = self.config["activity"]
        ratio = actual / (cfg["full_gate_ratio"] * target_value + cfg["epsilon"])
        gate = 1. if target_value <= cfg["inactive_target_rad_s"] else min(1., ratio)
        log_ratio = math.log((actual + cfg["intensity_epsilon"]) / (target_value + cfg["intensity_epsilon"])) / math.log(cfg["intensity_log_ratio"])
        intensity = math.exp(-min(abs(log_ratio), 1e150) ** 2)
        return gate, intensity, {"actual_activity_rad_s": actual, "target_activity_rad_s": target_value,
            "window_count": len(rows), "window_complete": len(rows) == self.activity_steps,
            "window_begin_tick": expected_begin, "window_end_tick": int(tick),
            "target_source": copy.deepcopy(target["source"]), "target_evidence": copy.deepcopy(target),
            "valid": True, "gate": gate, "gate_unclamped_ratio": ratio, "intensity_log_ratio": log_ratio,
            "intensity_score": intensity}

    def _music(self, intensity, activity):
        if self.music is None:
            raise ValueError("music features not_available")
        rows = list(self.window)[-self.beat_steps:]
        beat, beat_valid = 0., False
        raw = {"beat_definition": "bumi.metrics._derive_motion_beats+_beat_alignment",
               "beat_window_count": len(rows), "beat_window_complete": len(rows) == self.beat_steps,
               "beat_valid": False, "intensity_valid": True, "activity": copy.deepcopy(activity)}
        if len(rows) < self.beat_steps:
            raw["beat_reason"] = "insufficient_causal_history"
        else:
            ticks = np.asarray([row[0] for row in rows])
            if not ((ticks >= self.music_start_tick) & (ticks <= self.music_start_tick + len(self.music) * 20)).all():
                raise ValueError("music outside paired task")
            music_beats = torch.from_numpy(np.isin(ticks, self._beat_ticks))[None]
            speeds = np.stack([np.abs(row[2]) for row in rows])
            valid = torch.ones((1, len(rows)), dtype=torch.bool)
            motion_beats = _derive_motion_beats(torch.from_numpy(speeds)[None], valid)
            raw.update(music_beat_count=int(music_beats.sum()), motion_beat_count=int(motion_beats.sum()))
            if not raw["music_beat_count"]:
                raw["beat_reason"] = "no_music_beats"
            else:
                beat_valid = True
                _, alignment = _beat_alignment(music_beats, motion_beats, valid, round(1 / self.config["dt"]))
                beat = float(alignment)
        raw.update(beat_valid=beat_valid, beat_alignment=beat, intensity_alignment=intensity)
        score = self.config["music_mix"]["beat"] * beat + self.config["music_mix"]["intensity"] * intensity
        return score, True, raw, {"beat": beat, "intensity": intensity}

    def _track(self, row):
        """消费 GMT 原误差和选择证据；这里不重新对齐参考、不从评分反推误差。"""
        diagnostic = row.get("motion_tracking")
        if not isinstance(diagnostic, dict) or diagnostic.get("schema") != TRACKING_OBJECTIVE:
            raise ValueError("missing or incompatible frozen GMT motion_tracking diagnostics")
        if diagnostic.get("coordinate_frame") != "world" or diagnostic.get("quaternion_order") != "wxyz":
            raise ValueError("motion_tracking must retain GMT world coordinates and wxyz quaternions")
        if diagnostic.get("control_tick") != row["tick"] or diagnostic.get("reference_tick") != row["tick"]:
            raise ValueError("motion_tracking must use the actually consumed control/reference tick")
        body_names, joint_names = diagnostic.get("body_names"), diagnostic.get("joint_names")
        if not isinstance(body_names, (list, tuple)) or not body_names or any(not isinstance(n, str) or not n for n in body_names):
            raise ValueError("motion_tracking requires named MotionCommand bodies")
        if len(set(body_names)) != len(body_names) or diagnostic.get("anchor_body_name") not in body_names:
            raise ValueError("motion_tracking body order contains duplicates or lacks its anchor")
        if not isinstance(joint_names, (list, tuple)) or tuple(joint_names) != GMT_EXPECTED_JOINT_ORDER:
            raise ValueError("motion_tracking joint order must match frozen GMT native order")
        for side in ("reference", "actual"):
            values = diagnostic.get(side)
            if not isinstance(values, dict):
                raise ValueError(f"motion_tracking requires {side} state evidence")
            for field, shape in (("anchor_pos_w", (3,)), ("anchor_quat_w", (4,)),
                                 ("body_pos_w", (len(body_names), 3)), ("body_quat_w", (len(body_names), 4)),
                                 ("joint_pos", (len(joint_names),)), ("joint_vel", (len(joint_names),))):
                _array(values.get(field), f"motion_tracking.{side}.{field}", shape)
        for field, shape in (("body_pos_relative_w", (len(body_names), 3)), ("body_quat_relative_w", (len(body_names), 4))):
            _array(diagnostic["reference"].get(field), f"motion_tracking.reference.{field}", shape)
        terms = diagnostic.get("terms")
        if not isinstance(terms, dict) or set(terms) != set(TRACKING_FUNCTIONS):
            raise ValueError("motion_tracking requires all six original GMT terms")
        raw, normalized, scores = {}, {}, {}
        for name, function in TRACKING_FUNCTIONS.items():
            term = terms[name]
            if not isinstance(term, dict) or term.get("function") != function:
                raise ValueError(f"motion_tracking.{name} is not the original GMT function")
            if name.startswith(("body_", "joint_")):
                kind = "body" if name.startswith("body_") else "joint"
                names = body_names if kind == "body" else joint_names
                indices, selected = term.get(f"{kind}_indices"), term.get(f"{kind}_names")
                if (not isinstance(indices, (list, tuple)) or not indices
                        or any(type(i) is not int or not 0 <= i < len(names) for i in indices)
                        or len(set(indices)) != len(indices)
                        or list(selected or []) != [names[i] for i in indices]):
                    raise ValueError(f"motion_tracking.{name} selected {kind} names/order differ from indices")
            error = _scalar(term.get("error"), f"motion_tracking.{name}.error")
            if error < 0:
                raise ValueError(f"motion_tracking.{name} squared error must be nonnegative")
            std = self.config["tracking"]["std"][name]
            normalized[name] = error / std ** 2
            if not math.isfinite(normalized[name]):
                raise ValueError(f"nonfinite motion_tracking.{name} normalized error")
            scores[name] = math.exp(-normalized[name])
            raw[TRACKING_ERROR_FIELDS[name]] = error
        raw.update(source="frozen_gmt_current_motion_command", objective=TRACKING_OBJECTIVE,
                   anchor_body_name=diagnostic["anchor_body_name"], body_names=list(body_names),
                   joint_names=list(joint_names), terms=copy.deepcopy(terms), control_tick=int(row["tick"]))
        cfg = self.config["tracking"]
        score = sum(cfg["weights"][name] * value for name, value in scores.items())
        return score, True, raw, {"error_over_std_squared": normalized, "scores": scores,
                                 "std": copy.deepcopy(cfg["std"]), "weights": copy.deepcopy(cfg["weights"])}

    def _stable(self, row):
        raw, normalized, scores = {}, {}, {}
        for name, key, scale in (("root_height", "root_height_error_m", "root_height_m"),
                                 ("non_yaw", "non_yaw_orientation_error_rad", "non_yaw_rad")):
            raw[key] = _scalar(row["errors"].get(key), key)
            scores[name], normalized[name] = _score(raw[key], self.config["scales"][scale])
        return sum(self.config["stable_mix"][name] * score for name, score in scores.items()), True, raw, {"errors_over_scale": normalized, "scores": scores}

    def _alive(self, row):
        complete = row.get("completed_physics_steps", 4) == 4
        valid = row.get("transition_valid", True) and row.get("state_valid", True) and complete
        failed = bool(row.get("terminated", False))
        return float(valid and not failed), bool(valid), {"complete_control_interval": complete, "execution_valid": bool(valid), "execution_failed": failed}, {}

    def _physical_samples(self, row):
        samples = row.get("physics_substeps")
        if samples is None:
            if self.config["require_substeps"]:
                raise ValueError("four actual physical substeps are required")
            return [{"physical_diagnostics": row["physical_diagnostics"], "joint_vel_gmt": row["actual_joint_vel_gmt"],
                     "joint_position_target": row.get("joint_position_target"), "physics_tick": row["tick"], "dt_s": self.config["dt"]}], "control_end_proxy"
        if len(samples) != 4:
            raise ValueError("normal control interval requires four actual physical substeps")
        for index, sample in enumerate(samples):
            if abs(_scalar(sample.get("dt_s"), "substep dt") - self.config["dt"] / 4) > 1e-12:
                raise ValueError("invalid physical substep duration")
            if sample.get("physics_tick") != row["tick"] - 12 + (index + 1) * 3:
                raise ValueError("physical sample timestamp does not match executed control interval")
        return samples, "four_physics_substeps"

    def _cmd(self, row):
        target = _array(row.get("joint_position_target"), "joint_position_target", (21,))
        limit = _array(row["physical_diagnostics"].get("joint_velocity_limits_rad_s"), "joint velocity limits", (21,))
        if (limit <= 0).any():
            raise ValueError("joint velocity limits must be positive")
        first = self._previous_target is None
        rate = np.zeros(21) if first else (target - self._previous_target) / self.config["dt"]
        previous = None if first else self._previous_target.tolist()
        normalized = _array(np.abs(rate) / limit, "command rate normalized by velocity limit", (21,))
        per_joint = np.clip(normalized ** 2, 0., 1.)
        self._previous_target = target.copy()
        return float(per_joint.mean()), not first, {"joint_position_target_rad": target.tolist(), "previous_joint_position_target_rad": previous,
            "cmd_rate_rad_s": rate.tolist(), "joint_velocity_limits_rad_s": limit.tolist(), "first_step": first}, {"absolute_rate_over_limit": normalized.tolist(), "per_joint_cost": per_joint.tolist()}

    def _torque(self, row):
        samples, semantics = self._physical_samples(row)
        costs, ratios, details = [], [], []
        cfg = self.config["torque"]
        for sample in samples:
            diagnostic = sample["physical_diagnostics"]
            torque = _array(diagnostic.get("pd_torque_estimate_nm"), "PD torque estimate", (21,))
            limits = _array(diagnostic.get("joint_effort_limits_nm"), "joint effort limits", (21,))
            if (limits <= 0).any():
                raise ValueError("joint effort limits must be positive")
            ratio = _array(np.abs(torque) / limits, "PD torque normalized by effort limit", (21,))
            cost = np.clip((ratio - cfg["free_ratio"]) / (1 - cfg["free_ratio"]), 0., 1.) ** 2
            value = cfg["mean_weight"] * float(cost.mean()) + cfg["max_weight"] * float(cost.max())
            costs.append(value)
            ratios.append(ratio.tolist())
            details.append({"physics_tick": sample["physics_tick"], "time_s": sample["physics_tick"] / 600,
                "pd_torque_estimate_nm": torque.tolist(), "joint_effort_limits_nm": limits.tolist(),
                "per_joint_cost": cost.tolist(), "cost": value})
        return float(np.mean(costs)), True, {"sampling": semantics, "torque_semantics": "implicit_PD_torque_estimate_not_hardware_torque",
            "samples": details, "max_torque_ratio": float(np.max(ratios))}, {"absolute_torque_over_effort_limit": ratios, "substep_costs": costs}

    def _contact(self, row):
        samples, semantics = self._physical_samples(row)
        details, slides, bads = [], [], []
        for sample in samples:
            d = sample["physical_diagnostics"]
            feet, names = d["foot_body_names"], d["contact_body_names"]
            allowed = d.get("allowed_contact_body_names")
            if not isinstance(allowed, (list, tuple)) or not set(feet).issubset(allowed) or not set(allowed).issubset(names):
                raise ValueError("missing or invalid task allowed-contact body set")
            threshold = _scalar(d.get("undesired_contact_force_threshold_n"), "task undesired-contact threshold")
            support_threshold = _scalar(d.get("foot_contact_force_threshold_n"), "foot contact threshold")
            if threshold < 0 or support_threshold < 0:
                raise ValueError("contact thresholds must be nonnegative")
            force = _array(d.get("foot_net_contact_forces_w_n"), "foot contact forces", (len(feet), 3))
            net = _array(d.get("net_contact_forces_w_n"), "body contact forces", (len(names), 3))
            contact = np.linalg.norm(force, axis=-1) > support_threshold
            foot_cost, speed_raw, selected_raw = [], {}, {}
            sphere_speed, sphere_clearance = d.get("foot_support_sphere_tangent_speed_m_s"), d.get("foot_support_sphere_clearance_m")
            if sphere_speed is not None and sphere_clearance is not None:
                velocity_semantics = d.get("slide_velocity_semantics", "support_sphere_center_tangent_velocity_proxy")
                for index, name in enumerate(feet):
                    speed = _array(sphere_speed.get(name), f"{name} support sphere tangent speed")
                    clearance = _array(sphere_clearance.get(name), f"{name} support sphere clearance", speed.shape)
                    if speed.ndim != 1 or not len(speed) or (speed < 0).any():
                        raise ValueError("support sphere speed must be a nonempty nonnegative vector")
                    selected = contact[index] & (clearance <= self.config["contact"]["support_clearance_m"])
                    if np.any(selected):
                        foot_cost.append(float(np.clip((speed[selected] / self.config["scales"]["slide_m_s"]) ** 2, 0., 1.).mean()))
                    speed_raw[name], selected_raw[name] = speed.tolist(), np.asarray(selected).tolist()
            else:
                body_names = d["body_names"]
                velocity = _array(d.get("body_link_lin_vel_w"), "actual body link velocities", (len(body_names), 3))
                clearance = _array(d.get("foot_min_support_clearance_m"), "support clearance", (len(feet),))
                velocity_semantics = "ankle_link_horizontal_velocity_slide_proxy"
                for index, name in enumerate(feet):
                    speed = float(np.linalg.norm(velocity[body_names.index(name), :2]))
                    selected = bool(contact[index] and clearance[index] <= self.config["contact"]["support_clearance_m"])
                    if selected:
                        foot_cost.append(float(np.clip((speed / self.config["scales"]["slide_m_s"]) ** 2, 0., 1.)))
                    speed_raw[name], selected_raw[name] = speed, selected
            bad_names = [name for name in names if name not in allowed]
            magnitudes = np.linalg.norm(net, axis=-1)
            bad = float(any(magnitudes[names.index(name)] > threshold for name in bad_names))
            slide = float(np.mean(foot_cost)) if foot_cost else 0.
            slides.append(slide)
            bads.append(bad)
            details.append({"physics_tick": sample["physics_tick"], "contact_force_semantics": "net_all_colliders_proxy",
                "foot_net_contact_forces_w_n": force.tolist(), "net_contact_forces_w_n": net.tolist(),
                "contact_body_names": list(names), "allowed_contact_body_names": list(allowed), "undesired_contact_body_names": bad_names,
                "undesired_contact_force_threshold_n": threshold, "foot_contact_force_threshold_n": support_threshold,
                "slide_velocity_semantics": velocity_semantics, "slide_proxy": speed_raw, "support_selected": selected_raw,
                "supported_foot_count": len(foot_cost), "slide_cost": slide, "bad_contact_cost": bad,
                "peak_foot_net_force_n": float(np.linalg.norm(force, axis=-1).max())})
        slide, bad = float(np.mean(slides)), float(np.mean(bads))
        cost = self.config["contact"]["slide_weight"] * slide + self.config["contact"]["bad_contact_weight"] * bad
        return cost, True, {"sampling": semantics, "samples": details, "impact_reward_weight": 0.}, {"slide": slide, "bad_contact": bad, "substep_slide_costs": slides, "substep_bad_contact_costs": bads}

    def _joint_limit(self, row):
        d = row["physical_diagnostics"]
        hard = _array(d.get("joint_pos_limits_rad"), "actual joint position limits", (21, 2))
        if (hard[:, 1] <= hard[:, 0]).any():
            raise ValueError("joint position limits require positive ranges")
        soft = d.get("soft_joint_pos_limits_rad")
        if soft is None:
            margin = (hard[:, 1] - hard[:, 0]) * self.config["joint_limit"]["hard_range_margin_fraction"]
            safe = np.stack((hard[:, 0] + margin, hard[:, 1] - margin), axis=-1)
            source = "hard_range_inner_margin"
        else:
            safe = _array(soft, "soft joint position limits", (21, 2))
            source = "backend_soft_joint_position_limits"
        original_safe = safe.copy()
        tolerance = self.config["joint_limit"]["numeric_tolerance_rad"]
        if (safe[:, 0] < hard[:, 0] - tolerance).any() or (safe[:, 1] > hard[:, 1] + tolerance).any() or (safe[:, 1] < safe[:, 0]).any():
            raise ValueError("soft joint limits must lie within hard limits up to numeric tolerance")
        safe = np.clip(safe, hard[:, :1], hard[:, 1:])
        actual = _array(row.get("actual_joint_pos_gmt"), "actual joint position", (21,))
        reference = _array(row["reference"].get("joint_pos"), "currently consumed reference joint position", (21,))
        def joint_cost(position):
            lower_width, upper_width = safe[:, 0] - hard[:, 0], hard[:, 1] - safe[:, 1]
            lower = np.divide(safe[:, 0] - position, lower_width, out=np.zeros(21), where=lower_width > 0)
            upper = np.divide(position - safe[:, 1], upper_width, out=np.zeros(21), where=upper_width > 0)
            lower[(lower_width == 0) & (position <= hard[:, 0])] = 1.
            upper[(upper_width == 0) & (position >= hard[:, 1])] = 1.
            return np.clip(np.maximum(lower, upper), 0., 1.)
        a_cost, r_cost = joint_cost(actual), joint_cost(reference)
        value = self.config["joint_limit"]["actual_weight"] * float(a_cost.max()) + self.config["joint_limit"]["reference_weight"] * float(r_cost.max())
        return value, True, {"actual_joint_position_rad": actual.tolist(), "consumed_reference_joint_position_rad": reference.tolist(),
            "reference_tick": row.get("reference_tick", row["tick"]), "hard_limits_rad": hard.tolist(),
            "original_safe_limits_rad": original_safe.tolist(), "safe_limits_rad": safe.tolist(), "safe_limits_source": source,
            "numeric_tolerance_rad": tolerance, "numeric_boundary_clamped": bool(np.any(safe != original_safe)),
            "numeric_boundary_note": "only floating-point soft-limit excess within configured tolerance is clipped to hard limits"}, {"actual_per_joint_cost": a_cost.tolist(), "reference_per_joint_cost": r_cost.tolist()}

    def _consistency(self, row):
        data = row.get("reference_consistency")
        if not isinstance(data, dict) or data.get("valid") is not True:
            raise ValueError("reference derivative support or consistency evidence unavailable")
        raw, normalized = {}, {}
        for key, tolerance in self.config["consistency"].items():
            raw[key] = _scalar(data.get(key), key)
            normalized[key] = abs(raw[key]) / tolerance
            if abs(raw[key]) > tolerance:
                raise ValueError(f"reference consistency construction error: {key}={raw[key]} exceeds {tolerance}")
        return {"valid": True, "raw": raw, "normalized": normalized, "tolerances": copy.deepcopy(self.config["consistency"]), "reward_weight": 0.}

    def _power(self, row):
        samples, semantics = self._physical_samples(row)
        details = []
        for sample in samples:
            d = sample["physical_diagnostics"]
            torque = _array(d.get("pd_torque_estimate_nm"), "PD torque estimate", (21,))
            velocity = _array(sample.get("joint_vel_gmt"), "joint velocity at physical sample", (21,))
            power = _array(np.abs(torque * velocity), "PD estimated mechanical power", (21,))
            metadata = d.get("mechanical_power_pd_estimate")
            if not isinstance(metadata, dict):
                raise ValueError("PD mechanical power sampling-time evidence unavailable")
            for key in ("torque_sample_tick", "velocity_sample_tick", "time_s"):
                _scalar(metadata.get(key), f"mechanical power {key}")
            synchronized = metadata.get("sampling_synchronized")
            if not isinstance(synchronized, bool):
                raise ValueError("PD power sampling_synchronized must be explicit boolean")
            if synchronized != (metadata["torque_sample_tick"] == metadata["velocity_sample_tick"]):
                raise ValueError("PD power sampling synchronization metadata contradicts timestamps")
            if metadata["velocity_sample_tick"] != sample["physics_tick"]:
                raise ValueError("PD power velocity timestamp does not match sample")
            if not math.isclose(metadata["time_s"], sample["physics_tick"] / 600, abs_tol=1e-12):
                raise ValueError("PD power timestamp seconds do not match sample tick")
            details.append({"per_joint_w": power.tolist(), "mean_w": float(power.mean()), "sum_w": float(power.sum()),
                "max_w": float(power.max()), "torque_sample_tick": metadata["torque_sample_tick"],
                "velocity_sample_tick": metadata["velocity_sample_tick"], "time_s": metadata["time_s"],
                "sampling_synchronized": synchronized})
        return {"valid": True, "sampling": semantics, "semantics": "PD_estimated_mechanical_power_proxy_not_electrical_power_or_energy",
            "sampling_synchronized": all(item["sampling_synchronized"] for item in details),
            "sampling_note": "PD torque estimate and velocity may come from opposite boundaries of a physics substep; inspect each timestamp",
            "samples": details, "mean_w": float(np.mean([item["mean_w"] for item in details])),
            "mean_sum_w": float(np.mean([item["sum_w"] for item in details])), "max_w": max(item["max_w"] for item in details), "reward_weight": 0.}

    @profiled('collection.reward')
    def evaluate_step(self, trace):
        tick = trace.get("tick")
        if isinstance(tick, bool) or not isinstance(tick, (int, np.integer)) or tick % 12:
            raise ValueError("reward trace requires an integer 50Hz control tick")
        episode = trace.get("episode_id")
        if self._last_tick is not None and (episode != self._episode or tick != self._last_tick + 12):
            raise ValueError("reward stream must be continuous within an episode; reset explicitly")
        errors, components, diagnostics = [], {}, {}
        try:
            positions = _array(trace.get("actual_joint_pos_gmt"), "actual_joint_pos_gmt", (21,))
            velocity = _array(trace.get("actual_joint_vel_gmt"), "actual_joint_vel_gmt", (21,))
            self.window.append((int(tick), positions.copy(), velocity.copy()))
        except ValueError as error:
            errors.append(str(error))
        self._last_tick, self._episode = int(tick), episode
        try:
            gate, intensity, activity = self._activity(tick)
        except (ValueError, KeyError, IndexError, TypeError) as error:
            gate, intensity, activity = 0., 0., {"valid": False, "reason": str(error), "status": "not_available"}
            errors.append(f"activity: {error}")
        mapping = {"track": "track_weight", "music": "music_weight", "stable": "stable_weight", "alive": "alive_weight",
                   "cmd": "cmd_penalty_weight", "torque": "torque_penalty_weight", "contact": "contact_penalty_weight", "joint_limit": "joint_limit_penalty_weight"}
        for name, weight_key in mapping.items():
            try:
                if name == "music" and not activity["valid"]:
                    raise ValueError("music intensity not_available without paired target activity")
                score, valid, raw, normalized = self._music(intensity, activity) if name == "music" else getattr(self, "_" + name)(trace)
                if not math.isfinite(score) or not 0 <= score <= 1 + 1e-12:
                    raise ValueError(f"{name} score must be finite and within [0,1]")
                score = min(score, 1.)
            except (ValueError, KeyError, IndexError, TypeError) as error:
                score, valid, raw, normalized = 0., False, {"reason": str(error)}, {}
                errors.append(f"{name}: {error}")
            multiplier = gate if name in {"track", "music"} else 1.
            weight = self.config[weight_key] * (-1 if "penalty" in weight_key else 1)
            rate = score * weight * multiplier
            components[name] = {"enabled": True, "valid": bool(valid), "raw": raw, "normalized": normalized,
                "score": score, "weight": weight, "gate": multiplier, "activity_gate": multiplier,
                "weighted_rate": rate, "integrated_reward": self.config["dt"] * rate}
        for name in ("consistency", "power"):
            try:
                diagnostics[name] = getattr(self, "_" + name)(trace)
            except (ValueError, KeyError, IndexError, TypeError) as error:
                diagnostics[name] = {"valid": False, "reason": str(error), "reward_weight": 0.}
                if name == "consistency":
                    diagnostics[name]["raw"] = copy.deepcopy(trace.get("reference_consistency"))
                    diagnostics[name]["tolerances"] = copy.deepcopy(self.config["consistency"])
                errors.append(f"{name}: {error}")
        if not trace.get("transition_valid", True) or not trace.get("state_valid", True):
            errors.append("backend marked execution/state invalid")
        if trace.get("completed_physics_steps", 4) != 4:
            errors.append("incomplete control interval")
        total_rate = sum(item["weighted_rate"] for item in components.values())
        return {"version": "stage9.execution_reward.v2", "tick": int(tick), "episode_id": episode,
                "reward": self.config["dt"] * total_rate, "reward_rate": total_rate,
                "reward_is_integrated": True, "dt_s": self.config["dt"], "transition_valid": not errors,
                "errors": errors, "components": components, "activity": activity, "diagnostics": diagnostics,
                "scales": copy.deepcopy(self.config["scales"])}


def resolve_reward_config(config=None):
    """仅解析并验证奖励配置，不加载训练数据、不需要目标活动度或执行轨迹。"""
    return copy.deepcopy(ExecutionReward(config=config).config)
