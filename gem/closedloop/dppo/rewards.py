"""第九步真实执行奖励及与 RPC 分段无关的因果音乐统计。

奖励仅从实际控制步 trace、权威参考和实际物理诊断计算，不使用生成轨迹替代机器人
动作。控制步输出是 20ms 的积分奖励：2×音乐＋2×跟踪＋稳定性−0.2×执行器代理
−0.5×接触代价−0.5×位置速度一致性，统一乘 dt 一次。事件代价由调用方单独加入
一次；零步拒绝也可记录事件。各分项保留原始物理量、尺度、归一化分数、权重及
有效性，缺测不伪装成零误差；必需诊断缺失或非有限则标记整步不可训练。

音乐使用过去一秒真实关节速度的局部极小值，与既有 BUMI 节拍算法相同；新极小值
只能在后一个控制步确认，奖励归入确认时刻，不回填旧 RPC。活动幅度门控防止静止
获得完整节拍分，活动强度和 onset 匹配均有界。onset 是 EDGE35 第零列的强度代理，
不是音量或音乐语义。执行器字段始终标作隐式 PD 估计；接触使用全碰撞对象净力。
若存在四个 200Hz 子步，则子步平均代价及真实子步峰值均单独保存；无子步时明确
仅为 50Hz 末帧代理，不宣称捕获物理子步峰值。
"""
from __future__ import annotations

from collections import deque
import copy
import math

import numpy as np
import torch

from gem.robots.bumi.metrics import _derive_motion_beats, _beat_alignment


DEFAULT_CONFIG = {
    "dt_s": .02, "weights": {"music": 2., "track": 2., "stable": 1., "actuator": -.2, "contact": -.5, "consistency": -.5},
    "scales": {"joint_pos_rad": .25, "joint_vel_rad_s": 2.5, "root_xy_m": .25,
               "root_height_m": .20, "non_yaw_rad": .60, "yaw_rad": 1.50, "end_effector_m": .15,
               "angular_velocity_rad_s": 6., "joint_acceleration_rad_s2": 50., "activity_rad": .03,
               "activity_rad_s": 2., "onset_strength": 1., "slip_m_s": .20,
               "contact_force_n": 200., "nonfoot_force_n": 100., "target_velocity_rad_s": 5.,
               "consistency_joint_rad_s": 1e-4, "consistency_root_m_s": 1e-4, "consistency_root_rad_s": 1e-4},
    "support_force_n": 5., "support_clearance_m": .025, "music_window_steps": 50,
    "require_substeps": True, "enabled": {"music": True, "track": True, "stable": True,
                                            "actuator": True, "contact": True, "consistency": True},
    "failure_penalty": -5., "rejection_penalty": -1.,
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


def _cost(values, scale=1.):
    return float(np.mean(np.minimum(np.abs(np.asarray(values)) / scale, 1.) ** 2))


def _score(value, scale):
    # 奖励本身有界；只对奖励归一化量限制指数范围，绝不修改策略概率。
    return float(np.exp(-np.minimum(np.abs(np.asarray(value)) / scale, 30.) ** 2).mean())


def _tilt(quaternion):
    q = _array(quaternion, "root quaternion", (4,))
    if abs(np.linalg.norm(q) - 1.) > 1e-3:
        raise ValueError("root quaternion must be normalized")
    return float(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1., 1.)))


class ExecutionReward:
    def __init__(self, config=None, music_features=None, music_start_tick=600):
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        for key, value in (config or {}).items():
            if key not in self.config:
                raise ValueError(f"unknown reward setting {key}")
            if isinstance(self.config[key], dict):
                if set(value) - set(self.config[key]):
                    raise ValueError(f"unknown reward {key} entries")
                self.config[key].update(value)
            else:
                self.config[key] = value
        if self.config["dt_s"] != .02 or self.config["music_window_steps"] != 50:
            raise ValueError("Stage9 rewards require 50Hz and a 50-step causal music window")
        for name, value in self.config["scales"].items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"reward scale must be positive: {name}")
        if not all(math.isfinite(value) for value in self.config["weights"].values()):
            raise ValueError("reward weights must be finite")
        self.music = None if music_features is None else _array(music_features, "music_features").copy()
        if self.music is not None and (self.music.ndim != 2 or self.music.shape[1] != 35):
            raise ValueError("music_features must be EDGE35 [T,35]")
        self.music_start_tick = int(music_start_tick)
        self.window = deque(maxlen=50)
        self._last_tick = None
        self._episode = None
        self._previous_target = None
        self._previous_velocity = None

    def reset(self, *, music_features=None, music_start_tick=None):
        if music_features is not None:
            music = _array(music_features, "music_features").copy()
            if music.ndim != 2 or music.shape[1] != 35:
                raise ValueError("music_features must be EDGE35 [T,35]")
            self.music = music
        if music_start_tick is not None:
            self.music_start_tick = int(music_start_tick)
        self.window.clear()
        self._last_tick = self._episode = self._previous_target = self._previous_velocity = None

    def event_reward(self, reason):
        if reason in {"reference_rejected", "plan_rejected", "rejected"}:
            return float(self.config["rejection_penalty"])
        if reason in {"physical_failure", "task_failure", "reference_exhausted"}:
            return float(self.config["failure_penalty"])
        if reason in {None, "music_end", "task_complete", "collection_limit", "infrastructure_failure"}:
            return 0.
        raise ValueError(f"event must use an explicit Stage9 reason category, got {reason}")

    def _music(self):
        if self.music is None:
            return 0., False, {"reason": "music_unavailable"}
        if len(self.window) < 50:
            return 0., False, {"reason": "insufficient_causal_history", "steps": len(self.window)}
        ticks = np.asarray([row[0] for row in self.window])
        positions = np.stack([row[1] for row in self.window])
        speeds = np.stack([np.abs(row[2]) for row in self.window])
        idx = (ticks - self.music_start_tick) // 20
        valid = (idx >= 0) & (idx < len(self.music))
        if not valid.all():
            return 0., False, {"reason": "music_outside_task"}
        # 按既有 Stage8 规则将 30Hz 节拍映射到 50Hz tick，避免一个节拍重复计两次。
        beat_source = np.flatnonzero(self.music[:, 34] > .5)
        beat_ticks = self.music_start_tick + np.rint(beat_source * 20 / 12).astype(np.int64) * 12
        music_beats = torch.from_numpy(np.isin(ticks, beat_ticks))[None]
        motion_beats = _derive_motion_beats(torch.from_numpy(speeds)[None], torch.ones((1, 50), dtype=torch.bool))
        music_count, motion_count = int(music_beats.sum()), int(motion_beats.sum())
        amplitude = float(np.mean(np.ptp(positions, axis=0)))
        gate = min(amplitude / self.config["scales"]["activity_rad"], 1.)
        activity = min(float(np.mean(speeds)) / self.config["scales"]["activity_rad_s"], 1.)
        onset = min(max(float(np.mean(self.music[idx, 0])), 0.) / self.config["scales"]["onset_strength"], 1.)
        intensity = 1. - abs(activity - onset)
        if music_count == 0:
            return 0., False, {"reason": "no_music_beats", "activity_gate": gate, "motion_beat_count": motion_count,
                               "onset_intensity_proxy": onset, "activity": activity}
        alignment = 0.
        if motion_count:
            _, alignment_tensor = _beat_alignment(music_beats, motion_beats, torch.ones((1, 50), dtype=torch.bool), 50)
            alignment = float(alignment_tensor)
        return gate * (.7 * alignment + .3 * intensity), True, {
            "beat_alignment": alignment, "music_beat_count": music_count, "motion_beat_count": motion_count,
            "activity_gate": gate, "mean_joint_range_rad": amplitude, "activity": activity,
            "onset_intensity_proxy": onset, "intensity_alignment": intensity}

    def _track(self, row):
        s = self.config["scales"]
        actual = _array(row.get("actual_joint_pos_gmt"), "actual_joint_pos_gmt", (21,))
        velocity = _array(row.get("actual_joint_vel_gmt"), "actual_joint_vel_gmt", (21,))
        ref, err = row["reference"], row["errors"]
        q_error = actual - _array(ref.get("joint_pos"), "reference.joint_pos", (21,))
        dq_error = velocity - _array(ref.get("joint_vel"), "reference.joint_vel", (21,))
        qpos = _array(row.get("actual_qpos"), "actual_qpos", (28,))
        roots = _array(ref.get("body_pos_w"), "reference.body_pos_w")
        if roots.ndim != 2 or roots.shape[1] != 3 or not len(roots):
            raise ValueError("reference body positions must be [N,3]")
        raw = {"joint_pos_rmse_rad": float(np.sqrt(np.mean(q_error ** 2))),
               "joint_vel_rmse_rad_s": float(np.sqrt(np.mean(dq_error ** 2))),
               "root_xy_error_m": float(np.linalg.norm(qpos[:2] - roots[0, :2]))}
        scores = [_score(q_error, s["joint_pos_rad"]), _score(dq_error, s["joint_vel_rad_s"]),
                  _score(raw["root_xy_error_m"], s["root_xy_m"])]
        for key, scale in (("yaw_error_rad", "yaw_rad"), ("end_effector_relative_height_error_m", "end_effector_m")):
            raw[key] = _scalar(err.get(key), key)
            scores.append(_score(raw[key], s[scale]))
        return float(np.mean(scores)), True, raw

    def _stable(self, row):
        qpos = _array(row.get("actual_qpos"), "actual_qpos", (28,))
        reference = _array(row["reference"].get("body_quat_w"), "reference.body_quat_w")
        omega = _array(row.get("actual_root_ang_vel_b"), "actual_root_ang_vel_b", (3,))
        tilt, ref_tilt = _tilt(qpos[3:7]), _tilt(reference[0])
        omega_norm = float(np.linalg.norm(omega))
        height_error = _scalar(row["errors"].get("root_height_error_m"), "root_height_error_m")
        non_yaw = _scalar(row["errors"].get("non_yaw_orientation_error_rad"), "non_yaw_orientation_error_rad")
        velocity = _array(row.get("actual_joint_vel_gmt"), "actual_joint_vel_gmt", (21,))
        scores = [_score(height_error, self.config["scales"]["root_height_m"]),
                  _score(non_yaw, self.config["scales"]["non_yaw_rad"]),
                  _score(omega_norm, self.config["scales"]["angular_velocity_rad_s"])]
        acceleration = None
        if self._previous_velocity is not None:
            values = (velocity - self._previous_velocity) / self.config["dt_s"]
            acceleration = float(np.sqrt(np.mean(values ** 2)))
            scores.append(_score(values, self.config["scales"]["joint_acceleration_rad_s2"]))
        self._previous_velocity = velocity.copy()
        return float(np.mean(scores)), True, {"actual_tilt_rad": tilt, "reference_tilt_rad": ref_tilt,
            "root_height_error_m": height_error, "non_yaw_orientation_error_rad": non_yaw,
            "actual_angular_speed_rad_s": omega_norm,
            "joint_acceleration_rms_rad_s2": acceleration, "acceleration_valid": acceleration is not None,
            "supported_score_count": len(scores)}

    def _physical_samples(self, row):
        samples = row.get("physics_substeps")
        if samples is None:
            if self.config["require_substeps"]:
                raise ValueError("four physical substeps are required for enabled physical rewards")
            return [{"physical_diagnostics": row["physical_diagnostics"], "joint_vel_gmt": row["actual_joint_vel_gmt"],
                     "joint_position_target": row.get("joint_position_target"), "dt_s": .02}], "control_end_proxy"
        if len(samples) != 4 or any(abs(_scalar(item.get("dt_s"), "substep dt") - .005) > 1e-12 for item in samples):
            raise ValueError("normal control step requires four actual 5ms substep samples")
        return samples, "four_physics_substeps"

    def _actuator(self, row):
        samples, semantics = self._physical_samples(row)
        costs, peaks, powers, target_costs = [], [], [], []
        previous = self._previous_target
        for sample in samples:
            d = sample["physical_diagnostics"]
            torque = _array(d.get("applied_joint_torque_nm"), "implicit PD torque estimate", (21,))
            effort = _array(d.get("joint_effort_limits_nm"), "effort limits", (21,))
            velocity_limit = _array(d.get("joint_velocity_limits_rad_s"), "velocity limits", (21,))
            velocity = _array(sample.get("joint_vel_gmt"), "actual joint velocity", (21,))
            target = _array(sample.get("joint_position_target"), "joint position target", (21,))
            if (effort <= 0).any() or (velocity_limit <= 0).any():
                raise ValueError("actuator normalization limits must be positive")
            target_cost = 0. if previous is None else _cost((target - previous) / sample["dt_s"], self.config["scales"]["target_velocity_rad_s"])
            costs.append(.5 * _cost(torque / effort) + .25 * _cost((torque / effort) * (velocity / velocity_limit)) + .25 * target_cost)
            peaks.append(float(np.abs(torque).max()))
            powers.append(float(np.abs(torque * velocity).sum()))
            target_costs.append(target_cost)
            previous = target
        self._previous_target = previous.copy()
        return float(np.mean(costs)), True, {"sampling": semantics, "torque_semantics": "implicit_pd_estimate_proxy",
            "sampled_peak_estimated_torque_nm": max(peaks), "mean_estimated_abs_power_w": float(np.mean(powers)),
            "mean_target_change_cost": float(np.mean(target_costs))}

    def _contact(self, row):
        samples, semantics = self._physical_samples(row)
        costs, slides, peaks = [], [], []
        for sample in samples:
            d = sample["physical_diagnostics"]
            names, feet = d["body_names"], d["foot_body_names"]
            velocities = _array(d.get("body_link_lin_vel_w"), "actual body link velocity", (len(names), 3))
            foot_velocity = velocities[[names.index(name) for name in feet]]
            force = _array(d.get("foot_net_contact_forces_w_n"), "foot net contact force", (2, 3))
            clearance = _array(d.get("foot_min_support_clearance_m"), "support clearance", (2,))
            support = (force[:, 2] > self.config["support_force_n"]) & (clearance < self.config["support_clearance_m"])
            speed = np.linalg.norm(foot_velocity[:, :2], axis=-1)
            slip = _cost(speed[support], self.config["scales"]["slip_m_s"]) if support.any() else 0.
            force_norm = np.linalg.norm(force, axis=-1)
            impact = _cost(np.maximum(force_norm - self.config["scales"]["contact_force_n"], 0.), self.config["scales"]["contact_force_n"])
            sensor_names = d["contact_body_names"]
            net = _array(d.get("net_contact_forces_w_n"), "net contact force", (len(sensor_names), 3))
            nonfoot_ids = [i for i, name in enumerate(sensor_names) if name not in feet]
            nonfoot = _cost(np.linalg.norm(net[nonfoot_ids], axis=-1), self.config["scales"]["nonfoot_force_n"]) if nonfoot_ids else 0.
            costs.append(.5 * slip + .25 * impact + .25 * nonfoot)
            slides.append(float(speed[support].max()) if support.any() else 0.)
            peaks.append(float(force_norm.max()))
        return float(np.mean(costs)), True, {"sampling": semantics, "contact_force_semantics": "net_all_colliders_proxy",
            "supported_slide_max_m_s": max(slides), "sampled_peak_foot_net_force_n": max(peaks)}

    def _consistency(self, row):
        data = row.get("reference_consistency")
        if not isinstance(data, dict) or data.get("valid") is not True:
            raise ValueError("reference derivative support or consistency evidence unavailable")
        pairs = (("joint_vel_rms_rad_s", "consistency_joint_rad_s"), ("root_lin_vel_rms_m_s", "consistency_root_m_s"),
                 ("root_ang_vel_rms_rad_s", "consistency_root_rad_s"))
        raw = {key: _scalar(data.get(key), key) for key, _scale in pairs}
        return float(np.mean([_cost(raw[key], self.config["scales"][scale]) for key, scale in pairs])), True, raw

    def evaluate_step(self, trace):
        tick = trace.get("tick")
        if isinstance(tick, bool) or not isinstance(tick, (int, np.integer)) or tick % 12:
            raise ValueError("reward trace requires an integer 50Hz control tick")
        episode = trace.get("episode_id")
        if self._last_tick is not None and (episode != self._episode or tick != self._last_tick + 12):
            raise ValueError("reward stream must be continuous within an episode; reset explicitly")
        components, errors = {}, []
        try:
            positions = _array(trace.get("actual_joint_pos_gmt"), "actual_joint_pos_gmt", (21,))
            velocity = _array(trace.get("actual_joint_vel_gmt"), "actual_joint_vel_gmt", (21,))
            self.window.append((int(tick), positions.copy(), velocity.copy()))
        except ValueError as error:
            errors.append(str(error))
        self._last_tick, self._episode = int(tick), episode
        for name in self.config["weights"]:
            enabled = bool(self.config["enabled"][name])
            if not enabled:
                score, valid, raw = 0., False, {"reason": "explicitly_disabled"}
            else:
                try:
                    score, valid, raw = self._music() if name == "music" else getattr(self, "_" + name)(trace)
                    if not math.isfinite(score):
                        raise ValueError(f"nonfinite {name} reward")
                except (ValueError, KeyError, IndexError) as error:
                    score, valid, raw = 0., False, {"reason": str(error)}
                    errors.append(f"{name}: {error}")
            components[name] = {"enabled": enabled, "valid": valid, "raw": raw, "score": score,
                                "weight": self.config["weights"][name],
                                "weighted_rate": score * self.config["weights"][name]}
        total = self.config["dt_s"] * sum(item["weighted_rate"] for item in components.values())
        return {"version": "stage9.execution_reward.v1", "tick": int(tick), "episode_id": episode,
                "reward": total, "reward_is_integrated": True, "dt_s": self.config["dt_s"],
                "transition_valid": not errors and trace.get("transition_valid", True),
                "errors": errors, "components": components, "scales": copy.deepcopy(self.config["scales"])}
