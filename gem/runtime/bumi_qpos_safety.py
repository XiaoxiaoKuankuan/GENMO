# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""现行 BUMI qpos30 部署链的发布前运动学安全门。

本模块只接收已经由当前 GENMO qpos30/contact v5 链解码得到的 30 Hz、MuJoCo 原生
``qpos[T,28]``。它在轨迹进入 30→50 Hz 重采样和 GMT reference 构造之前，检查有限值、
单位四元数、根高度、fe934 运动学资产给出的关节上下限，以及根、关节和根姿态的跨帧
速度。安全门会保留上一分块末帧，因此分块边界同样接受速度检查；任何失败都会拒绝完整
输入块，不进行静默裁剪、滤波或自动修复。

该模块只保留当前常驻 BUMI→GMT 安全桥需要的公共能力，不定义额外的传输协议，也不承担
动力学、控制稳定性或实机安全认证；阈值仍必须针对实际 BUMI 资产、GMT 策略和运行环境
单独核验。
"""

from __future__ import annotations

import math

import numpy as np

from gem.robots.bumi.kinematics import BumiKinematics
from gem.runtime.bumi_online_stream import BUMI_QPOS_DIM, BUMI_SOURCE_FPS


class BumiQposSafetyGate:
    """在当前实时发布边界执行跨分块 qpos 运动学检查。"""

    def __init__(
        self,
        kinematics: BumiKinematics,
        *,
        joint_limit_tolerance_rad: float = 0.05,
        max_joint_velocity_radps: float = 18.0,
        max_root_linear_velocity_mps: float = 4.0,
        max_root_angular_velocity_radps: float = 8.0,
        min_root_height_m: float = 0.25,
        max_root_height_m: float = 1.20,
    ) -> None:
        if not isinstance(kinematics, BumiKinematics):
            raise TypeError("BumiQposSafetyGate requires BumiKinematics")
        self.kinematics = kinematics
        values = {
            "joint_limit_tolerance_rad": joint_limit_tolerance_rad,
            "max_joint_velocity_radps": max_joint_velocity_radps,
            "max_root_linear_velocity_mps": max_root_linear_velocity_mps,
            "max_root_angular_velocity_radps": max_root_angular_velocity_radps,
        }
        if any(not math.isfinite(float(value)) or value < 0.0 for value in values.values()):
            raise ValueError("BUMI safety tolerances/speed limits must be finite and >= 0")
        if not 0.0 <= min_root_height_m < max_root_height_m:
            raise ValueError("BUMI root height bounds are invalid")
        self.joint_limit_tolerance_rad = float(joint_limit_tolerance_rad)
        self.max_joint_velocity_radps = float(max_joint_velocity_radps)
        self.max_root_linear_velocity_mps = float(max_root_linear_velocity_mps)
        self.max_root_angular_velocity_radps = float(max_root_angular_velocity_radps)
        self.min_root_height_m = float(min_root_height_m)
        self.max_root_height_m = float(max_root_height_m)
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def validate(self, qpos: np.ndarray) -> np.ndarray:
        """验证一个连续 30 Hz qpos 块，返回四元数符号连续的独立副本。"""

        values = np.asarray(qpos, dtype=np.float32).copy()
        if values.ndim != 2 or values.shape[1] != BUMI_QPOS_DIM or len(values) <= 0:
            raise ValueError(f"qpos must have shape [T,{BUMI_QPOS_DIM}] with T > 0")
        if not np.isfinite(values).all():
            raise ValueError("BUMI safety gate rejected NaN/Inf qpos")

        quat_norm = np.linalg.norm(values[:, 3:7], axis=1)
        if np.any(np.abs(quat_norm - 1.0) > 2.0e-3):
            raise ValueError("BUMI safety gate rejected non-unit root quaternion")
        values[:, 3:7] /= quat_norm[:, None]
        if self._previous is not None and float(
            np.dot(self._previous[3:7], values[0, 3:7])
        ) < 0.0:
            values[:, 3:7] *= -1.0
        for index in range(1, len(values)):
            if float(np.dot(values[index - 1, 3:7], values[index, 3:7])) < 0.0:
                values[index, 3:7] *= -1.0

        if np.any(values[:, 2] < self.min_root_height_m) or np.any(
            values[:, 2] > self.max_root_height_m
        ):
            frame = int(
                np.argmax(
                    np.maximum(
                        self.min_root_height_m - values[:, 2],
                        values[:, 2] - self.max_root_height_m,
                    )
                )
            )
            raise ValueError(
                "BUMI safety gate rejected root height outside configured bounds: "
                f"frame={frame}, value={float(values[frame, 2]):.6f}, "
                f"bounds=[{self.min_root_height_m:.6f},{self.max_root_height_m:.6f}]"
            )

        lower = self.kinematics.joint_lower_limits.detach().cpu().numpy()
        upper = self.kinematics.joint_upper_limits.detach().cpu().numpy()
        joint_values = values[:, 7:]
        lower_excess = lower[None] - self.joint_limit_tolerance_rad - joint_values
        upper_excess = joint_values - (upper[None] + self.joint_limit_tolerance_rad)
        excess = np.maximum(lower_excess, upper_excess)
        if float(excess.max(initial=0.0)) > 0.0:
            frame, joint = np.unravel_index(int(np.argmax(excess)), excess.shape)
            side = "lower" if lower_excess[frame, joint] >= upper_excess[frame, joint] else "upper"
            bound = lower[joint] if side == "lower" else upper[joint]
            raise ValueError(
                "BUMI safety gate rejected XML joint-limit violation: "
                f"frame={frame}, joint={self.kinematics.joint_order[joint]}, "
                f"value={float(joint_values[frame, joint]):.6f}, {side}_bound={float(bound):.6f}, "
                f"tolerance={self.joint_limit_tolerance_rad:.6f}, "
                f"excess_after_tolerance={float(excess[frame, joint]):.6f}"
            )

        sequence = (
            values
            if self._previous is None
            else np.concatenate((self._previous[None], values), axis=0)
        )
        if len(sequence) > 1:
            root_speed = (
                np.linalg.norm(np.diff(sequence[:, :3], axis=0), axis=1) * BUMI_SOURCE_FPS
            )
            joint_speed = np.abs(np.diff(sequence[:, 7:], axis=0)) * BUMI_SOURCE_FPS
            quat_dot = np.abs(np.sum(sequence[1:, 3:7] * sequence[:-1, 3:7], axis=1))
            angular_speed = (
                2.0 * np.arccos(np.clip(quat_dot, 0.0, 1.0)) * BUMI_SOURCE_FPS
            )
            if float(root_speed.max(initial=0.0)) > self.max_root_linear_velocity_mps:
                frame = int(np.argmax(root_speed))
                raise ValueError(
                    "BUMI safety gate rejected excessive root linear velocity: "
                    f"transition={frame}->{frame + 1}, value={float(root_speed[frame]):.6f}, "
                    f"limit={self.max_root_linear_velocity_mps:.6f}"
                )
            if float(joint_speed.max(initial=0.0)) > self.max_joint_velocity_radps:
                frame, joint = np.unravel_index(int(np.argmax(joint_speed)), joint_speed.shape)
                raise ValueError(
                    "BUMI safety gate rejected excessive joint velocity: "
                    f"transition={frame}->{frame + 1}, joint={self.kinematics.joint_order[joint]}, "
                    f"value={float(joint_speed[frame, joint]):.6f}, "
                    f"limit={self.max_joint_velocity_radps:.6f}"
                )
            if float(angular_speed.max(initial=0.0)) > self.max_root_angular_velocity_radps:
                frame = int(np.argmax(angular_speed))
                raise ValueError(
                    "BUMI safety gate rejected excessive root angular velocity: "
                    f"transition={frame}->{frame + 1}, value={float(angular_speed[frame]):.6f}, "
                    f"limit={self.max_root_angular_velocity_radps:.6f}"
                )

        self._previous = values[-1].copy()
        return values


__all__ = ["BumiQposSafetyGate"]
