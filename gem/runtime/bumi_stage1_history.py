"""部署侧Stage1因果本体历史构造器，逐条保持当前训练/演示实现的物理语义。

输入为截止决策帧的30Hz世界qpos28，输出50个50Hz proprio48历史槽、bool有效掩码和
float64绝对时间。48维顺序为身体重力3、身体角速度3、GMT关节顺序的位置偏差21、
速度偏差21。默认姿态使用训练名义值，不读取部署GMT policy中经舍入的默认值。
使用150Hz整数公共时基和最新已到达样本保持；速度仅用后向差分，不引入未来帧。
第一个源样本缺少前驱，因此历史无效。调用方只能传入0..decision_frame的因果前缀。
本文件迁入当前Stage1的CausalDemoProprio48Builder必要部分，不依赖训练数据集、
训练Actor或真实机器人反馈；在线和buffered模式都使用自己的生成轨迹构造历史。
"""

from __future__ import annotations

import torch

from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec, make_quaternion_continuous
from gem.robots.bumi.kinematics import BumiKinematics
from gem.runtime.bumi_music_contract import HISTORY_STEPS, PROPRIO_DIM, SOURCE_FPS
from gem.utils.rotation_conversions import (
    matrix_to_axis_angle, quaternion_apply, quaternion_invert, quaternion_to_matrix,
)

MOTION_FPS = SOURCE_FPS
PROPRIO_HISTORY_STEPS = HISTORY_STEPS
_COMMON_TIMEBASE_HZ = 150
_MOTION_TICKS = 5
_PROPRIO_TICKS = 3

GMT_EXPECTED_JOINT_ORDER = (
    "l_leg_pitch_joint",
    "r_leg_pitch_joint",
    "waist_yaw_joint",
    "l_leg_roll_joint",
    "r_leg_roll_joint",
    "l_arm_pitch_joint",
    "r_arm_pitch_joint",
    "l_leg_yaw_joint",
    "r_leg_yaw_joint",
    "l_arm_roll_joint",
    "r_arm_roll_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_arm_yaw_joint",
    "r_arm_yaw_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_elbow_pitch_joint",
    "r_elbow_pitch_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
)

# Bumi_CFG 名义 default pose，已按 GMT_EXPECTED_JOINT_ORDER 排列。GMT 训练任务启动时会对
# active default_joint_pos 的每个环境、每个关节独立加 U[-0.02, 0.02] rad；joint_pos_rel
# 始终相对当时 active default，而不是无条件相对下面这张名义表。
GMT_NOMINAL_DEFAULT_JOINT_POS_RAD = (
    -0.1495,
    -0.1495,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.3,
    -0.3,
    0.3215,
    0.3215,
    0.0,
    0.0,
    -0.1720,
    -0.1720,
    0.0,
    0.0,
    0.0,
    0.0,
)
GMT_DEFAULT_JOINT_VEL_RAD_S = (0.0,) * 21

def _validate_qpos(qpos: torch.Tensor) -> torch.Tensor:
    if not isinstance(qpos, torch.Tensor) or qpos.ndim != 2 or qpos.shape[1] != 28:
        raise ValueError(f"qpos must have shape [T,28], got {getattr(qpos, 'shape', None)}")
    if qpos.shape[0] <= 0 or not bool(torch.isfinite(qpos).all()):
        raise ValueError("qpos must contain at least one finite frame")
    return BumiMotionFeatureCodec.normalize_qpos_sequence(qpos.detach().cpu().float())


class CausalDemoProprio48Builder:
    """把 30 Hz 示范 qpos 构造成严格因果的 50 Hz proprio48 proxy。"""

    def __init__(self, kinematics: BumiKinematics) -> None:
        if not isinstance(kinematics, BumiKinematics):
            raise TypeError("CausalDemoProprio48Builder requires BumiKinematics")
        self.kinematics = kinematics
        source_order = tuple(kinematics.joint_order)
        if len(source_order) != 21 or len(set(source_order)) != 21:
            raise ValueError("BUMI kinematics must expose 21 unique joint names")
        missing = [name for name in GMT_EXPECTED_JOINT_ORDER if name not in source_order]
        extra = [name for name in source_order if name not in GMT_EXPECTED_JOINT_ORDER]
        if missing or extra:
            raise ValueError(
                "BUMI/GMT joint-name sets do not match: "
                f"missing_from_qpos={missing}, extra_in_qpos={extra}"
            )
        self.source_joint_order = source_order
        self.gmt_joint_order = tuple(GMT_EXPECTED_JOINT_ORDER)
        self.gmt_from_source = torch.tensor(
            [source_order.index(name) for name in self.gmt_joint_order], dtype=torch.long
        )
        self.default_joint_pos = torch.tensor(
            GMT_NOMINAL_DEFAULT_JOINT_POS_RAD, dtype=torch.float32
        )
        self.default_joint_vel = torch.tensor(GMT_DEFAULT_JOINT_VEL_RAD_S, dtype=torch.float32)

    def source_observations(self, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回每个 30 Hz 源时刻的 proprio48 与整帧有效 mask。

        姿态、关节位置只使用当前帧；角速度和关节速度只使用 ``t-1 -> t`` 后向差分。
        因 48 维契约只有整帧 history mask，源第 0 帧的速度不可得，所以整帧标无效；其中
        的有限零速度只是 padding 占位，不能被消费者解释为真实静止速度。
        """

        timeline = _validate_qpos(qpos)
        quaternion = make_quaternion_continuous(timeline[:, 3:7])
        source_joint = timeline[:, 7:]
        permutation = self.gmt_from_source.to(source_joint.device)
        joint = source_joint.index_select(-1, permutation)

        gravity_world = timeline.new_tensor((0.0, 0.0, -1.0)).expand(len(timeline), 3)
        projected_gravity = quaternion_apply(quaternion_invert(quaternion), gravity_world)
        joint_pos_rel = joint - self.default_joint_pos.to(joint)

        base_ang_vel = timeline.new_zeros((len(timeline), 3))
        joint_vel_rel = timeline.new_zeros((len(timeline), 21))
        valid = torch.zeros(len(timeline), dtype=torch.bool, device=timeline.device)
        if len(timeline) > 1:
            rotation = quaternion_to_matrix(quaternion)
            relative_world = rotation[1:] @ rotation[:-1].transpose(-1, -2)
            angular_velocity_world = matrix_to_axis_angle(relative_world) * float(MOTION_FPS)
            base_ang_vel[1:] = quaternion_apply(
                quaternion_invert(quaternion[1:]), angular_velocity_world
            )
            joint_velocity = (joint[1:] - joint[:-1]) * float(MOTION_FPS)
            joint_vel_rel[1:] = joint_velocity - self.default_joint_vel.to(joint_velocity)
            valid[1:] = True

        observations = torch.cat(
            (projected_gravity, base_ang_vel, joint_pos_rel, joint_vel_rel), dim=-1
        ).contiguous()
        if observations.shape != (len(timeline), PROPRIO_DIM):
            raise RuntimeError(f"internal proprio48 shape error: {tuple(observations.shape)}")
        if not bool(torch.isfinite(observations).all()):
            raise ValueError("constructed proprio48 contains NaN or Inf")
        return observations, valid

    def build_history(
        self,
        causal_qpos_prefix: torch.Tensor,
        *,
        decision_frame: int,
        history_steps: int = PROPRIO_HISTORY_STEPS,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """在 150 Hz 整数公共时基上构造截至 decision frame 的 50 Hz 历史。

        ``causal_qpos_prefix`` 必须恰好终止于 ``decision_frame``，从接口上阻止调用方把未来
        qpos 交给历史构造器。每个 50 Hz tick 只取时间戳不晚于它的最新 30 Hz 源样本，
        不做需要右侧括点的线性插值、SLERP、中心差分或滤波。
        """

        history_steps = int(history_steps)
        decision_frame = int(decision_frame)
        if history_steps <= 0:
            raise ValueError("history_steps must be positive")
        if (
            not isinstance(causal_qpos_prefix, torch.Tensor)
            or causal_qpos_prefix.ndim != 2
            or causal_qpos_prefix.shape[1] != 28
        ):
            raise ValueError(
                "causal_qpos_prefix must have shape [decision_frame+1,28]; "
                f"got {getattr(causal_qpos_prefix, 'shape', None)}"
            )
        if decision_frame < 0 or len(causal_qpos_prefix) != decision_frame + 1:
            raise ValueError(
                "causal_qpos_prefix must contain frames 0..decision_frame exactly; "
                f"got len={len(causal_qpos_prefix)}, decision_frame={decision_frame}"
            )

        offsets = torch.arange(history_steps - 1, -1, -1, dtype=torch.int64)
        history_ticks = decision_frame * _MOTION_TICKS - offsets * _PROPRIO_TICKS
        nonnegative = history_ticks >= 0
        source_index = torch.div(history_ticks.clamp_min(0), _MOTION_TICKS, rounding_mode="floor")
        source_index = source_index.clamp_max(decision_frame)

        # 只计算 H 个历史槽实际需要的源帧；最早再向左多取一帧，供其后向速度使用。
        # 调用接口仍要求完整的 0..decision causal prefix，因此不会接受 decision 之后的数据。
        nonnegative_source = source_index[nonnegative]
        minimum_source = int(nonnegative_source.min()) if len(nonnegative_source) else 0
        first_computed_source = max(minimum_source - 1, 0)
        timeline = _validate_qpos(causal_qpos_prefix[first_computed_source:])
        source_observation, source_valid = self.source_observations(timeline)
        local_source_index = source_index.clamp_min(first_computed_source) - first_computed_source
        history_valid = nonnegative & source_valid.index_select(0, local_source_index)
        history = source_observation.index_select(0, local_source_index).clone()
        history[~history_valid] = 0.0
        history_times = history_ticks.to(torch.float64) / float(_COMMON_TIMEBASE_HZ)
        return history.contiguous(), history_valid.contiguous(), history_times.contiguous()
