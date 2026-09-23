"""BUMI closed-loop Stage 1 的独立监督样本构造路径。

本模块从一条完整、已配对且已经现有 BUMI reader 验收的 ``qpos28 + EDGE35`` 序列中，
在 decision frame 处分出三类互不混淆的数据：截至决策时刻可因果获得的 50 Hz
``proprio48`` 历史、未来 120 点 30 Hz 音乐/动作监督，以及位于该未来窗口开头的
teacher-forced committed qpos30 prefix。它复用现有 ``BumiMusicDatasetReader``、
``BumiMotionFeatureCodec`` 和版本化左右足接触标签，不修改旧 music-only Dataset 的返回值。

示范动作只有 30 Hz qpos，没有真实 GMT action 或动力学 rollout。为避免未来泄漏，本模块
对历史采用“最新已到达的 30 Hz 样本保持到 50 Hz 时间点”，速度只使用相邻 30 Hz 样本的
后向差分；第 0 个源样本因没有前驱而整体标为 history invalid。这里得到的是可从示范构造的
因果 proprio proxy，不冒充真实 GMT/BUMI rollout。它不包含 last_action、根线速度、根高或
接触观测，也不复用 GMT 69 维 normalizer。

qpos30 target 始终是 stats 标准化前的 physical 表示。构造时额外读取窗口右侧一个 qpos
样本作为根 XY 位移 halo；若该下一帧不存在，只把 ``target_qpos30[..., 0:2]`` 标无效，
不会用 codec 的 terminal-repeat 值冒充标签。known prefix 从有限零占位开始，仅复制逐坐标
mask 为真的 teacher-forced 值；contact2 只作为监督，不进入 prefix。本模块不改网络、不启动
训练/GMT/Isaac Lab，也不实现 Stage 2、Critic 或 DPPO。

地面只属于损失监督 provenance：已有 floor-zero 来源继续使用世界 Z=0；legacy body-origin
来源在完整未裁剪源序列上复用原接触标签器的足底分位估计，按源文件身份缓存一个标量。
该标量放在既有 meta 中，不新增条件字段，不改变 root Z、接触 payload 或 qpos30 anchor。
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

from gem.datasets.music_dance.music_dance_bumi import (
    BUMI_MUSIC_FPS,
    BumiMusicDatasetReader,
    duration_repeat_count,
)
from gem.robots.bumi.contacts import _resolve_ground_height, derive_bumi_foot_contact
from gem.robots.bumi.feature_codec import (
    BumiMotionFeatureCodec,
    make_quaternion_continuous,
)
from gem.robots.bumi.kinematics import BumiKinematics
from gem.utils.rotation_conversions import (
    matrix_to_axis_angle,
    quaternion_apply,
    quaternion_invert,
    quaternion_to_matrix,
)

from .contracts import (
    GMT_DEFAULT_JOINT_VEL_RAD_S,
    GMT_EXPECTED_JOINT_ORDER,
    GMT_NOMINAL_DEFAULT_JOINT_POS_RAD,
    MOTION_FPS,
    MOTION_WINDOW_FRAMES,
    PROPRIO_DIM,
    PROPRIO_FPS,
    PROPRIO_HISTORY_STEPS,
    QPOS30_DIM,
    STAGE1_CONDITION_KEYS,
    STAGE1_TARGET_KEYS,
    validate_stage1_training_batch,
)

CAUSAL_PROPRIO_CONSTRUCTION_VERSION = "genmo.bumi_demo_proprio48.causal_hold.v1"
PREFIX_SOURCE = "teacher_forced_demo_reference_v1"
STAGE1_GROUND_SUPERVISION_VERSION = "genmo.bumi_closedloop.full_sequence_ground.v1"
# 读取既有标签器默认值，防止新路径复制后悄悄漂移成另一套估地阈值。
STAGE1_CONTACT_GROUND_QUANTILE = float(
    inspect.signature(derive_bumi_foot_contact).parameters["ground_quantile"].default
)
STAGE1_FLOOR_ZERO_SEMANTICS = frozenset(
    {
        "gmr_foot_sole_ground_zero_v1",
        "robot_retargeter_floor_zero_v1",
        "umr_foot_sole_ground_zero_v1",
        "mixed_floor_zero_fk_contact_v2",
    }
)
_COMMON_TIMEBASE_HZ = 150
_MOTION_TICKS = _COMMON_TIMEBASE_HZ // MOTION_FPS
_PROPRIO_TICKS = _COMMON_TIMEBASE_HZ // PROPRIO_FPS


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

    def full_causal_50hz_timeline(
        self, qpos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """为 train-only stats 返回整条序列的非前视 50 Hz 时间线。"""

        timeline = _validate_qpos(qpos)
        source_observation, source_valid = self.source_observations(timeline)
        last_tick = (len(timeline) - 1) * _MOTION_TICKS
        target_ticks = torch.arange(0, last_tick + 1, _PROPRIO_TICKS, dtype=torch.int64)
        source_index = torch.div(target_ticks, _MOTION_TICKS, rounding_mode="floor")
        values = source_observation.index_select(0, source_index)
        valid = source_valid.index_select(0, source_index)
        times = target_ticks.to(torch.float64) / float(_COMMON_TIMEBASE_HZ)
        return values.contiguous(), valid.contiguous(), times.contiguous()


class BumiClosedLoopStage1Dataset(Dataset):
    """从完整 BUMI 音乐配对序列返回 Stage 1 条件和监督标签。"""

    def __init__(
        self,
        root: str | Path,
        dataset_name: str,
        kinematics_path: str | Path,
        split: str = "train",
        history_steps: int = PROPRIO_HISTORY_STEPS,
        prefix_min_frames: int = 0,
        prefix_max_frames: int = 0,
        prefix_random_seed: int = 42,
        duration_aware_sampling: bool = True,
        random_decision: bool | None = None,
        eval_decision_mode: str = "center",
        strict_alignment: bool = True,
        strict_contract: bool = True,
        require_quality_filter: bool = True,
        validate_payloads_on_init: bool = False,
        validate_source_hashes_on_init: bool = False,
        quaternion_norm_tolerance: float = 1.0e-3,
        joint_limit_tolerance: float = 1.0e-3,
        limit_size: int | None = None,
    ) -> None:
        super().__init__()
        self.motion_frames = MOTION_WINDOW_FRAMES
        self.history_steps = int(history_steps)
        self.prefix_min_frames = int(prefix_min_frames)
        self.prefix_max_frames = int(prefix_max_frames)
        self.prefix_random_seed = int(prefix_random_seed)
        self.dataset_name = str(dataset_name)
        self.split = str(split)
        self.duration_aware_sampling = bool(duration_aware_sampling)
        self.random_decision = (
            self.split == "train" if random_decision is None else bool(random_decision)
        )
        self.eval_decision_mode = str(eval_decision_mode)
        self.limit_size = limit_size
        if self.history_steps <= 0:
            raise ValueError("history_steps must be positive")
        if self.limit_size is not None and int(self.limit_size) <= 0:
            raise ValueError("limit_size must be positive when provided")
        if not 0 <= self.prefix_min_frames <= self.prefix_max_frames:
            raise ValueError("prefix frames require 0 <= min <= max")
        if self.prefix_max_frames >= MOTION_WINDOW_FRAMES:
            raise ValueError(
                f"prefix_max_frames must be < {MOTION_WINDOW_FRAMES} so unknown future remains"
            )
        if self.eval_decision_mode not in {"start", "center", "end"}:
            raise ValueError("eval_decision_mode must be 'start', 'center', or 'end'")

        self.kinematics = BumiKinematics(kinematics_path)
        self.reader = BumiMusicDatasetReader(
            root,
            self.dataset_name,
            self.split,
            self.kinematics,
            strict_alignment=strict_alignment,
            strict_contract=strict_contract,
            require_quality_filter=require_quality_filter,
            quaternion_norm_tolerance=quaternion_norm_tolerance,
            joint_limit_tolerance=joint_limit_tolerance,
            validate_payloads_on_init=validate_payloads_on_init,
            validate_source_hashes_on_init=validate_source_hashes_on_init,
        )
        self.root = self.reader.root
        self.rows = self.reader.rows
        self.codec = BumiMotionFeatureCodec(self.kinematics)
        self.proprio_builder = CausalDemoProprio48Builder(self.kinematics)
        # 只缓存标量与来源，不保存完整 FK 或动作；文件变化会触发重新估计。
        self._ground_supervision_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.idx2meta: list[int] = []
        for row_index, row in enumerate(self.rows):
            repeats = (
                duration_repeat_count(int(row["num_frames"]), MOTION_WINDOW_FRAMES)
                if self.duration_aware_sampling
                else 1
            )
            self.idx2meta.extend([row_index] * repeats)

    def __len__(self) -> int:
        length = len(self.idx2meta)
        return min(length, int(self.limit_size)) if self.limit_size is not None else length

    @staticmethod
    def _zero_pad(value: torch.Tensor, target_length: int) -> torch.Tensor:
        if value.shape[0] > target_length:
            raise ValueError(f"cannot pad length {value.shape[0]} to shorter {target_length}")
        output = value.new_zeros((target_length, *value.shape[1:]))
        output[: value.shape[0]] = value
        return output.contiguous()

    def _select_decision_frame(self, sequence_length: int) -> int:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.random_decision:
            return int(np.random.randint(0, sequence_length))
        if self.eval_decision_mode == "start":
            return 0
        if self.eval_decision_mode == "end":
            return sequence_length - 1
        return (sequence_length - 1) // 2

    def _requested_prefix_frames(self, sample_id: str, decision_frame: int) -> int:
        if self.prefix_min_frames == self.prefix_max_frames:
            return self.prefix_min_frames
        key = (
            f"{self.prefix_random_seed}\0{self.dataset_name}\0{self.split}\0"
            f"{sample_id}\0{int(decision_frame)}"
        ).encode()
        draw = int.from_bytes(hashlib.sha256(key).digest()[:8], byteorder="big")
        width = self.prefix_max_frames - self.prefix_min_frames + 1
        return self.prefix_min_frames + draw % width

    def _contact_timeline(
        self, sequence: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        qpos = sequence["qpos"]
        supplied = sequence.get("foot_contact")
        available = sequence.get("foot_contact_available")
        if isinstance(supplied, torch.Tensor) and isinstance(available, torch.Tensor):
            supplied = supplied.float()
            available = available.bool()
            if bool(available.all()):
                return (
                    supplied.contiguous(),
                    available[:, None].expand(-1, 2).contiguous(),
                    str(sequence.get("foot_contact_source", "versioned_motion_payload")),
                )

        semantics = str(self.reader.dataset_info["ground_semantics"])
        estimate_ground = semantics == "legacy_body_origin_min_zero"
        derived = derive_bumi_foot_contact(
            qpos,
            self.kinematics,
            valid_mask=torch.ones(len(qpos), dtype=torch.bool),
            fps=BUMI_MUSIC_FPS,
            ground_height=0.0,
            estimate_ground_mask=torch.tensor(estimate_ground),
        )
        if not isinstance(supplied, torch.Tensor):
            return derived.contact, derived.valid_mask, "derived_from_full_sequence_qpos_fk"
        if not isinstance(available, torch.Tensor) or available.shape != (len(qpos),):
            raise ValueError("foot_contact_available must have shape [T]")
        merged = torch.where(available[:, None], supplied.to(derived.contact), derived.contact)
        return merged.contiguous(), derived.valid_mask, "payload_with_fk_fallback"

    @torch.no_grad()
    def _ground_supervision(self, sequence: Mapping[str, Any]) -> dict[str, Any]:
        """返回损失专用世界地面与可复核来源，绝不从当前 crop 估计地面。"""

        semantics = str(self.reader.dataset_info["ground_semantics"])
        if semantics not in STAGE1_FLOOR_ZERO_SEMANTICS | {"legacy_body_origin_min_zero"}:
            raise ValueError(f"Stage1 unsupported ground semantics: {semantics!r}")
        qpos = sequence["qpos"]
        source_frames = int(sequence["source_lengths"]["qpos"])
        if semantics == "legacy_body_origin_min_zero" and source_frames != len(qpos):
            raise ValueError(
                "legacy ground supervision requires the full uncropped source sequence"
            )
        motion_path = Path(sequence["motion_path"]).resolve()
        stat = motion_path.stat()
        source_sha = str(sequence["row"]["source_motion_sha256"])
        cache_key = (
            str(motion_path),
            stat.st_mtime_ns,
            stat.st_size,
            source_sha,
            self.kinematics.kinematics_sha256,
            source_frames,
            semantics,
        )
        if cache_key in self._ground_supervision_cache:
            return dict(self._ground_supervision_cache[cache_key])

        height = 0.0
        ground_quantile = None
        method = "explicit_floor_zero"
        if semantics == "legacy_body_origin_min_zero":
            # 与 derive_bumi_foot_contact 默认 ground_quantile=0.02 的同一个实现；
            # 不计算或替换已有 contact 标签，且不对验证集拟合归一化统计量。
            ground_quantile = STAGE1_CONTACT_GROUND_QUANTILE
            fk = self.kinematics.forward_kinematics(qpos)
            sole = self.kinematics.aggregate_sole_by_foot(fk["body_pos_w"], fk["body_quat_w"])
            height = float(
                _resolve_ground_height(
                    sole["foot_bottom_height"],
                    torch.ones(len(qpos), dtype=torch.bool, device=qpos.device),
                    0.0,
                    ground_quantile,
                    torch.tensor(True, device=qpos.device),
                )
            )
            method = "full_sequence_fk_sole_quantile"
        result = {
            "contract_version": STAGE1_GROUND_SUPERVISION_VERSION,
            "ground_semantics": semantics,
            "ground_height_world_m": height,
            "method": method,
            "ground_quantile": ground_quantile,
            "source_sequence_frames": source_frames,
            "source_motion_path": str(motion_path),
            "source_motion_sha256": source_sha,
            "kinematics_sha256": self.kinematics.kinematics_sha256,
            "scope": "loss_supervision_only_not_actor_condition",
        }
        self._ground_supervision_cache[cache_key] = result
        return dict(result)

    def get_window(self, row_index: int, start_frame: int | None = None) -> dict[str, Any]:
        """构造一条样本；``start_frame`` 即 decision frame，可显式落在序列尾部。"""

        if not 0 <= int(row_index) < len(self.rows):
            raise IndexError(row_index)
        row = self.rows[int(row_index)]
        sequence = self.reader.load_aligned_sequence(row)
        qpos = sequence["qpos"]
        music = sequence["music"]
        sequence_length = int(qpos.shape[0])
        decision_frame = (
            self._select_decision_frame(sequence_length)
            if start_frame is None
            else int(start_frame)
        )
        if not 0 <= decision_frame < sequence_length:
            raise ValueError(
                f"{row['sample_id']}: decision frame {decision_frame} outside "
                f"[0,{sequence_length - 1}]"
            )

        future_count = min(MOTION_WINDOW_FRAMES, sequence_length - decision_frame)
        future_valid = torch.zeros(MOTION_WINDOW_FRAMES, dtype=torch.bool)
        future_valid[:future_count] = True
        music_features = self._zero_pad(
            music[decision_frame : decision_frame + future_count], MOTION_WINDOW_FRAMES
        )
        music_valid = future_valid.clone()

        # 额外读取第 121 个 qpos，只为给窗口内第 120 帧的 root delta 提供真实 t+1 halo。
        qpos_halo = qpos[
            decision_frame : min(sequence_length, decision_frame + MOTION_WINDOW_FRAMES + 1)
        ]
        encoded_halo = self.codec.encode(qpos_halo).physical_features
        target_qpos30 = torch.zeros(MOTION_WINDOW_FRAMES, QPOS30_DIM, dtype=torch.float32)
        target_qpos30_valid = torch.zeros(MOTION_WINDOW_FRAMES, QPOS30_DIM, dtype=torch.bool)
        target_qpos30[:future_count] = encoded_halo[:future_count]
        target_qpos30_valid[:future_count, 2:] = True
        real_delta_count = min(future_count, max(sequence_length - decision_frame - 1, 0))
        target_qpos30_valid[:real_delta_count, :2] = True
        target_qpos30[~target_qpos30_valid] = 0.0

        contact_timeline, contact_timeline_valid, contact_source = self._contact_timeline(sequence)
        target_contact = torch.zeros(MOTION_WINDOW_FRAMES, 2, dtype=torch.float32)
        target_contact_valid = torch.zeros(MOTION_WINDOW_FRAMES, 2, dtype=torch.bool)
        target_contact[:future_count] = contact_timeline[
            decision_frame : decision_frame + future_count
        ]
        target_contact_valid[:future_count] = contact_timeline_valid[
            decision_frame : decision_frame + future_count
        ]
        target_contact[~target_contact_valid] = 0.0

        history, history_valid, history_times = self.proprio_builder.build_history(
            qpos[: decision_frame + 1],
            decision_frame=decision_frame,
            history_steps=self.history_steps,
        )
        decision_time = torch.tensor(decision_frame / float(MOTION_FPS), dtype=torch.float64)
        future_times = (
            decision_frame + torch.arange(MOTION_WINDOW_FRAMES, dtype=torch.float64)
        ) / float(MOTION_FPS)

        requested_prefix = self._requested_prefix_frames(str(row["sample_id"]), decision_frame)
        effective_prefix = min(requested_prefix, max(future_count - 1, 0))
        known_qpos30 = torch.zeros_like(target_qpos30)
        known_qpos30_mask = torch.zeros_like(target_qpos30_valid)
        if effective_prefix > 0:
            known_qpos30_mask[:effective_prefix, 2:] = target_qpos30_valid[:effective_prefix, 2:]
            if effective_prefix > 1:
                known_qpos30_mask[: effective_prefix - 1, :2] = target_qpos30_valid[
                    : effective_prefix - 1, :2
                ]
            known_qpos30[known_qpos30_mask] = target_qpos30[known_qpos30_mask]

        sample: dict[str, Any] = {
            "music_features": music_features,
            "music_valid": music_valid,
            "proprio_history": history,
            "proprio_history_valid": history_valid,
            "proprio_history_times": history_times,
            "known_qpos30": known_qpos30,
            "known_qpos30_mask": known_qpos30_mask,
            "future_valid": future_valid,
            "future_times": future_times,
            "decision_time": decision_time,
            "target_qpos30": target_qpos30,
            "target_qpos30_valid": target_qpos30_valid,
            "target_contact": target_contact,
            "target_contact_valid": target_contact_valid,
            "prefix_frames": torch.tensor(effective_prefix, dtype=torch.int64),
            "meta": {
                "dataset_id": self.dataset_name,
                "sample_id": str(row["sample_id"]),
                "sequence_id": str(row["sequence_id"]),
                "split": self.split,
                "decision_frame": decision_frame,
                "decision_time_seconds": float(decision_time),
                "source_sequence_frames": sequence_length,
                "future_valid_frames": future_count,
                "requested_prefix_frames": requested_prefix,
                "effective_prefix_frames": effective_prefix,
                "prefix_source": PREFIX_SOURCE,
                "proprio_source": "causal_demo_kinematic_proxy_not_actual_gmt_rollout",
                "proprio_construction_version": CAUSAL_PROPRIO_CONSTRUCTION_VERSION,
                "proprio_joint_default": "gmt_nominal_default_no_startup_randomization",
                "contact_source": contact_source,
                "ground_semantics": self.reader.dataset_info["ground_semantics"],
                "ground_supervision": self._ground_supervision(sequence),
                "qpos30_value_domain": "physical_before_existing_stats_normalization",
                "source_manifest": str(self.reader.manifest_path),
                "motion_path": str(sequence["motion_path"]),
                "music_feature_path": str(sequence["music_path"]),
            },
        }
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get_window(self.idx2meta[index])


def collate_stage1_training_samples(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """堆叠 Stage 1 样本并立即执行动态 H 的训练 batch 契约校验。"""

    if not batch:
        raise ValueError("Stage 1 collate received an empty batch")
    tensor_keys = (*STAGE1_CONDITION_KEYS, *STAGE1_TARGET_KEYS, "prefix_frames")
    required = set(tensor_keys) | {"meta"}
    for index, item in enumerate(batch):
        missing = required - set(item)
        unknown = set(item) - required
        if missing or unknown:
            raise ValueError(
                f"Stage 1 sample {index} fields mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
    result = {key: default_collate([item[key] for item in batch]) for key in tensor_keys}
    result["meta"] = [item["meta"] for item in batch]
    result["B"] = len(batch)
    history_steps = int(result["proprio_history"].shape[1])
    validate_stage1_training_batch(result, history_steps=history_steps)
    return result


__all__ = [
    "CAUSAL_PROPRIO_CONSTRUCTION_VERSION",
    "PREFIX_SOURCE",
    "STAGE1_GROUND_SUPERVISION_VERSION",
    "STAGE1_FLOOR_ZERO_SEMANTICS",
    "BumiClosedLoopStage1Dataset",
    "CausalDemoProprio48Builder",
    "collate_stage1_training_samples",
]
