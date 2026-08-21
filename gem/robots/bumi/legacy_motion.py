"""GMR 生产 BUMI3 重定向动作的 legacy 序列化契约适配。

当前 ``data/motions`` 下的 pickle 不是 ``genmo.bumi_music.v1`` 正式训练格式，
而是参考 GMR 脚本直接保存的 legacy 中间结果。这里的 ``legacy`` 只表示 pickle
序列化契约，不表示机器人资产是旧版；当前数据使用的正是用户确认的生产 BUMI3。
该中间格式有两个容易造成静默错误的差异：

* 根四元数按 ``xyzw`` 保存，而 GENMO/MuJoCo 正式 qpos 使用 ``wxyz``；
* 21 个关节按生产 MJCF qpos 顺序保存，不能假设与其他 BUMI kinematics 顺序相同。

本模块把不安全、易混淆的解析集中在一个边界内：只允许读取受信任的本地 pickle，
兼容 NumPy 2 生成而由 NumPy 1 环境读取的数组，严格检查字段/shape/body 顺序，
并提供显式的四元数转换、按关节名重排以及 25 个生产 link 世界坐标重建。质量筛选
和后续正式数据转换都应复用这里的契约，不能在各脚本中各自猜测格式。
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LEGACY_BUMI_MOTION_CONTRACT_VERSION = "genmo.bumi_legacy_motion.v1"
LEGACY_BUMI_QUATERNION_CONVENTION = "xyzw"

# 该顺序来自生成 data/motions 的 GMR 生产 bumi3.xml 的 qpos address 7..27。
LEGACY_BUMI_JOINT_ORDER = (
    "l_leg_pitch_joint",
    "l_leg_roll_joint",
    "l_leg_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_leg_pitch_joint",
    "r_leg_roll_joint",
    "r_leg_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "waist_yaw_joint",
    "l_arm_pitch_joint",
    "l_arm_roll_joint",
    "l_arm_yaw_joint",
    "l_elbow_pitch_joint",
    "r_arm_pitch_joint",
    "r_arm_roll_joint",
    "r_arm_yaw_joint",
    "r_elbow_pitch_joint",
)

# local_body_pos 的 25 个 body；三个 ``*_virtual`` link 只用于离线质量筛选。
LEGACY_BUMI_BODY_ORDER = (
    "base_link",
    "l_leg_pitch_link",
    "l_leg_roll_link",
    "l_leg_yaw_link",
    "l_knee_pitch_link",
    "l_ankle_pitch_link",
    "l_ankle_roll_link",
    "r_leg_pitch_link",
    "r_leg_roll_link",
    "r_leg_yaw_link",
    "r_knee_pitch_link",
    "r_ankle_pitch_link",
    "r_ankle_roll_link",
    "waist_yaw_link",
    "torso_link_virtual",
    "l_arm_pitch_link",
    "l_arm_roll_link",
    "l_arm_yaw_link",
    "l_elbow_pitch_link",
    "l_arm_hand_link_virtual",
    "r_arm_pitch_link",
    "r_arm_roll_link",
    "r_arm_yaw_link",
    "r_elbow_pitch_link",
    "r_arm_hand_link_virtual",
)


class _NumpyCompatibleUnpickler(pickle.Unpickler):
    """兼容 NumPy 2 ``numpy._core`` 路径的受信任本地 pickle reader。"""

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module == "numpy._core" or module.startswith("numpy._core."):
                legacy_module = "numpy.core" + module[len("numpy._core") :]
                return super().find_class(legacy_module, name)
            raise


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """返回文件内容 SHA256，供报告和恢复校验使用。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_finite_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int | None, ...],
    path: Path,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {name} cannot be converted to float64") from exc
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected for actual, expected in zip(array.shape, shape)
    ):
        raise ValueError(f"{path}: {name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: {name} contains NaN or Inf")
    return np.ascontiguousarray(array)


def normalize_xyzw(quaternion: np.ndarray, *, source: str = "quaternion") -> np.ndarray:
    """归一化 ``[...,4]`` xyzw 四元数，零范数输入直接失败。"""

    value = np.asarray(quaternion, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != 4:
        raise ValueError(f"{source} must have shape [...,4], got {value.shape}")
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norm < 1.0e-8):
        raise ValueError(f"{source} contains non-finite or zero-norm quaternion")
    return value / norm


def canonicalize_xyzw_sequence(quaternion: np.ndarray) -> np.ndarray:
    """在时间轴上消除表示同一旋转的 q/-q 符号跳变。"""

    output = normalize_xyzw(quaternion).copy()
    if output.ndim != 2:
        raise ValueError(f"quaternion sequence must have shape [T,4], got {output.shape}")
    for frame in range(1, len(output)):
        if float(np.dot(output[frame - 1], output[frame])) < 0.0:
            output[frame] *= -1.0
    return output


def xyzw_to_wxyz(quaternion: np.ndarray, *, canonicalize: bool = True) -> np.ndarray:
    """显式把旧 ``xyzw`` 序列转换成 GENMO/MuJoCo ``wxyz``。"""

    value = canonicalize_xyzw_sequence(quaternion) if canonicalize else normalize_xyzw(quaternion)
    return np.ascontiguousarray(value[..., [3, 0, 1, 2]])


def xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """将归一化 xyzw 四元数转换为旋转矩阵，不依赖 SciPy。"""

    value = normalize_xyzw(quaternion)
    x, y, z, w = np.moveaxis(value, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    matrix = np.empty((*value.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    matrix[..., 0, 1] = 2.0 * (xy - wz)
    matrix[..., 0, 2] = 2.0 * (xz + wy)
    matrix[..., 1, 0] = 2.0 * (xy + wz)
    matrix[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    matrix[..., 1, 2] = 2.0 * (yz - wx)
    matrix[..., 2, 0] = 2.0 * (xz - wy)
    matrix[..., 2, 1] = 2.0 * (yz + wx)
    matrix[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return matrix


def reorder_joints(
    values: np.ndarray,
    source_order: Sequence[str],
    target_order: Sequence[str],
) -> np.ndarray:
    """按完整关节名集合重排最后一维；缺失、重复或多余关节均失败。"""

    source = tuple(map(str, source_order))
    target = tuple(map(str, target_order))
    if len(source) != len(set(source)) or len(target) != len(set(target)):
        raise ValueError("source_order and target_order must not contain duplicate names")
    if set(source) != set(target):
        raise ValueError(
            "source/target joint name sets differ: "
            f"missing={sorted(set(target) - set(source))}, "
            f"extra={sorted(set(source) - set(target))}"
        )
    array = np.asarray(values)
    if array.shape[-1] != len(source):
        raise ValueError(f"joint values last dimension must be {len(source)}, got {array.shape}")
    lookup = {name: index for index, name in enumerate(source)}
    return np.ascontiguousarray(array[..., [lookup[name] for name in target]])


@dataclass(frozen=True)
class LegacyBumiMotion:
    """经过严格解析、仍保留旧 source 顺序和 xyzw 语义的一条动作。"""

    path: Path
    fps: int
    root_pos: np.ndarray
    root_rot_xyzw: np.ndarray
    dof_pos: np.ndarray
    local_body_pos: np.ndarray
    body_names: tuple[str, ...]
    declared_dof_names: tuple[str, ...] | None = None
    quality: Mapping[str, Any] | None = None

    @property
    def num_frames(self) -> int:
        return int(self.root_pos.shape[0])

    @property
    def body_name_to_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.body_names)}

    def root_rotation_matrices(self) -> np.ndarray:
        return xyzw_to_matrix(self.root_rot_xyzw)

    def world_body_positions(self) -> np.ndarray:
        """用保存的局部 FK 和世界根变换精确恢复 ``[T,25,3]``。"""

        rotation = self.root_rotation_matrices()
        rotated = np.einsum("tij,tbj->tbi", rotation, self.local_body_pos)
        return rotated + self.root_pos[:, None, :]

    def root_tilt_degrees(self) -> np.ndarray:
        """返回根局部 Z 轴相对世界 Z 轴的夹角，范围为 0..180 度。"""

        local_z_world = self.root_rotation_matrices()[..., :, 2]
        cosine = np.clip(local_z_world[..., 2], -1.0, 1.0)
        return np.rad2deg(np.arccos(cosine))

    def qpos_wxyz(self, target_joint_order: Sequence[str]) -> np.ndarray:
        """组成显式 wxyz、按目标关节名排序的 ``[T,28]`` qpos。"""

        joints = reorder_joints(
            self.dof_pos,
            source_order=LEGACY_BUMI_JOINT_ORDER,
            target_order=target_joint_order,
        )
        return np.concatenate((self.root_pos, xyzw_to_wxyz(self.root_rot_xyzw), joints), axis=-1)


def load_legacy_bumi_motion(
    path: str | Path,
    *,
    expected_fps: int = 30,
    expected_body_order: Iterable[str] = LEGACY_BUMI_BODY_ORDER,
) -> LegacyBumiMotion:
    """读取一条受信任的旧 BUMI pickle，并执行完整结构/数值契约检查。"""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    # pickle 可执行任意 Python 对象，因此该 reader 只适用于本地可信重定向产物。
    with source.open("rb") as handle:
        payload = _NumpyCompatibleUnpickler(handle).load()
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: legacy motion payload must be a dictionary")
    required = {"fps", "root_pos", "root_rot", "dof_pos", "local_body_pos", "link_body_list"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{source}: missing legacy fields {sorted(missing)}")
    try:
        fps_value = float(np.asarray(payload["fps"]).item())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: fps must be a scalar") from exc
    if not np.isfinite(fps_value) or not np.isclose(fps_value, expected_fps):
        raise ValueError(f"{source}: fps must be {expected_fps}, got {fps_value!r}")

    root_pos = _as_finite_array(payload["root_pos"], name="root_pos", shape=(None, 3), path=source)
    frames = int(root_pos.shape[0])
    if frames <= 0:
        raise ValueError(f"{source}: motion must contain at least one frame")
    root_rot = _as_finite_array(
        payload["root_rot"], name="root_rot", shape=(frames, 4), path=source
    )
    dof_pos = _as_finite_array(
        payload["dof_pos"],
        name="dof_pos",
        shape=(frames, len(LEGACY_BUMI_JOINT_ORDER)),
        path=source,
    )
    body_names = tuple(map(str, payload["link_body_list"]))
    expected_names = tuple(map(str, expected_body_order))
    if body_names != expected_names:
        raise ValueError(
            f"{source}: link_body_list does not match the legacy BUMI body contract; "
            f"expected={expected_names}, got={body_names}"
        )
    local_body_pos = _as_finite_array(
        payload["local_body_pos"],
        name="local_body_pos",
        shape=(frames, len(body_names), 3),
        path=source,
    )
    quaternion_norm = np.linalg.norm(root_rot, axis=-1)
    if np.any(quaternion_norm < 1.0e-8):
        raise ValueError(f"{source}: root_rot contains a zero-norm quaternion")
    declared_dof_names: tuple[str, ...] | None = None
    if "dof_names" in payload:
        declared_dof_names = tuple(map(str, payload["dof_names"]))
        if declared_dof_names != LEGACY_BUMI_JOINT_ORDER:
            raise ValueError(
                f"{source}: dof_names does not match the legacy BUMI joint contract; "
                f"expected={LEGACY_BUMI_JOINT_ORDER}, got={declared_dof_names}"
            )
    quality = payload.get("quality")
    if quality is not None and not isinstance(quality, Mapping):
        raise ValueError(f"{source}: quality must be a mapping when present")
    return LegacyBumiMotion(
        path=source,
        fps=int(expected_fps),
        root_pos=root_pos,
        root_rot_xyzw=root_rot,
        dof_pos=dof_pos,
        local_body_pos=local_body_pos,
        body_names=body_names,
        declared_dof_names=declared_dof_names,
        quality=quality,
    )
