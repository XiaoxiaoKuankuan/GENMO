"""Self-contained differentiable forward kinematics for the BUMI robot.

The module consumes a versioned JSON file exported from the real BUMI MJCF.
It intentionally has no MuJoCo dependency: MuJoCo is confined to the exporter,
offline parity validator, and renderer under ``tools/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from gem.utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
    standardize_quaternion,
)

KINEMATICS_CONTRACT_VERSION = "genmo.bumi_kinematics.v1"


def resolve_asset_path(path: str | Path) -> Path:
    """Resolve absolute paths or paths relative to the GENMO repository."""

    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path(__file__).resolve().parents[3] / value
    return value.resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_list(spec: dict[str, Any], key: str, length: int) -> list[Any]:
    value = spec.get(key)
    if not isinstance(value, list) or len(value) != length:
        actual = type(value).__name__ if not isinstance(value, list) else len(value)
        raise ValueError(f"kinematics.{key} must be a list of length {length}; got {actual}")
    return value


def _view_for_batch(value: torch.Tensor, batch_ndim: int) -> torch.Tensor:
    return value.view(*([1] * batch_ndim), *value.shape)


class BumiKinematics(nn.Module):
    """Differentiable FK for BUMI's free root and 21 one-DoF joints."""

    robot_name = "bumi"
    qpos_dim = 28
    num_joints = 21
    num_feature_bodies = 21
    num_bodies = 22

    def __init__(self, kinematics_path: str | Path) -> None:
        super().__init__()
        if path_text := str(kinematics_path).strip():
            path = resolve_asset_path(path_text)
        else:
            raise ValueError(
                "BUMI kinematics_path is required; export it from the real BUMI MJCF "
                "with tools/robots/export_bumi_kinematics.py"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"BUMI kinematics JSON does not exist: {path}. Export it from the real MJCF first."
            )
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid BUMI kinematics JSON {path}: {exc}") from exc
        if not isinstance(spec, dict):
            raise ValueError(f"BUMI kinematics spec must be a JSON object: {path}")
        self._validate_header(spec, path)

        self.kinematics_path = str(path)
        self.kinematics_sha256 = sha256_file(path)
        self.contract_version = str(spec["contract_version"])
        self.source_mjcf_sha256 = str(spec.get("source_mjcf_sha256", ""))
        self.root_body = str(spec["root_body"])
        self.root_link = self.root_body
        self.joint_order = tuple(str(value) for value in spec["joint_order"])
        self.feature_body_names = tuple(str(value) for value in spec["feature_body_names"])
        self.body_order = (self.root_body, *self.feature_body_names)
        self.joint_name_to_index = {name: index for index, name in enumerate(self.joint_order)}
        self.body_name_to_index = {name: index for index, name in enumerate(self.body_order)}

        parent = [int(value) for value in _require_list(spec, "parent_body_indices", 21)]
        child = [int(value) for value in _require_list(spec, "child_body_indices", 21)]
        expected_child = list(range(1, 22))
        if child != expected_child:
            raise ValueError(
                "kinematics.child_body_indices must be [1, ..., 21] in joint_order; "
                f"got {child}"
            )
        for index, parent_index in enumerate(parent):
            if not 0 <= parent_index < child[index]:
                raise ValueError(
                    "BUMI feature bodies must be topologically ordered: "
                    f"joint={self.joint_order[index]!r}, parent={parent_index}, child={child[index]}"
                )
        self._parent_body_indices_py = tuple(parent)
        self._child_body_indices_py = tuple(child)
        self.register_buffer(
            "parent_body_indices", torch.tensor(parent, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "child_body_indices", torch.tensor(child, dtype=torch.long), persistent=False
        )

        self.register_buffer(
            "joint_axes",
            self._float_tensor(spec, "joint_axes", (21, 3), path),
            persistent=False,
        )
        axis_norm = torch.linalg.vector_norm(self.joint_axes, dim=-1)
        if not torch.allclose(axis_norm, torch.ones_like(axis_norm), atol=1.0e-5, rtol=0.0):
            raise ValueError(f"BUMI joint axes must be unit length in {path}")
        self.register_buffer(
            "joint_origin_xyz",
            self._float_tensor(spec, "joint_origin_xyz", (21, 3), path),
            persistent=False,
        )
        origin_quat = self._float_tensor(spec, "joint_origin_quat_wxyz", (21, 4), path)
        origin_norm = torch.linalg.vector_norm(origin_quat, dim=-1)
        if not torch.allclose(origin_norm, torch.ones_like(origin_norm), atol=1.0e-5, rtol=0.0):
            raise ValueError(f"BUMI joint-origin quaternions must be normalized in {path}")
        self.register_buffer(
            "joint_origin_rot",
            quaternion_to_matrix(F.normalize(origin_quat, dim=-1)),
            persistent=False,
        )
        self.register_buffer(
            "joint_anchor_xyz",
            self._float_tensor(spec, "joint_anchor_xyz", (21, 3), path),
            persistent=False,
        )
        lower = self._float_tensor(spec, "joint_lower_limits", (21,), path)
        upper = self._float_tensor(spec, "joint_upper_limits", (21,), path)
        if not bool((lower < upper).all()):
            raise ValueError(f"Every BUMI joint limit must satisfy lower < upper: {path}")
        self.register_buffer("joint_lower_limits", lower, persistent=False)
        self.register_buffer("joint_upper_limits", upper, persistent=False)
        default_qpos = self._float_tensor(spec, "default_qpos", (28,), path)
        default_qpos = default_qpos.clone()
        default_quat_norm = torch.linalg.vector_norm(default_qpos[3:7])
        if float(default_quat_norm) < 1.0e-8:
            raise ValueError(f"BUMI default_qpos has a zero root quaternion in {path}")
        default_qpos[3:7] = default_qpos[3:7] / default_quat_norm
        if bool((default_qpos[7:] < lower).any()) or bool((default_qpos[7:] > upper).any()):
            raise ValueError(f"BUMI default_qpos joint values exceed exported ranges in {path}")
        self.register_buffer("default_qpos", default_qpos, persistent=False)

        addresses = [int(value) for value in _require_list(spec, "joint_qpos_addresses", 21)]
        if addresses != list(range(7, 28)):
            raise ValueError(
                "BUMI qpos must be MuJoCo-native free-root order with joint addresses 7..27; "
                f"got {addresses} in {path}"
            )
        self.joint_qpos_addresses = tuple(addresses)
        expected_indices = {name: index for index, name in enumerate(self.joint_order)}
        expected_addresses = {name: index + 7 for index, name in enumerate(self.joint_order)}
        if spec.get("joint_name_to_qpos_index") != expected_indices:
            raise ValueError(
                f"BUMI joint_name_to_qpos_index must exactly encode joint_order in {path}"
            )
        if spec.get("joint_name_to_qpos_address") != expected_addresses:
            raise ValueError(
                f"BUMI joint_name_to_qpos_address must exactly encode MuJoCo qpos addresses in {path}"
            )
        self._load_sole_proxies(spec, path)
        self.evaluation_proxies = tuple(spec.get("evaluation_proxies", ()))

    @staticmethod
    def _validate_header(spec: dict[str, Any], path: Path) -> None:
        required = {
            "contract_version",
            "robot_name",
            "qpos_dim",
            "joint_dim",
            "quaternion_convention",
            "qpos_order",
            "root_body",
            "joint_order",
            "feature_body_names",
            "source_mjcf_sha256",
        }
        missing = required - set(spec)
        if missing:
            raise ValueError(f"BUMI kinematics {path} is missing fields: {sorted(missing)}")
        expected = {
            "contract_version": KINEMATICS_CONTRACT_VERSION,
            "robot_name": "bumi",
            "qpos_dim": 28,
            "joint_dim": 21,
            "quaternion_convention": "wxyz",
            "qpos_order": "mujoco_native",
        }
        for key, expected_value in expected.items():
            if spec.get(key) != expected_value:
                raise ValueError(
                    f"BUMI kinematics {path}: {key} must be {expected_value!r}, "
                    f"got {spec.get(key)!r}"
                )
        source_mjcf_sha256 = str(spec["source_mjcf_sha256"])
        if len(source_mjcf_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in source_mjcf_sha256
        ):
            raise ValueError(
                f"BUMI kinematics {path}: source_mjcf_sha256 must be lowercase SHA-256"
            )
        joints = _require_list(spec, "joint_order", 21)
        bodies = _require_list(spec, "feature_body_names", 21)
        if len(set(map(str, joints))) != 21:
            raise ValueError(f"BUMI joint_order contains duplicates: {path}")
        if len(set(map(str, bodies))) != 21:
            raise ValueError(f"BUMI feature_body_names contains duplicates: {path}")
        body_order = spec.get("body_order")
        expected_bodies = [str(spec["root_body"]), *map(str, bodies)]
        if body_order != expected_bodies:
            raise ValueError(
                f"BUMI body_order must equal [root_body, *feature_body_names] in {path}"
            )

    @staticmethod
    def _float_tensor(
        spec: dict[str, Any], key: str, shape: tuple[int, ...], path: Path
    ) -> torch.Tensor:
        if key not in spec:
            raise ValueError(f"BUMI kinematics {path} is missing {key}")
        value = torch.as_tensor(spec[key], dtype=torch.float32)
        if tuple(value.shape) != shape:
            raise ValueError(
                f"BUMI kinematics {path}: {key} must have shape {shape}, "
                f"got {tuple(value.shape)}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"BUMI kinematics {path}: {key} contains NaN or Inf")
        return value

    def _load_sole_proxies(self, spec: dict[str, Any], path: Path) -> None:
        proxies = spec.get("sole_proxies")
        if not isinstance(proxies, list) or not proxies:
            raise ValueError(
                f"BUMI kinematics {path} must contain real left/right sole_proxies exported "
                "from the MJCF; no proxy names are guessed at runtime"
            )
        indices: list[int] = []
        positions: list[list[float]] = []
        radii: list[float] = []
        foot_ids: list[int] = []
        names: list[str] = []
        for proxy_index, item in enumerate(proxies):
            if not isinstance(item, dict):
                raise ValueError(f"sole_proxies[{proxy_index}] must be an object in {path}")
            required = {"name", "feature_body_index", "local_position", "radius", "foot_id"}
            missing = required - set(item)
            if missing:
                raise ValueError(
                    f"sole_proxies[{proxy_index}] in {path} is missing {sorted(missing)}"
                )
            body_index = int(item["feature_body_index"])
            foot_id = int(item["foot_id"])
            radius = float(item["radius"])
            local_position = [float(value) for value in item["local_position"]]
            if body_index <= 0 or body_index >= 22:
                raise ValueError(f"Invalid sole proxy feature_body_index={body_index} in {path}")
            if foot_id not in (0, 1):
                raise ValueError(f"sole proxy foot_id must be 0 (left) or 1 (right) in {path}")
            if len(local_position) != 3 or not torch.isfinite(torch.tensor(local_position)).all():
                raise ValueError(f"Invalid sole proxy local_position in {path}: {local_position}")
            if not torch.isfinite(torch.tensor(radius)) or radius < 0.0:
                raise ValueError(f"Invalid sole proxy radius={radius} in {path}")
            indices.append(body_index)
            positions.append(local_position)
            radii.append(radius)
            foot_ids.append(foot_id)
            names.append(str(item["name"]))
        if set(foot_ids) != {0, 1}:
            raise ValueError(f"BUMI sole proxies must cover both left and right feet in {path}")
        self.sole_proxy_names = tuple(names)
        self.register_buffer(
            "sole_proxy_body_indices", torch.tensor(indices, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "sole_proxy_local_positions",
            torch.tensor(positions, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sole_proxy_radii", torch.tensor(radii, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "sole_proxy_foot_ids", torch.tensor(foot_ids, dtype=torch.long), persistent=False
        )

    @property
    def joint_limits(self) -> torch.Tensor:
        return torch.stack((self.joint_lower_limits, self.joint_upper_limits), dim=-1)

    @staticmethod
    def body_quat_wxyz_to_matrix(body_quat_wxyz: torch.Tensor) -> torch.Tensor:
        return quaternion_to_matrix(F.normalize(body_quat_wxyz, dim=-1))

    @staticmethod
    def matrix_to_body_quat_wxyz(matrix: torch.Tensor) -> torch.Tensor:
        quat = F.normalize(matrix_to_quaternion(matrix), dim=-1)
        return standardize_quaternion(quat)

    def normalize_qpos(self, qpos: torch.Tensor) -> torch.Tensor:
        if not isinstance(qpos, torch.Tensor) or qpos.shape[-1] != 28:
            raise ValueError(
                f"BUMI qpos must be a tensor with last dimension 28; "
                f"got {getattr(qpos, 'shape', None)}"
            )
        if not bool(torch.isfinite(qpos).all()):
            raise ValueError("BUMI qpos contains NaN or Inf")
        quaternion = qpos[..., 3:7]
        quat_norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        if bool((quat_norm < 1.0e-8).any()):
            raise ValueError("BUMI qpos contains a zero-length root quaternion")
        return torch.cat(
            (qpos[..., :3], quaternion / quat_norm, qpos[..., 7:]), dim=-1
        )

    def clamp_joint_positions(self, joint_dof: torch.Tensor) -> torch.Tensor:
        if joint_dof.shape[-1] != 21:
            raise ValueError(f"joint_dof must have last dimension 21, got {joint_dof.shape}")
        return joint_dof.clamp(
            min=self.joint_lower_limits.to(joint_dof),
            max=self.joint_upper_limits.to(joint_dof),
        )

    def _split_qpos(
        self, qpos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qpos = self.normalize_qpos(qpos)
        root_pos = qpos[..., :3]
        root_quat = qpos[..., 3:7]
        return root_pos, root_quat, quaternion_to_matrix(root_quat), qpos[..., 7:]

    def _forward_body_pos_rot(self, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos, _root_quat, root_rot, joints = self._split_qpos(qpos)
        batch_ndim = qpos.ndim - 1
        axes = self.joint_axes.to(qpos)
        origins = self.joint_origin_xyz.to(qpos)
        origin_rotations = self.joint_origin_rot.to(qpos)
        anchors = self.joint_anchor_xyz.to(qpos)
        body_positions: list[torch.Tensor | None] = [None] * 22
        body_rotations: list[torch.Tensor | None] = [None] * 22
        body_positions[0] = root_pos
        body_rotations[0] = root_rot
        for joint_index in range(21):
            parent_index = self._parent_body_indices_py[joint_index]
            child_index = self._child_body_indices_py[joint_index]
            parent_pos = body_positions[parent_index]
            parent_rot = body_rotations[parent_index]
            if parent_pos is None or parent_rot is None:
                raise RuntimeError(
                    "BUMI kinematics are not topologically ordered at joint "
                    f"{self.joint_order[joint_index]!r}"
                )
            angle = joints[..., joint_index : joint_index + 1]
            axis = _view_for_batch(axes[joint_index], batch_ndim)
            joint_rot = axis_angle_to_matrix(axis * angle)
            origin_rot = _view_for_batch(origin_rotations[joint_index], batch_ndim)
            origin = _view_for_batch(origins[joint_index], batch_ndim)
            anchor = _view_for_batch(anchors[joint_index], batch_ndim)
            rotated_anchor = (joint_rot @ anchor.unsqueeze(-1)).squeeze(-1)
            local_position = origin + (
                origin_rot @ (anchor - rotated_anchor).unsqueeze(-1)
            ).squeeze(-1)
            body_positions[child_index] = parent_pos + (
                parent_rot @ local_position.unsqueeze(-1)
            ).squeeze(-1)
            body_rotations[child_index] = parent_rot @ origin_rot @ joint_rot
        if any(value is None for value in body_positions) or any(
            value is None for value in body_rotations
        ):
            raise RuntimeError("BUMI forward kinematics did not resolve every feature body")
        positions = torch.stack(
            [value for value in body_positions if value is not None], dim=-2
        )
        rotations = torch.stack(
            [value for value in body_rotations if value is not None], dim=-3
        )
        return positions, rotations

    def forward_body_positions(self, qpos: torch.Tensor) -> torch.Tensor:
        return self._forward_body_pos_rot(qpos)[0]

    def forward_kinematics(self, qpos: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map ``[..., 28]`` qpos to the ordered root + 21 feature bodies."""

        positions, rotations = self._forward_body_pos_rot(qpos)
        return {
            "body_pos_w": positions,
            "body_quat_w": self.matrix_to_body_quat_wxyz(rotations),
        }

    def forward_kinematics_full(self, qpos: torch.Tensor) -> dict[str, torch.Tensor]:
        positions, rotations = self._forward_body_pos_rot(qpos)
        batch_ndim = qpos.ndim - 1
        joint_positions: list[torch.Tensor] = []
        joint_axes: list[torch.Tensor] = []
        for joint_index, parent_index in enumerate(self._parent_body_indices_py):
            parent_pos = positions[..., parent_index, :]
            parent_rot = rotations[..., parent_index, :, :]
            origin = _view_for_batch(self.joint_origin_xyz.to(qpos)[joint_index], batch_ndim)
            origin_rot = _view_for_batch(self.joint_origin_rot.to(qpos)[joint_index], batch_ndim)
            anchor = _view_for_batch(self.joint_anchor_xyz.to(qpos)[joint_index], batch_ndim)
            axis = _view_for_batch(self.joint_axes.to(qpos)[joint_index], batch_ndim)
            joint_positions.append(
                parent_pos
                + (
                    parent_rot
                    @ (origin + (origin_rot @ anchor.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)
                ).squeeze(-1)
            )
            joint_axes.append(
                F.normalize((parent_rot @ origin_rot @ axis.unsqueeze(-1)).squeeze(-1), dim=-1)
            )
        return {
            "body_pos_w": positions,
            "body_rot_w": rotations,
            "body_quat_w": self.matrix_to_body_quat_wxyz(rotations),
            "joint_pos_w": torch.stack(joint_positions, dim=-2),
            "joint_axis_w": torch.stack(joint_axes, dim=-2),
        }

    def get_sole_proxy_points(
        self, body_pos_w: torch.Tensor, body_quat_w: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return real exported sole proxy centers, radii, IDs, and bottom heights."""

        expected_shape = (*body_pos_w.shape[:-2], 22, 3)
        if tuple(body_pos_w.shape) != expected_shape:
            raise ValueError(
                f"body_pos_w must have shape [...,22,3], got {tuple(body_pos_w.shape)}"
            )
        if tuple(body_quat_w.shape) != (*body_pos_w.shape[:-2], 22, 4):
            raise ValueError(
                f"body_quat_w must have shape [...,22,4], got {tuple(body_quat_w.shape)}"
            )
        body_dim = body_pos_w.ndim - 2
        indices = self.sole_proxy_body_indices.to(device=body_pos_w.device)
        positions = body_pos_w.index_select(body_dim, indices)
        rotations = quaternion_to_matrix(body_quat_w.index_select(body_dim, indices))
        offsets = self.sole_proxy_local_positions.to(body_pos_w)
        offsets = offsets.view(*([1] * (body_pos_w.ndim - 2)), *offsets.shape)
        points = positions + (rotations @ offsets.unsqueeze(-1)).squeeze(-1)
        radii = self.sole_proxy_radii.to(body_pos_w)
        radius_view = radii.view(*([1] * (points.ndim - 2)), -1)
        return {
            "points_w": points,
            "radii": radii,
            "foot_ids": self.sole_proxy_foot_ids.to(device=body_pos_w.device),
            "bottom_height": points[..., 2] - radius_view,
        }

    def aggregate_sole_by_foot(
        self, body_pos_w: torch.Tensor, body_quat_w: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Aggregate multiple exported proxies into ordered left/right foot signals."""

        proxies = self.get_sole_proxy_points(body_pos_w, body_quat_w)
        foot_points: list[torch.Tensor] = []
        foot_bottom: list[torch.Tensor] = []
        for foot_id in (0, 1):
            select = proxies["foot_ids"] == foot_id
            foot_points.append(proxies["points_w"][..., select, :].mean(dim=-2))
            foot_bottom.append(proxies["bottom_height"][..., select].amin(dim=-1))
        return {
            **proxies,
            "foot_points_w": torch.stack(foot_points, dim=-2),
            "foot_bottom_height": torch.stack(foot_bottom, dim=-1),
        }


__all__ = ["BumiKinematics", "KINEMATICS_CONTRACT_VERSION", "resolve_asset_path", "sha256_file"]
