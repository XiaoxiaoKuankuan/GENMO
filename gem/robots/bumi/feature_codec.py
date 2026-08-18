"""The canonical 93D BUMI motion representation.

Feature slices live in this module and are imported everywhere else.  The
codec uses a first-frame, Z-up horizontal/yaw anchor and never performs IK.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from gem.utils.rotation_conversions import (
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_apply,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
)

from .kinematics import BumiKinematics

BUMI_QPOS_DIM = 28
BUMI_JOINT_DIM = 21
BUMI_FEATURE_DIM = 93
BUMI_FEATURE_SLICES: Mapping[str, tuple[int, int]] = MappingProxyType(
    {
        "root_pos_local": (0, 3),
        "root_rot_local": (3, 9),
        "joint_dof": (9, 30),
        "body_link_pos_local": (30, 93),
    }
)
BUMI_ANCHOR_MODE = "first_frame_xy_yaw_default_height"
BUMI_QUATERNION_CONVENTION = "wxyz"
BUMI_QPOS_ORDER = "mujoco_native"


@dataclass(frozen=True)
class BumiMotionComponents:
    root_pos_local: torch.Tensor
    root_rot_local_quat: torch.Tensor
    joint_dof: torch.Tensor
    body_link_pos_local: torch.Tensor


@dataclass(frozen=True)
class BumiCanonicalAnchor:
    position_w: torch.Tensor
    heading_quat_wxyz: torch.Tensor
    heading_inverse_quat_wxyz: torch.Tensor
    yaw: torch.Tensor
    default_root_height: torch.Tensor


@dataclass(frozen=True)
class BumiEncodedFeatures:
    physical_features: torch.Tensor
    canonical_qpos: torch.Tensor
    body_link_pos_local: torch.Tensor
    anchor: BumiCanonicalAnchor
    normalized_world_qpos: torch.Tensor


def normalize_quaternion_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    if not isinstance(quaternion, torch.Tensor) or quaternion.shape[-1] != 4:
        raise ValueError(
            f"wxyz quaternion must be a tensor with last dimension 4; "
            f"got {getattr(quaternion, 'shape', None)}"
        )
    if not bool(torch.isfinite(quaternion).all()):
        raise ValueError("wxyz quaternion contains NaN or Inf")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if bool((norm < 1.0e-8).any()):
        raise ValueError("wxyz quaternion contains a zero-length value")
    return quaternion / norm


def make_quaternion_continuous(
    quaternion: torch.Tensor, *, time_dim: int = -2
) -> torch.Tensor:
    """Normalize wxyz quaternions and flip temporal signs when dot < 0."""

    quaternion = normalize_quaternion_wxyz(quaternion)
    if quaternion.ndim < 2:
        return quaternion
    resolved_time_dim = time_dim if time_dim >= 0 else quaternion.ndim + time_dim
    if resolved_time_dim < 0 or resolved_time_dim >= quaternion.ndim - 1:
        raise ValueError(
            f"time_dim={time_dim} does not identify a sequence axis in {quaternion.shape}"
        )
    moved = quaternion.movedim(resolved_time_dim, -2)
    if moved.shape[-2] <= 1:
        return moved.movedim(-2, resolved_time_dim)
    transition = torch.where(
        (moved[..., 1:, :] * moved[..., :-1, :]).sum(dim=-1) < 0.0,
        moved.new_tensor(-1.0),
        moved.new_tensor(1.0),
    )
    signs = torch.cat(
        (torch.ones_like(transition[..., :1]), torch.cumprod(transition, dim=-1)), dim=-1
    )
    moved = moved * signs.unsqueeze(-1)
    return moved.movedim(-2, resolved_time_dim)


def quaternion_sign_is_continuous(
    quaternion: torch.Tensor, *, tolerance: float = 1.0e-6
) -> bool:
    quaternion = normalize_quaternion_wxyz(quaternion)
    if quaternion.shape[-2] <= 1:
        return True
    adjacent_dot = (quaternion[..., 1:, :] * quaternion[..., :-1, :]).sum(dim=-1)
    return bool((adjacent_dot >= -float(tolerance)).all())


class BumiMotionFeatureCodec:
    """Encode BUMI qpos28 and differentiable FK into physical 93D features."""

    feature_dim = BUMI_FEATURE_DIM
    feature_slices = BUMI_FEATURE_SLICES
    rotation_representation = "rot6d"
    anchor_mode = BUMI_ANCHOR_MODE

    def __init__(self, kinematics: BumiKinematics) -> None:
        if not isinstance(kinematics, BumiKinematics):
            raise TypeError("BumiMotionFeatureCodec requires BumiKinematics")
        self.kinematics = kinematics

    @property
    def default_root_height(self) -> torch.Tensor:
        return self.kinematics.default_qpos[2]

    @staticmethod
    def _heading_from_root_quaternion(
        root_quat_wxyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_quat_wxyz = normalize_quaternion_wxyz(root_quat_wxyz)
        rotation = quaternion_to_matrix(root_quat_wxyz)
        yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
        axis_angle = torch.zeros(
            (*yaw.shape, 3), dtype=root_quat_wxyz.dtype, device=root_quat_wxyz.device
        )
        axis_angle[..., 2] = yaw
        heading = normalize_quaternion_wxyz(axis_angle_to_quaternion(axis_angle))
        return yaw, heading, quaternion_invert(heading)

    def build_canonical_anchor(self, qpos: torch.Tensor) -> BumiCanonicalAnchor:
        if qpos.ndim < 2 or qpos.shape[-1] != 28 or qpos.shape[-2] <= 0:
            raise ValueError(f"qpos must have shape [...,T,28] with T > 0; got {qpos.shape}")
        qpos = self.normalize_qpos_sequence(qpos)
        first_pos = qpos[..., 0:1, :3]
        first_quat = qpos[..., 0:1, 3:7]
        yaw, heading, heading_inverse = self._heading_from_root_quaternion(first_quat)
        anchor_position = torch.cat(
            (
                first_pos[..., :2],
                torch.ones_like(first_pos[..., 2:3]) * self.default_root_height.to(qpos),
            ),
            dim=-1,
        )
        return BumiCanonicalAnchor(
            position_w=anchor_position,
            heading_quat_wxyz=heading,
            heading_inverse_quat_wxyz=heading_inverse,
            yaw=yaw,
            default_root_height=self.default_root_height.to(qpos),
        )

    @staticmethod
    def normalize_qpos_sequence(qpos: torch.Tensor) -> torch.Tensor:
        if not isinstance(qpos, torch.Tensor) or qpos.ndim < 2 or qpos.shape[-1] != 28:
            raise ValueError(
                f"BUMI qpos must have shape [...,T,28]; got {getattr(qpos, 'shape', None)}"
            )
        if not bool(torch.isfinite(qpos).all()):
            raise ValueError("BUMI qpos contains NaN or Inf")
        return torch.cat(
            (
                qpos[..., :3],
                make_quaternion_continuous(qpos[..., 3:7]),
                qpos[..., 7:],
            ),
            dim=-1,
        )

    @staticmethod
    def rotation_quat_to_features(quaternion: torch.Tensor) -> torch.Tensor:
        quaternion = normalize_quaternion_wxyz(quaternion)
        return matrix_to_rotation_6d(quaternion_to_matrix(quaternion))

    @staticmethod
    def rotation_features_to_quat(rot6d: torch.Tensor) -> torch.Tensor:
        if rot6d.shape[-1] != 6:
            raise ValueError(f"rot6d must have last dimension 6, got {rot6d.shape}")
        if not bool(torch.isfinite(rot6d).all()):
            raise ValueError("rot6d contains NaN or Inf")
        quaternion = normalize_quaternion_wxyz(
            matrix_to_quaternion(rotation_6d_to_matrix(rot6d))
        )
        if quaternion.ndim >= 2:
            quaternion = make_quaternion_continuous(quaternion)
        return quaternion

    def canonicalize(
        self,
        qpos: torch.Tensor,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor | None = None,
        *,
        anchor: BumiCanonicalAnchor | None = None,
    ) -> tuple[BumiMotionComponents, BumiCanonicalAnchor, torch.Tensor]:
        """Canonicalize one crop using its own first frame XY/yaw anchor."""

        qpos = self.normalize_qpos_sequence(qpos)
        expected_body_shape = (*qpos.shape[:-1], 22, 3)
        if tuple(body_pos_w.shape) != expected_body_shape:
            raise ValueError(
                f"body_pos_w must have shape {expected_body_shape}, got {tuple(body_pos_w.shape)}"
            )
        if not bool(torch.isfinite(body_pos_w).all()):
            raise ValueError("body_pos_w contains NaN or Inf")
        del body_quat_w  # Body orientations are not part of the 93D representation.
        anchor = self.build_canonical_anchor(qpos) if anchor is None else anchor
        root_pos_w = qpos[..., :3]
        root_quat_w = qpos[..., 3:7]
        heading_inverse = anchor.heading_inverse_quat_wxyz.expand(*root_pos_w.shape[:-1], 4)
        root_pos_local = quaternion_apply(
            heading_inverse, root_pos_w - anchor.position_w
        )
        root_rot_local = quaternion_multiply(heading_inverse, root_quat_w)
        root_rot_local = make_quaternion_continuous(root_rot_local)

        body_link_pos_w = body_pos_w[..., 1:, :]
        heading_body = anchor.heading_inverse_quat_wxyz.unsqueeze(-2).expand(
            *body_link_pos_w.shape[:-2], 21, 4
        )
        body_link_pos_local = quaternion_apply(
            heading_body, body_link_pos_w - anchor.position_w.unsqueeze(-2)
        )
        components = BumiMotionComponents(
            root_pos_local=root_pos_local,
            root_rot_local_quat=root_rot_local,
            joint_dof=qpos[..., 7:],
            body_link_pos_local=body_link_pos_local,
        )
        canonical_qpos = torch.cat(
            (
                root_pos_local,
                make_quaternion_continuous(root_rot_local),
                qpos[..., 7:],
            ),
            dim=-1,
        )
        return components, anchor, canonical_qpos

    def encode(self, qpos: torch.Tensor) -> BumiEncodedFeatures:
        normalized_qpos = self.normalize_qpos_sequence(qpos)
        fk = self.kinematics.forward_kinematics(normalized_qpos)
        components, anchor, canonical_qpos = self.canonicalize(
            normalized_qpos,
            fk["body_pos_w"],
            fk["body_quat_w"],
        )
        physical = self.assemble_features(components)
        return BumiEncodedFeatures(
            physical_features=physical,
            canonical_qpos=canonical_qpos,
            body_link_pos_local=components.body_link_pos_local,
            anchor=anchor,
            normalized_world_qpos=normalized_qpos,
        )

    def assemble_features(self, components: BumiMotionComponents) -> torch.Tensor:
        expected = {
            "root_pos_local": (components.root_pos_local, 3),
            "root_rot_local_quat": (components.root_rot_local_quat, 4),
            "joint_dof": (components.joint_dof, 21),
        }
        prefix = components.root_pos_local.shape[:-1]
        for name, (value, width) in expected.items():
            if value.shape[:-1] != prefix or value.shape[-1] != width:
                raise ValueError(
                    f"{name} must have shape {(*prefix, width)}, got {tuple(value.shape)}"
                )
        if tuple(components.body_link_pos_local.shape) != (*prefix, 21, 3):
            raise ValueError(
                "body_link_pos_local must have shape [...,21,3], got "
                f"{tuple(components.body_link_pos_local.shape)}"
            )
        result = torch.cat(
            (
                components.root_pos_local,
                self.rotation_quat_to_features(components.root_rot_local_quat),
                components.joint_dof,
                components.body_link_pos_local.flatten(start_dim=-2),
            ),
            dim=-1,
        )
        if result.shape[-1] != 93:
            raise RuntimeError(f"Internal BUMI feature dimension error: {result.shape}")
        return result

    def split_features(self, features: torch.Tensor) -> BumiMotionComponents:
        if not isinstance(features, torch.Tensor) or features.shape[-1] != 93:
            raise ValueError(
                f"BUMI features must be a tensor with last dimension 93; "
                f"got {getattr(features, 'shape', None)}"
            )
        if not bool(torch.isfinite(features).all()):
            raise ValueError("BUMI features contain NaN or Inf")
        slices = BUMI_FEATURE_SLICES
        root_rot6d = features[..., slices["root_rot_local"][0] : slices["root_rot_local"][1]]
        body_flat = features[
            ..., slices["body_link_pos_local"][0] : slices["body_link_pos_local"][1]
        ]
        return BumiMotionComponents(
            root_pos_local=features[
                ..., slices["root_pos_local"][0] : slices["root_pos_local"][1]
            ],
            root_rot_local_quat=self.rotation_features_to_quat(root_rot6d),
            joint_dof=features[..., slices["joint_dof"][0] : slices["joint_dof"][1]],
            body_link_pos_local=body_flat.reshape(*features.shape[:-1], 21, 3),
        )

    def decode_to_canonical_qpos(self, physical_features: torch.Tensor) -> torch.Tensor:
        components = self.split_features(physical_features)
        return torch.cat(
            (
                components.root_pos_local,
                make_quaternion_continuous(components.root_rot_local_quat),
                components.joint_dof,
            ),
            dim=-1,
        )

    def apply_world_anchor(
        self,
        canonical_qpos: torch.Tensor,
        world_anchor: Mapping[str, torch.Tensor | float | list[float]] | torch.Tensor,
    ) -> torch.Tensor:
        """Place canonical qpos at a desired world root XY/yaw (and optional anchor Z).

        Tensor anchors use ``[x, y, yaw]`` or ``[x, y, z, yaw]``.  Mapping
        anchors use ``root_xy``/``xy``, ``yaw``, and optional ``anchor_z``.
        """

        canonical_qpos = self.normalize_qpos_sequence(canonical_qpos)
        root_pos_local = canonical_qpos[..., :3]
        if isinstance(world_anchor, torch.Tensor):
            anchor_tensor = world_anchor.to(root_pos_local)
            if anchor_tensor.shape[-1] == 3:
                xy, yaw = anchor_tensor[..., :2], anchor_tensor[..., 2]
                anchor_z: torch.Tensor | float = self.default_root_height
            elif anchor_tensor.shape[-1] == 4:
                xy, anchor_z, yaw = (
                    anchor_tensor[..., :2],
                    anchor_tensor[..., 2],
                    anchor_tensor[..., 3],
                )
            else:
                raise ValueError("world_anchor tensor must end in [x,y,yaw] or [x,y,z,yaw]")
        else:
            xy_value = world_anchor.get("root_xy", world_anchor.get("xy"))
            if xy_value is None or "yaw" not in world_anchor:
                raise ValueError("world_anchor mapping requires root_xy (or xy) and yaw")
            xy = torch.as_tensor(xy_value, dtype=root_pos_local.dtype, device=root_pos_local.device)
            yaw = torch.as_tensor(
                world_anchor["yaw"], dtype=root_pos_local.dtype, device=root_pos_local.device
            )
            anchor_z = world_anchor.get("anchor_z", self.default_root_height)
        xy = torch.as_tensor(xy, dtype=root_pos_local.dtype, device=root_pos_local.device)
        yaw = torch.as_tensor(yaw, dtype=root_pos_local.dtype, device=root_pos_local.device)
        anchor_z = torch.as_tensor(
            anchor_z, dtype=root_pos_local.dtype, device=root_pos_local.device
        )
        if xy.shape[-1] != 2:
            raise ValueError(f"world anchor XY must have last dimension 2, got {xy.shape}")
        while xy.ndim < root_pos_local.ndim:
            xy = xy.unsqueeze(-2)
        while yaw.ndim < root_pos_local.ndim - 1:
            yaw = yaw.unsqueeze(-1)
        while anchor_z.ndim < root_pos_local.ndim - 1:
            anchor_z = anchor_z.unsqueeze(-1)
        xy = xy.expand(*root_pos_local.shape[:-1], 2)
        yaw = yaw.expand(root_pos_local.shape[:-1])
        anchor_z = anchor_z.expand(root_pos_local.shape[:-1])
        axis_angle = torch.zeros_like(root_pos_local)
        axis_angle[..., 2] = yaw
        heading = normalize_quaternion_wxyz(axis_angle_to_quaternion(axis_angle))
        anchor_position = torch.cat((xy, anchor_z.unsqueeze(-1)), dim=-1)
        world_root_pos = quaternion_apply(heading, root_pos_local) + anchor_position
        world_root_quat = quaternion_multiply(heading, canonical_qpos[..., 3:7])
        world_root_quat = make_quaternion_continuous(world_root_quat)
        return torch.cat((world_root_pos, world_root_quat, canonical_qpos[..., 7:]), dim=-1)


__all__ = [
    "BUMI_ANCHOR_MODE",
    "BUMI_FEATURE_DIM",
    "BUMI_FEATURE_SLICES",
    "BUMI_JOINT_DIM",
    "BUMI_QPOS_DIM",
    "BUMI_QPOS_ORDER",
    "BUMI_QUATERNION_CONVENTION",
    "BumiCanonicalAnchor",
    "BumiEncodedFeatures",
    "BumiMotionComponents",
    "BumiMotionFeatureCodec",
    "make_quaternion_continuous",
    "normalize_quaternion_wxyz",
    "quaternion_sign_is_continuous",
]
