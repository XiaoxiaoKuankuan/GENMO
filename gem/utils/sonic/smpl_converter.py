# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Convert GEM SMPL parameters to SONIC's FK joint definition."""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


class InvalidSMPLFrameError(ValueError):
    """A recoverable invalid source pose that should be dropped."""


def _shape_text(shape: torch.Size | tuple[int, ...]) -> str:
    if not shape:
        return "()"
    suffix = "," if len(shape) == 1 else ""
    return f"({','.join(map(str, shape))}{suffix})"


def _require_finite(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Reject a non-finite tensor with enough detail to locate its source."""
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return tensor

    nan_count = int(torch.isnan(tensor).sum().item())
    posinf_count = int(torch.isposinf(tensor).sum().item())
    neginf_count = int(torch.isneginf(tensor).sum().item())
    raise InvalidSMPLFrameError(
        f"{name} contains non-finite values:\n"
        f"shape={_shape_text(tensor.shape)}, nan={nan_count}, "
        f"posinf={posinf_count}, neginf={neginf_count}"
    )


def _normalize_quaternion_safe(
    quaternion: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Validate and normalize a scalar-first quaternion without hiding errors."""
    if quaternion.ndim < 1 or quaternion.shape[-1] != 4:
        raise ValueError(f"{name} must end in 4 quaternion components, got {quaternion.shape}")

    _require_finite(quaternion, name)
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    _require_finite(norm, f"{name} norm")
    near_zero = norm < 1e-8
    if bool(near_zero.any()):
        raise InvalidSMPLFrameError(
            f"{name} contains a zero or near-zero quaternion norm:\n"
            f"shape={_shape_text(quaternion.shape)}, "
            f"near_zero={int(near_zero.sum().item())}, "
            f"min_norm={float(norm.min().item()):.9g}"
        )

    normalized = quaternion / norm
    _require_finite(normalized, f"{name} normalized")
    return normalized.contiguous()


def _as_rows(value: Any, width: int, name: str) -> torch.Tensor:
    """Convert an array-like SMPL parameter to contiguous CPU float rows."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", dtype=torch.float32)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")

    if tensor.numel() == 0 or tensor.numel() % width:
        raise ValueError(f"{name} must contain a multiple of {width} values, got {tensor.shape}")
    return tensor.reshape(-1, width).contiguous()


def _match_frames(value: torch.Tensor, num_frames: int, name: str) -> torch.Tensor:
    """Broadcast one root parameter across a temporal pose chunk."""
    if len(value) == num_frames:
        return value
    if len(value) == 1:
        return value.expand(num_frames, -1).contiguous()
    raise InvalidSMPLFrameError(f"{name} has {len(value)} frames; expected 1 or {num_frames}")


class SonicSMPLConverter:
    """Use GR00T-WholeBodyControl FK to produce SONIC-local SMPL joints.

    The GR00T repository is added to ``sys.path`` only by this SONIC adapter.
    All computation runs on CPU in the publisher's consumer thread and does not
    instantiate or call a GEM/SMPL-X body layer.
    """

    PELVIS_INDEX = 0
    LEFT_WRIST_INDEX = 20
    RIGHT_WRIST_INDEX = 21

    def __init__(
        self,
        sonic_repo_path: str | Path,
        enable_yaw_calibration: bool = False,
    ) -> None:
        repo_path = Path(sonic_repo_path).expanduser().resolve()
        torch_transform_path = repo_path / "gear_sonic/trl/utils/torch_transform.py"
        human_joints_info_path = repo_path / "gear_sonic/data/human/human_joints_info.pkl"
        if not torch_transform_path.is_file():
            raise FileNotFoundError(
                "SONIC torch_transform.py was not found under "
                f"{repo_path}. Pass the GR00T-WholeBodyControl repository path."
            )
        if not human_joints_info_path.is_file():
            raise FileNotFoundError(
                f"SONIC human joint metadata was not found: {human_joints_info_path}"
            )

        repo_path_str = str(repo_path)
        if repo_path_str in sys.path:
            sys.path.remove(repo_path_str)
        sys.path.insert(0, repo_path_str)

        from gear_sonic.isaac_utils.rotations import (
            remove_smpl_base_rot,
            smpl_root_ytoz_up,
        )
        from gear_sonic.trl.utils.torch_transform import (
            angle_axis_to_quaternion,
            compute_human_joints,
            get_heading_q,
            quat_apply,
            quat_inv,
            quat_mul,
            quaternion_to_angle_axis,
        )

        imported_path = Path(sys.modules[compute_human_joints.__module__].__file__).resolve()
        if repo_path not in imported_path.parents:
            raise ImportError(
                "gear_sonic was already imported from a different location: "
                f"{imported_path}; expected it under {repo_path}"
            )

        self.sonic_repo_path = repo_path
        self.compute_human_joints_source = f"{imported_path}::compute_human_joints"
        self.enable_yaw_calibration = bool(enable_yaw_calibration)
        self.using_y_up_to_z_up = True
        self.using_remove_smpl_base_rot = True
        self._human_joints_info_path = str(human_joints_info_path)
        self._compute_human_joints = compute_human_joints
        self._angle_axis_to_quaternion = angle_axis_to_quaternion
        self._quaternion_to_angle_axis = quaternion_to_angle_axis
        self._smpl_root_ytoz_up = smpl_root_ytoz_up
        self._remove_smpl_base_rot = remove_smpl_base_rot
        self._quat_apply = quat_apply
        self._quat_inv = quat_inv
        self._quat_mul = quat_mul
        self._get_heading_q = get_heading_q
        self._initial_heading: torch.Tensor | None = None
        self._fallback_warnings_once: set[str] = set()
        self._last_fallback_warning_time: dict[str, float] = {}

    @property
    def initial_heading(self) -> torch.Tensor | None:
        if self._initial_heading is None:
            return None
        return self._initial_heading.clone()

    def reset_yaw_calibration(self) -> None:
        self._initial_heading = None

    def _warn_fallback_once(self, key: str, message: str) -> None:
        if key in self._fallback_warnings_once:
            return
        self._fallback_warnings_once.add(key)
        print(f"[SONIC WARNING] {message}")

    def _warn_fallback_rate_limited(self, key: str, message: str) -> None:
        now = time.monotonic()
        if now - self._last_fallback_warning_time.get(key, float("-inf")) < 1.0:
            return
        self._last_fallback_warning_time[key] = now
        print(f"[SONIC WARNING] {message}")

    def _select_finite_rows(
        self,
        candidates: tuple[tuple[str, Mapping[str, Any]], ...],
        field_name: str,
        width: int,
    ) -> torch.Tensor:
        """Choose the first present finite candidate, preserving source priority."""
        invalid_candidates: list[tuple[str, InvalidSMPLFrameError]] = []
        found_candidate = False

        for source_name, params in candidates:
            value = params.get(field_name)
            if value is None:
                continue
            found_candidate = True
            tensor = _as_rows(value, width, f"{source_name} {field_name}")
            try:
                _require_finite(tensor, f"{source_name} {field_name}")
            except InvalidSMPLFrameError as exc:
                invalid_candidates.append((source_name, exc))
                continue

            if invalid_candidates:
                invalid_source = invalid_candidates[0][0]
                message = f"invalid {invalid_source} {field_name}; using {source_name} {field_name}"
                if field_name == "body_pose":
                    self._warn_fallback_once(field_name, message)
                else:
                    self._warn_fallback_rate_limited(field_name, message)
            return tensor

        if not found_candidate:
            raise InvalidSMPLFrameError(
                f"{field_name} is missing from all GEM body-parameter candidates"
            )

        details = "\n".join(f"- {source}: {error}" for source, error in invalid_candidates)
        raise InvalidSMPLFrameError(f"no finite {field_name} candidate is available:\n{details}")

    def _remove_initial_yaw(self, body_quat_w: torch.Tensor) -> torch.Tensor:
        if not self.enable_yaw_calibration:
            return body_quat_w
        if self._initial_heading is None:
            initial_heading = self._get_heading_q(body_quat_w[:1])
            initial_heading = _normalize_quaternion_safe(
                initial_heading,
                "initial_heading",
            )
            self._initial_heading = initial_heading.detach().clone()
        initial_heading_inv = self._quat_inv(self._initial_heading).expand_as(body_quat_w)
        return self._quat_mul(initial_heading_inv, body_quat_w)

    @torch.inference_mode()
    def convert(
        self,
        body_params_global: Mapping[str, Any],
        body_params_incam: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Convert GEM parameters to SONIC pose and root-local FK joints.

        ``transl`` is intentionally not applied: SONIC consumes the 24 joint
        positions after removal of the root rotation, not world-space joints.
        """
        body_pose = self._select_finite_rows(
            (
                ("in-camera", body_params_incam),
                ("global", body_params_global),
            ),
            "body_pose",
            63,
        )
        _require_finite(body_pose, "body_pose")
        num_frames = len(body_pose)

        global_orient = _match_frames(
            self._select_finite_rows(
                (
                    ("global", body_params_global),
                    ("in-camera", body_params_incam),
                ),
                "global_orient",
                3,
            ),
            num_frames,
            "global_orient",
        )
        _require_finite(global_orient, "global_orient")

        global_orient_quat = self._angle_axis_to_quaternion(global_orient)
        global_orient_quat = _normalize_quaternion_safe(
            global_orient_quat,
            "global_orient_quat_y_up",
        )
        global_orient_quat = self._smpl_root_ytoz_up(global_orient_quat)
        global_orient_quat = _normalize_quaternion_safe(
            global_orient_quat,
            "global_orient_quat_z_up",
        )
        global_orient_sonic = self._quaternion_to_angle_axis(global_orient_quat)
        _require_finite(global_orient_sonic, "global_orient_sonic")

        joints = self._compute_human_joints(
            body_pose=body_pose[..., :63],
            global_orient=global_orient_sonic,
            human_joints_info_path=self._human_joints_info_path,
        )
        _require_finite(joints, "compute_human_joints output")
        joints = joints.reshape(num_frames, -1, 3)
        if joints.shape != (num_frames, 24, 3):
            raise ValueError(
                f"SONIC compute_human_joints returned {joints.shape}, expected "
                f"({num_frames}, 24, 3)"
            )

        body_quat_w = self._remove_smpl_base_rot(global_orient_quat, w_last=False)
        body_quat_w = _normalize_quaternion_safe(body_quat_w, "body_quat_w")
        root_quat_inv = self._quat_inv(body_quat_w)
        root_quat_inv = _normalize_quaternion_safe(root_quat_inv, "root_quat_inv")
        root_quat_inv_expanded = root_quat_inv.unsqueeze(1).expand(-1, 24, -1)
        smpl_joints_local = self._quat_apply(root_quat_inv_expanded, joints).contiguous()
        _require_finite(smpl_joints_local, "smpl_joints_local")
        body_quat_w = self._remove_initial_yaw(body_quat_w)
        body_quat_w = _normalize_quaternion_safe(
            body_quat_w,
            "body_quat_w after yaw calibration",
        )

        return {
            "smpl_pose": body_pose.reshape(num_frames, 21, 3),
            "smpl_joints_local": smpl_joints_local,
            "body_quat_w": body_quat_w,
        }
