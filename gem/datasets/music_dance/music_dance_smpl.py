# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Manifest-backed canonical SMPL music-dance training dataset.

The AIOZ-GDANCE, FineDance and CoMPAS3D converters all export the same
training contract: 30 Hz SMPL body motion and one aligned EDGE baseline35
feature tensor.  This loader turns that data into GEM's existing batch
contract without making image, 2D, camera, audio or text a valid condition.

All converted files are already in GENMO's Y-up metric world coordinates.
The static camera below is only a self-consistent geometry construction for
the unchanged 151D encoder and auxiliary losses; it never enters the
music-only denoiser because ``pipeline.args.in_attr == ["encoded_music"]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from gem.datasets.aistpp.aistplusplus import (
    load_music_feature_tensor,
    validate_musicfeat_v2,
)
from gem.utils.cam_utils import create_camera_sensor
from gem.utils.geo_transform import compute_cam_angvel, compute_cam_tvel, normalize_T_w2c
from gem.utils.motion_utils import get_c_rootparam, get_R_c2gv
from gem.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict
from gem.utils.pylogger import Log
from gem.utils.smplx_utils import make_smplx


def _safe_torch_load(path: Path) -> Any:
    """Load a trusted local conversion artifact across Torch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_relative(root: Path, relative: Any, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes dataset root: {relative}") from exc
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"music-dance manifest does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: manifest row must be an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"music-dance manifest is empty: {path}")
    return rows


def duration_repeat_count(valid_frames: int, motion_frames: int) -> int:
    """Natural duration-proportional repetitions used by one training epoch."""
    if valid_frames <= 0 or motion_frames <= 0:
        raise ValueError("valid_frames and motion_frames must be positive")
    return max(int(valid_frames) // int(motion_frames), 1)


def select_training_window(sequence_length: int, target_length: int) -> tuple[int, int]:
    """Select a random aligned crop, including the final legal start index."""
    if sequence_length <= 0 or target_length <= 0:
        raise ValueError("sequence_length and target_length must be positive")
    if sequence_length <= target_length:
        return 0, sequence_length
    start = int(np.random.randint(0, sequence_length - target_length + 1))
    return start, target_length


class MusicDanceSmplDataset(Dataset):
    """Read one canonical music-dance manifest as music-only GEM samples."""

    def __init__(
        self,
        root: str | Path,
        dataset_name: str,
        split: str = "train",
        motion_frames: int = 120,
        duration_aware_sampling: bool = False,
        strict_alignment: bool = True,
        enable_contact_supervision: bool = True,
        camera_distance: float = 8.0,
        limit_size: int | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.dataset_name = str(dataset_name)
        self.split = str(split)
        self.motion_frames = int(motion_frames)
        self.duration_aware_sampling = bool(duration_aware_sampling)
        self.strict_alignment = bool(strict_alignment)
        self.enable_contact_supervision = bool(enable_contact_supervision)
        self.camera_distance = float(camera_distance)
        self.limit_size = limit_size
        if self.motion_frames <= 0:
            raise ValueError("motion_frames must be positive")
        if not np.isfinite(self.camera_distance) or self.camera_distance <= 0:
            raise ValueError("camera_distance must be finite and positive")

        self.rows = _read_jsonl(self.root / "manifests" / f"{self.split}.jsonl")
        self._validate_manifest_rows()
        self.idx2meta: list[int] = []
        for row_index, row in enumerate(self.rows):
            repeats = (
                duration_repeat_count(int(row["num_frames"]), self.motion_frames)
                if self.duration_aware_sampling
                else 1
            )
            self.idx2meta.extend([row_index] * repeats)
        self.raw_sequence_count = len(self.rows)
        self.total_valid_frames = sum(int(row["num_frames"]) for row in self.rows)
        self.sampling_summary = {
            "dataset_name": self.dataset_name,
            "raw_sequences": self.raw_sequence_count,
            "valid_frames": self.total_valid_frames,
            "hours": self.total_valid_frames / 30.0 / 3600.0,
            "effective_len": len(self.idx2meta),
            "duration_aware_sampling": self.duration_aware_sampling,
        }
        self._smpl_model = None
        Log.info(
            f"[{self.dataset_name}] split={self.split}, raw={self.raw_sequence_count}, "
            f"hours={self.sampling_summary['hours']:.3f}, effective_len={len(self.idx2meta)}"
        )

    def _validate_manifest_rows(self) -> None:
        seen: set[str] = set()
        required = {
            "sample_id",
            "motion_path",
            "music_feature_path",
            "fps",
            "num_frames",
            "split",
        }
        for row_number, row in enumerate(self.rows, 1):
            missing = required - set(row)
            if missing:
                raise ValueError(
                    f"{self.dataset_name} manifest row {row_number} missing {sorted(missing)}"
                )
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id in manifest: {sample_id}")
            seen.add(sample_id)
            if row["split"] != self.split:
                raise ValueError(
                    f"{sample_id}: split={row['split']!r} differs from requested {self.split!r}"
                )
            if not np.isclose(float(row["fps"]), 30.0):
                raise ValueError(f"{sample_id}: canonical fps must be 30, got {row['fps']}")
            if not isinstance(row["num_frames"], int) or row["num_frames"] <= 0:
                raise ValueError(f"{sample_id}: num_frames must be a positive integer")
            for field in ("motion_path", "music_feature_path"):
                path = _resolve_relative(self.root, row[field], field)
                if not path.is_file():
                    raise FileNotFoundError(f"{sample_id}: missing {field}: {path}")

    @property
    def smpl_model(self):
        # Lazy construction avoids serialising an initialized SMPL module into
        # every DataLoader worker and keeps config-compose/data-stat audits cheap.
        if self._smpl_model is None:
            self._smpl_model = make_smplx("supermotion")
        return self._smpl_model

    def __len__(self) -> int:
        length = len(self.idx2meta)
        return min(length, self.limit_size) if self.limit_size is not None else length

    @staticmethod
    def _canonical_motion(payload: Any, source: Path) -> dict[str, torch.Tensor]:
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: motion payload must be a dict")
        try:
            global_orient = torch.as_tensor(payload["global_orient"]).detach().cpu().float()
            body_pose = torch.as_tensor(payload["body_pose"]).detach().cpu().float()
            transl = torch.as_tensor(payload["transl"]).detach().cpu().float()
        except KeyError as exc:
            raise ValueError(f"{source}: missing canonical motion field {exc.args[0]}") from exc
        if "betas" in payload:
            betas = torch.as_tensor(payload["betas"]).detach().cpu().float()
        else:
            betas = torch.zeros(len(global_orient), 10, dtype=torch.float32)
        expected = {
            "global_orient": (global_orient, 3),
            "body_pose": (body_pose, 63),
            "transl": (transl, 3),
            "betas": (betas, 10),
        }
        lengths: set[int] = set()
        result: dict[str, torch.Tensor] = {}
        for key, (value, width) in expected.items():
            if value.ndim != 2 or value.shape[1] != width:
                raise ValueError(f"{source}: {key} must have shape [T,{width}], got {tuple(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{source}: {key} contains NaN or Inf")
            lengths.add(int(value.shape[0]))
            result[key] = value.contiguous()
        if len(lengths) != 1 or next(iter(lengths)) <= 0:
            raise ValueError(f"{source}: canonical motion fields have inconsistent lengths")
        return result

    def _load_aligned_crop(self, idx: int) -> dict[str, Any]:
        row = self.rows[self.idx2meta[idx]]
        motion_path = _resolve_relative(self.root, row["motion_path"], "motion_path")
        music_path = _resolve_relative(self.root, row["music_feature_path"], "music_feature_path")
        motion = self._canonical_motion(_safe_torch_load(motion_path), motion_path)
        music = load_music_feature_tensor(music_path)
        validate_musicfeat_v2(music, source=music_path)
        motion_length = int(motion["body_pose"].shape[0])
        music_length = int(music.shape[0])
        manifest_length = int(row["num_frames"])
        if self.strict_alignment and not (
            motion_length == music_length == manifest_length
        ):
            raise ValueError(
                f"{row['sample_id']}: strict alignment requires motion_T == music_T == "
                f"manifest_T, got {motion_length}, {music_length}, {manifest_length}"
            )
        valid_length = min(motion_length, music_length, manifest_length)
        start, length = select_training_window(valid_length, self.motion_frames)
        end = start + length
        return {
            "row": row,
            "motion_path": motion_path,
            "music_path": music_path,
            "motion": {key: value[start:end] for key, value in motion.items()},
            "music": music[start:end].contiguous(),
            "start": start,
            "end": end,
            "length": length,
            "source_lengths": {
                "motion": motion_length,
                "music": music_length,
                "manifest": manifest_length,
            },
        }

    def _make_static_camera(
        self, smpl_params_w: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Create one fixed OpenCV camera centred on the crop's first root."""
        length = int(smpl_params_w["body_pose"].shape[0])
        # World is Y-up. OpenCV camera is X-right/Y-down/Z-forward. A 180
        # degree X rotation looks along world -Z while preserving handedness.
        R_w2c = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
        root0 = smpl_params_w["transl"][0]
        desired_root_c = torch.tensor([0.0, 0.0, self.camera_distance])
        t_w2c = desired_root_c - R_w2c @ root0
        T_w2c = torch.eye(4, dtype=torch.float32)
        T_w2c[:3, :3] = R_w2c
        T_w2c[:3, 3] = t_w2c
        T_w2c = T_w2c.repeat(length, 1, 1)
        width, height, K = create_camera_sensor(1000, 1000, 43.3)
        return T_w2c, K.repeat(length, 1, 1), R_w2c, width, height

    def __getitem__(self, idx: int) -> dict[str, Any]:
        loaded = self._load_aligned_crop(idx)
        row = loaded["row"]
        source_crop_length = int(loaded["length"])
        max_len = self.motion_frames
        smpl_params_w = loaded["motion"]
        T_w2c, K_fullimg, _, width, height = self._make_static_camera(smpl_params_w)
        offset = self.smpl_model.get_skeleton(smpl_params_w["betas"][0])[0]
        global_orient_c, transl_c = get_c_rootparam(
            smpl_params_w["global_orient"],
            smpl_params_w["transl"],
            T_w2c,
            offset,
        )
        smpl_params_c = {
            "body_pose": smpl_params_w["body_pose"].clone(),
            "betas": smpl_params_w["betas"].clone(),
            "global_orient": global_orient_c,
            "transl": transl_c,
        }
        gravity_vec = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)
        R_c2gv = get_R_c2gv(T_w2c[:, :3, :3], gravity_vec)
        normed_T_w2c = normalize_T_w2c(T_w2c)
        cam_angvel = compute_cam_angvel(normed_T_w2c[:, :3, :3])
        cam_tvel = compute_cam_tvel(normed_T_w2c[:, :3, 3])
        music = loaded["music"]

        result = {
            "meta": {
                "data_name": self.dataset_name,
                "dataset_id": self.dataset_name,
                "idx": idx,
                "sample_id": row["sample_id"],
                "source_manifest": str(
                    (self.root / "manifests" / f"{self.split}.jsonl").resolve()
                ),
                "motion_path": str(loaded["motion_path"]),
                "music_feature_path": str(loaded["music_path"]),
                "start_end": (loaded["start"], loaded["end"]),
                "source_crop_length": source_crop_length,
                "source_lengths": loaded["source_lengths"],
                "coordinate_system": "GENMO Y-up metric (already canonical; no runtime axis conversion)",
                "synthetic_camera": True,
                "height": height,
                "width": width,
            },
            # Short canonical clips are synchronously last-frame padded below.
            # Their one natural repetition remains a full 120-frame training
            # item, matching the fixed-window specialist contract.
            "length": max_len,
            "smpl_params_c": smpl_params_c,
            "smpl_params_w": smpl_params_w,
            "R_c2gv": R_c2gv,
            "gravity_vec": gravity_vec,
            "bbx_xys": torch.zeros(source_crop_length, 3, dtype=torch.float32),
            "K_fullimg": K_fullimg,
            "f_imgseq": torch.zeros(source_crop_length, 1024, dtype=torch.float32),
            "kp2d": torch.zeros(source_crop_length, 17, 3, dtype=torch.float32),
            "cam_angvel": cam_angvel,
            "cam_tvel": cam_tvel,
            "noisy_cam_tvel": cam_tvel.clone(),
            "T_w2c": normed_T_w2c,
            "music_embed": music,
            "music_beats": music[:, 34].clone(),
            "music_array": torch.zeros(source_crop_length, 1024, dtype=torch.float32),
            "music_fps": 30,
            "mask": {
                "valid": get_valid_mask(max_len, max_len),
                "humanoid": get_valid_mask(max_len, 0),
                "has_img_mask": get_valid_mask(max_len, 0),
                "has_2d_mask": get_valid_mask(max_len, 0),
                "has_cam_mask": get_valid_mask(max_len, 0),
                "has_audio_mask": get_valid_mask(max_len, 0),
                "has_music_mask": get_valid_mask(max_len, max_len),
                "2d_only": False,
                "vitpose": False,
                "bbx_xys": False,
                "f_imgseq": False,
                "spv_incam_only": False,
                "invalid_contact": not self.enable_contact_supervision,
            },
        }
        for key in ("smpl_params_c", "smpl_params_w"):
            result[key] = repeat_to_max_len_dict(result[key], max_len)
        for key in (
            "R_c2gv",
            "bbx_xys",
            "K_fullimg",
            "f_imgseq",
            "kp2d",
            "cam_angvel",
            "cam_tvel",
            "noisy_cam_tvel",
            "T_w2c",
            "music_embed",
            "music_beats",
            "music_array",
        ):
            result[key] = repeat_to_max_len(result[key], max_len)
        return result


__all__ = [
    "MusicDanceSmplDataset",
    "duration_repeat_count",
    "select_training_window",
]
