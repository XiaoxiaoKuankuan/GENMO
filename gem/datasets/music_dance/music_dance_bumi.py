"""Strict, versioned BUMI music-dance dataset with aligned 30 Hz crops."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from gem.robots.bumi.contacts import BUMI_CONTACT_CONTRACT_VERSION
from gem.robots.bumi.feature_codec import (
    BUMI_QPOS_ORDER,
    BUMI_QUATERNION_CONVENTION,
    make_quaternion_continuous,
    quaternion_sign_is_continuous,
)
from gem.robots.bumi.kinematics import BumiKinematics
from gem.robots.bumi.legacy_motion import enforce_root_tilt_gate, xyzw_to_matrix
from gem.utils.pylogger import Log

BUMI_MUSIC_CONTRACT_VERSION = "genmo.bumi_music.v1"
BUMI_MUSIC_FPS = 30
BUMI_MUSIC_DIM = 35
BUMI_GROUND_SEMANTICS = frozenset(
    {
        "legacy_body_origin_min_zero",
        "gmr_foot_sole_ground_zero_v1",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_torch_load(path: Path) -> Any:
    """Load trusted local conversion artifacts across supported Torch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_contract_path(root: Path, relative: Any, field: str, sample_id: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{sample_id}: {field} must be a non-empty relative path")
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValueError(f"{sample_id}: {field} must be relative, got {relative}")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{sample_id}: {field} escapes dataset root: {relative}") from exc
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"BUMI music manifest does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: manifest row must be an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"BUMI music manifest is empty: {path}")
    return rows


def duration_repeat_count(valid_frames: int, motion_frames: int) -> int:
    if valid_frames <= 0 or motion_frames <= 0:
        raise ValueError("valid_frames and motion_frames must be positive")
    return max(int(valid_frames) // int(motion_frames), 1)


def select_aligned_window(
    sequence_length: int,
    target_length: int,
    *,
    random_crop: bool,
    eval_clip_mode: str = "center",
) -> tuple[int, int]:
    if sequence_length <= 0 or target_length <= 0:
        raise ValueError("sequence_length and target_length must be positive")
    if sequence_length <= target_length:
        return 0, sequence_length
    last_start = sequence_length - target_length
    if random_crop:
        return int(np.random.randint(0, last_start + 1)), target_length
    if eval_clip_mode == "start":
        return 0, target_length
    if eval_clip_mode == "center":
        return last_start // 2, target_length
    if eval_clip_mode == "end":
        return last_start, target_length
    raise ValueError(f"Unsupported eval_clip_mode={eval_clip_mode!r}")


class BumiMusicDatasetReader:
    """Isolate the v1 disk layout and strict payload parsing from model code."""

    def __init__(
        self,
        root: str | Path,
        dataset_name: str,
        split: str,
        kinematics: BumiKinematics,
        *,
        strict_alignment: bool = True,
        strict_contract: bool = True,
        require_quality_filter: bool = True,
        quaternion_norm_tolerance: float = 1.0e-3,
        joint_limit_tolerance: float = 1.0e-3,
        validate_payloads_on_init: bool = True,
        validate_source_hashes_on_init: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.dataset_name = str(dataset_name)
        self.split = str(split)
        self.kinematics = kinematics
        self.strict_alignment = bool(strict_alignment)
        self.strict_contract = bool(strict_contract)
        self.require_quality_filter = bool(require_quality_filter)
        self.quaternion_norm_tolerance = float(quaternion_norm_tolerance)
        self.joint_limit_tolerance = float(joint_limit_tolerance)
        if not np.isfinite(self.joint_limit_tolerance) or self.joint_limit_tolerance < 0.0:
            raise ValueError("joint_limit_tolerance must be finite and non-negative")
        self.validate_source_hashes = bool(validate_source_hashes_on_init)
        self._sha256_cache: dict[Path, str] = {}
        if not self.root.is_dir():
            raise FileNotFoundError(f"BUMI dataset root does not exist: {self.root}")
        self.dataset_info_path = self.root / "meta" / "dataset_info.json"
        self.dataset_info = self._read_dataset_info()
        self.manifest_path = self.root / "manifests" / f"{self.split}.jsonl"
        self.rows = read_jsonl(self.manifest_path)
        self._validate_manifest_rows()
        if self.strict_contract and validate_payloads_on_init:
            for row in self.rows:
                self.load_aligned_sequence(row)

    def _read_dataset_info(self) -> dict[str, Any]:
        path = self.dataset_info_path
        if not path.is_file():
            raise FileNotFoundError(f"BUMI dataset_info.json does not exist: {path}")
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid BUMI dataset_info JSON {path}: {exc}") from exc
        if not isinstance(info, dict):
            raise ValueError(f"BUMI dataset_info must be an object: {path}")
        expected = {
            "contract_version": BUMI_MUSIC_CONTRACT_VERSION,
            "robot_name": "bumi",
            "qpos_dim": 28,
            "joint_dim": 21,
            "quaternion_convention": BUMI_QUATERNION_CONVENTION,
            "qpos_order": BUMI_QPOS_ORDER,
            "fps": BUMI_MUSIC_FPS,
        }
        for key, expected_value in expected.items():
            if info.get(key) != expected_value:
                raise ValueError(
                    f"BUMI dataset_info {path}: {key} must be {expected_value!r}, "
                    f"got {info.get(key)!r}"
                )
        if tuple(map(str, info.get("joint_names", ()))) != self.kinematics.joint_order:
            raise ValueError(
                f"BUMI dataset_info {path}: joint_names must exactly match kinematics joint_order"
            )
        if info.get("kinematics_sha256") != self.kinematics.kinematics_sha256:
            raise ValueError(
                f"BUMI dataset_info {path}: kinematics_sha256={info.get('kinematics_sha256')!r} "
                f"does not match loaded asset {self.kinematics.kinematics_sha256!r}"
            )
        for key in (
            "mjcf_sha256",
            "kinematics_sha256",
            "retarget_config_sha256",
            "quality_config_sha256",
        ):
            value = info.get(key)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(
                    f"BUMI dataset_info {path}: {key} must be a real SHA-256 hex digest"
                )
        if self.require_quality_filter and info.get("quality_filter_applied") is not True:
            raise ValueError(
                f"BUMI dataset_info {path}: quality_filter_applied must be true for formal training"
            )
        if info.get("source_mjcf_sha256") != info.get("mjcf_sha256"):
            raise ValueError(f"BUMI dataset_info {path}: source_mjcf_sha256 must equal mjcf_sha256")
        if info.get("ground_semantics") not in BUMI_GROUND_SEMANTICS:
            raise ValueError(
                f"BUMI dataset_info {path}: unsupported ground_semantics="
                f"{info.get('ground_semantics')!r}; expected one of "
                f"{sorted(BUMI_GROUND_SEMANTICS)}"
            )
        if not isinstance(info.get("root_z_adjusted"), bool):
            raise ValueError(f"BUMI dataset_info {path}: root_z_adjusted must be a boolean")
        declared_joint_tolerance = info.get("reader_joint_limit_tolerance_rad")
        if declared_joint_tolerance is not None and not np.isclose(
            float(declared_joint_tolerance),
            self.joint_limit_tolerance,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"BUMI dataset_info {path}: reader_joint_limit_tolerance_rad="
                f"{declared_joint_tolerance!r} does not match reader tolerance "
                f"{self.joint_limit_tolerance!r}"
            )
        return info

    def _validate_manifest_rows(self) -> None:
        required = {
            "sample_id",
            "sequence_id",
            "dataset",
            "motion_path",
            "music_feature_path",
            "fps",
            "num_frames",
            "split",
            "quality_accepted",
            "music_group_id",
            "audio_key",
            "audio_path",
            "source_motion_sha256",
            "source_music_feature_sha256",
            "source_audio_sha256",
        }
        seen: set[str] = set()
        for line_number, row in enumerate(self.rows, 1):
            missing = required - set(row)
            if missing:
                raise ValueError(
                    f"{self.manifest_path}:{line_number}: missing fields {sorted(missing)}"
                )
            sample_id = str(row["sample_id"])
            if not sample_id:
                raise ValueError(f"{self.manifest_path}:{line_number}: sample_id is empty")
            if sample_id in seen:
                raise ValueError(
                    f"{self.manifest_path}:{line_number}: duplicate sample_id={sample_id}"
                )
            seen.add(sample_id)
            if not str(row["sequence_id"]):
                raise ValueError(f"{sample_id}: sequence_id is empty")
            if row["dataset"] != self.dataset_name:
                raise ValueError(
                    f"{sample_id}: manifest dataset={row['dataset']!r} does not match "
                    f"configured dataset_name={self.dataset_name!r}"
                )
            if row["split"] != self.split:
                raise ValueError(
                    f"{sample_id}: manifest split={row['split']!r} does not match {self.split!r}"
                )
            if int(row["fps"]) != 30:
                raise ValueError(f"{sample_id}: fps must be 30, got {row['fps']!r}")
            if not isinstance(row["num_frames"], int) or int(row["num_frames"]) <= 0:
                raise ValueError(f"{sample_id}: num_frames must be a positive integer")
            if self.require_quality_filter and row["quality_accepted"] is not True:
                raise ValueError(f"{sample_id}: quality_accepted must be true")
            for field in ("music_group_id", "audio_key"):
                if not isinstance(row[field], str) or not row[field]:
                    raise ValueError(f"{sample_id}: {field} must be a non-empty string")
            for field in (
                "source_motion_sha256",
                "source_music_feature_sha256",
                "source_audio_sha256",
            ):
                if not isinstance(row[field], str) or _SHA256.fullmatch(row[field]) is None:
                    raise ValueError(f"{sample_id}: {field} must be a lowercase SHA-256")
            for field in ("motion_path", "music_feature_path", "audio_path"):
                path = resolve_contract_path(self.root, row[field], field, sample_id)
                if not path.is_file():
                    raise FileNotFoundError(f"{sample_id}: missing {field}: {path}")
            if self.validate_source_hashes:
                expected_by_field = {
                    "music_feature_path": "source_music_feature_sha256",
                    "audio_path": "source_audio_sha256",
                }
                for path_field, sha_field in expected_by_field.items():
                    source = resolve_contract_path(
                        self.root, row[path_field], path_field, sample_id
                    )
                    if self._digest(source) != row[sha_field]:
                        raise ValueError(f"{sample_id}: {sha_field} does not match {source}")

    def _digest(self, path: Path) -> str:
        if path not in self._sha256_cache:
            self._sha256_cache[path] = sha256_file(path)
        return self._sha256_cache[path]

    def read_motion(self, row: dict[str, Any]) -> tuple[dict[str, Any], Path]:
        sample_id = str(row["sample_id"])
        path = resolve_contract_path(self.root, row["motion_path"], "motion_path", sample_id)
        payload = safe_torch_load(path)
        if not isinstance(payload, dict):
            raise ValueError(f"{sample_id}: {path}: motion payload must be a dictionary")
        expected_metadata = {
            "fps": 30,
            "robot_name": "bumi",
            "quaternion_convention": BUMI_QUATERNION_CONVENTION,
            "qpos_order": BUMI_QPOS_ORDER,
        }
        for key, expected_value in expected_metadata.items():
            if payload.get(key) != expected_value:
                raise ValueError(
                    f"{sample_id}: {path}: {key} must be {expected_value!r}, "
                    f"got {payload.get(key)!r}"
                )
        if tuple(map(str, payload.get("joint_names", ()))) != self.kinematics.joint_order:
            raise ValueError(f"{sample_id}: {path}: joint_names do not match kinematics")
        if self.require_quality_filter and payload.get("quality_accepted", True) is not True:
            raise ValueError(f"{sample_id}: {path}: quality_accepted must not be false")
        for key, expected_value in (
            ("source_motion_sha256", row["source_motion_sha256"]),
            ("source_mjcf_sha256", self.dataset_info["source_mjcf_sha256"]),
            ("quality_config_sha256", self.dataset_info["quality_config_sha256"]),
            ("retarget_config_sha256", self.dataset_info["retarget_config_sha256"]),
            ("ground_semantics", self.dataset_info["ground_semantics"]),
            ("root_z_adjusted", self.dataset_info["root_z_adjusted"]),
        ):
            if payload.get(key) != expected_value:
                raise ValueError(
                    f"{sample_id}: {path}: {key} must be {expected_value!r}, "
                    f"got {payload.get(key)!r}"
                )
        if "qpos" not in payload:
            raise ValueError(f"{sample_id}: {path}: missing qpos")
        qpos = torch.as_tensor(payload["qpos"]).detach().cpu().float()
        if qpos.ndim != 2 or qpos.shape[1] != 28 or qpos.shape[0] <= 0:
            raise ValueError(
                f"{sample_id}: {path}: qpos must have shape [T,28], got {tuple(qpos.shape)}"
            )
        if not bool(torch.isfinite(qpos).all()):
            raise ValueError(f"{sample_id}: {path}: qpos contains NaN or Inf")
        quat = qpos[:, 3:7]
        quat_norm = torch.linalg.vector_norm(quat, dim=-1)
        max_norm_error = float((quat_norm - 1.0).abs().max())
        if max_norm_error > self.quaternion_norm_tolerance or bool((quat_norm < 1.0e-8).any()):
            raise ValueError(
                f"{sample_id}: {path}: root quaternion norm error {max_norm_error:.6g} "
                f"exceeds tolerance {self.quaternion_norm_tolerance}"
            )
        was_continuous = quaternion_sign_is_continuous(quat)
        qpos[:, 3:7] = make_quaternion_continuous(quat)
        quat_xyzw = qpos[:, [4, 5, 6, 3]].numpy()
        local_z_world = xyzw_to_matrix(quat_xyzw)[..., :, 2]
        enforce_root_tilt_gate(
            np.rad2deg(np.arccos(np.clip(local_z_world[..., 2], -1.0, 1.0))),
            context=f"{sample_id}: {path}",
        )
        joints = qpos[:, 7:]
        lower = self.kinematics.joint_lower_limits.cpu() - self.joint_limit_tolerance
        upper = self.kinematics.joint_upper_limits.cpu() + self.joint_limit_tolerance
        violation = (joints < lower) | (joints > upper)
        if bool(violation.any()):
            frame, joint = violation.nonzero(as_tuple=False)[0].tolist()
            raise ValueError(
                f"{sample_id}: {path}: qpos joint limit violation at frame={frame}, "
                f"joint={self.kinematics.joint_order[joint]!r}, value={float(joints[frame, joint]):.7g}, "
                f"allowed=[{float(lower[joint]):.7g}, {float(upper[joint]):.7g}]"
            )
        result: dict[str, Any] = {
            "qpos": qpos.contiguous(),
            "quaternion_sign_corrected": not was_continuous,
        }
        if "foot_contact" in payload:
            contact = torch.as_tensor(payload["foot_contact"]).detach().cpu().float()
            if tuple(contact.shape) != (qpos.shape[0], 2):
                raise ValueError(
                    f"{sample_id}: {path}: foot_contact must have shape "
                    f"[{qpos.shape[0]},2], got {tuple(contact.shape)}"
                )
            if not bool(torch.isfinite(contact).all()) or bool(
                ((contact < 0.0) | (contact > 1.0)).any()
            ):
                raise ValueError(f"{sample_id}: {path}: foot_contact must be finite in [0,1]")
            if payload.get("foot_contact_contract_version") == BUMI_CONTACT_CONTRACT_VERSION:
                result["foot_contact"] = contact.contiguous()
                result["foot_contact_available"] = torch.ones(qpos.shape[0], dtype=torch.bool)
                result["foot_contact_contract_version"] = BUMI_CONTACT_CONTRACT_VERSION
                result["foot_contact_source"] = "versioned_motion_payload"
        for key in ("source_dataset", "source_sample_id", "retarget_quality"):
            if key in payload:
                result[key] = payload[key]
        return result, path

    def read_music(self, row: dict[str, Any]) -> tuple[torch.Tensor, Path]:
        sample_id = str(row["sample_id"])
        path = resolve_contract_path(
            self.root, row["music_feature_path"], "music_feature_path", sample_id
        )
        payload = safe_torch_load(path)
        if not isinstance(payload, torch.Tensor):
            raise ValueError(
                f"{sample_id}: {path}: genmo.bumi_music.v1 music files must contain a raw "
                "EDGE35 Tensor[T,35]"
            )
        music = payload.detach().cpu().float()
        if music.ndim != 2 or music.shape[1] != 35 or music.shape[0] <= 0:
            raise ValueError(
                f"{sample_id}: {path}: music must have shape [T,35], got {tuple(music.shape)}"
            )
        if not bool(torch.isfinite(music).all()):
            raise ValueError(f"{sample_id}: {path}: music contains NaN or Inf")
        if self.validate_source_hashes and self._digest(path) != row["source_music_feature_sha256"]:
            raise ValueError(f"{sample_id}: source_music_feature_sha256 mismatch: {path}")
        return music.contiguous(), path

    def load_aligned_sequence(self, row: dict[str, Any]) -> dict[str, Any]:
        motion, motion_path = self.read_motion(row)
        music, music_path = self.read_music(row)
        lengths = {
            "qpos": int(motion["qpos"].shape[0]),
            "music": int(music.shape[0]),
            "manifest": int(row["num_frames"]),
        }
        if self.strict_alignment and len(set(lengths.values())) != 1:
            raise ValueError(
                f"{row['sample_id']}: strict qpos/music/manifest alignment failed: {lengths}; "
                f"motion={motion_path}, music={music_path}"
            )
        valid_frames = min(lengths.values())
        if valid_frames <= 0:
            raise ValueError(f"{row['sample_id']}: aligned sequence is empty")
        result = {
            "row": row,
            "qpos": motion["qpos"][:valid_frames],
            "music": music[:valid_frames],
            "motion_path": motion_path,
            "music_path": music_path,
            "source_lengths": lengths,
            "quaternion_sign_corrected": motion["quaternion_sign_corrected"],
        }
        if "foot_contact" in motion:
            result["foot_contact"] = motion["foot_contact"][:valid_frames]
            result["foot_contact_available"] = motion["foot_contact_available"][:valid_frames]
            result["foot_contact_contract_version"] = motion["foot_contact_contract_version"]
            result["foot_contact_source"] = motion["foot_contact_source"]
        return result


class BumiMusicDanceDataset(Dataset):
    """Return BUMI qpos28 and aligned EDGE35 music-only training windows."""

    def __init__(
        self,
        root: str | Path,
        dataset_name: str,
        kinematics_path: str | Path,
        split: str = "train",
        motion_frames: int = 120,
        duration_aware_sampling: bool = True,
        strict_alignment: bool = True,
        strict_contract: bool = True,
        require_quality_filter: bool = True,
        validate_payloads_on_init: bool = True,
        validate_source_hashes_on_init: bool = False,
        random_crop: bool | None = None,
        eval_clip_mode: str = "center",
        quaternion_norm_tolerance: float = 1.0e-3,
        joint_limit_tolerance: float = 1.0e-3,
        limit_size: int | None = None,
    ) -> None:
        super().__init__()
        self.motion_frames = int(motion_frames)
        if self.motion_frames <= 0:
            raise ValueError("motion_frames must be positive")
        self.dataset_name = str(dataset_name)
        self.split = str(split)
        self.duration_aware_sampling = bool(duration_aware_sampling)
        self.random_crop = self.split == "train" if random_crop is None else bool(random_crop)
        self.eval_clip_mode = str(eval_clip_mode)
        self.limit_size = limit_size
        self.kinematics = BumiKinematics(kinematics_path)
        self.reader = BumiMusicDatasetReader(
            root,
            dataset_name,
            split,
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
        Log.info(
            f"[{self.dataset_name}] BUMI split={self.split}, raw={self.raw_sequence_count}, "
            f"hours={self.sampling_summary['hours']:.3f}, effective_len={len(self.idx2meta)}"
        )

    def __len__(self) -> int:
        length = len(self.idx2meta)
        return min(length, int(self.limit_size)) if self.limit_size is not None else length

    @staticmethod
    def _pad_last(value: torch.Tensor, target_length: int) -> torch.Tensor:
        if value.shape[0] > target_length:
            raise ValueError(f"cannot pad length {value.shape[0]} to shorter {target_length}")
        if value.shape[0] == target_length:
            return value.contiguous()
        padding = value[-1:].expand(target_length - value.shape[0], *value.shape[1:])
        return torch.cat((value, padding), dim=0).contiguous()

    @staticmethod
    def _pad_zero(value: torch.Tensor, target_length: int) -> torch.Tensor:
        if value.shape[0] > target_length:
            raise ValueError(f"cannot pad length {value.shape[0]} to shorter {target_length}")
        if value.shape[0] == target_length:
            return value.contiguous()
        padding = torch.zeros((target_length - value.shape[0], *value.shape[1:]), dtype=value.dtype)
        return torch.cat((value, padding), dim=0).contiguous()

    def get_window(self, row_index: int, start_frame: int | None = None) -> dict[str, Any]:
        """Load one row with either legacy random crop or a sampler-selected crop."""

        if not 0 <= int(row_index) < len(self.rows):
            raise IndexError(row_index)
        row = self.rows[int(row_index)]
        sequence = self.reader.load_aligned_sequence(row)
        sequence_length = int(sequence["qpos"].shape[0])
        if start_frame is None:
            start, crop_length = select_aligned_window(
                sequence_length,
                self.motion_frames,
                random_crop=self.random_crop,
                eval_clip_mode=self.eval_clip_mode,
            )
        else:
            start = int(start_frame)
            last_start = max(sequence_length - self.motion_frames, 0)
            if not 0 <= start <= last_start:
                raise ValueError(
                    f"{row['sample_id']}: explicit crop start={start} outside [0,{last_start}]"
                )
            crop_length = min(self.motion_frames, sequence_length - start)
        end = start + crop_length
        qpos = sequence["qpos"][start:end]
        music = sequence["music"][start:end]
        valid = torch.zeros(self.motion_frames, dtype=torch.bool)
        valid[:crop_length] = True
        qpos_padded = self._pad_last(qpos, self.motion_frames)
        music_padded = self._pad_zero(music, self.motion_frames)
        result: dict[str, Any] = {
            "qpos": qpos_padded,
            "music_embed": music_padded,
            "music_beats": music_padded[:, 34].clone(),
            "length": crop_length,
            "fps": 30,
            "mask": {
                "valid": valid,
                "has_music_mask": valid.clone(),
            },
            "meta": {
                "contract_version": BUMI_MUSIC_CONTRACT_VERSION,
                "data_name": self.dataset_name,
                "dataset_id": self.dataset_name,
                "sample_id": str(row["sample_id"]),
                "sequence_id": str(row["sequence_id"]),
                "music_group_id": str(row["music_group_id"]),
                "audio_key": str(row["audio_key"]),
                "audio_path": str(
                    resolve_contract_path(
                        self.root, row["audio_path"], "audio_path", str(row["sample_id"])
                    )
                ),
                "split": self.split,
                "source_manifest": str(self.reader.manifest_path),
                "motion_path": str(sequence["motion_path"]),
                "music_feature_path": str(sequence["music_path"]),
                "start_end": (start, end),
                "source_lengths": sequence["source_lengths"],
                "quaternion_sign_corrected": sequence["quaternion_sign_corrected"],
                "robot_name": "bumi",
                "joint_names": list(self.kinematics.joint_order),
                "quaternion_convention": BUMI_QUATERNION_CONVENTION,
                "qpos_order": BUMI_QPOS_ORDER,
                "ground_semantics": self.reader.dataset_info["ground_semantics"],
                "foot_contact_contract_version": sequence.get(
                    "foot_contact_contract_version", BUMI_CONTACT_CONTRACT_VERSION
                ),
                "foot_contact_source": sequence.get(
                    "foot_contact_source", "derived_from_crop_qpos_fk"
                ),
            },
        }
        if "foot_contact" in sequence:
            result["foot_contact"] = self._pad_last(
                sequence["foot_contact"][start:end], self.motion_frames
            )
            available = sequence["foot_contact_available"][start:end]
            result["foot_contact_available"] = self._pad_zero(
                available,
                self.motion_frames,
            ).bool()
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get_window(self.idx2meta[index])


__all__ = [
    "BUMI_MUSIC_CONTRACT_VERSION",
    "BUMI_MUSIC_DIM",
    "BUMI_MUSIC_FPS",
    "BumiMusicDanceDataset",
    "BumiMusicDatasetReader",
    "duration_repeat_count",
    "read_jsonl",
    "resolve_contract_path",
    "safe_torch_load",
    "sha256_file",
    "select_aligned_window",
]
