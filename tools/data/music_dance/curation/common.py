"""Shared contracts for exporting and curating music-dance body motion.

The review package deliberately contains motion only.  Music remains in the
private source datasets and is re-associated through the immutable master
index after reviewers return a decision CSV.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gem.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle


SCHEMA_VERSION = 1
DEFAULT_EXPORT_ID = "music_only_4set_v1"
SPLITS = ("train", "val", "test")
DATASET_ORDER = ("aistpp", "aioz_gdance", "finedance", "compas3d")
DATASET_DIRS = {
    "aistpp": "AIST++",
    "aioz_gdance": "AIOZ-GDANCE",
    "finedance": "FineDance",
    "compas3d": "CoMPAS3D",
}
VALID_DECISIONS = {"keep", "reject", "unsure"}
ISSUE_CODES = {
    "pose_corruption",
    "root_drift_or_teleport",
    "floor_or_height_error",
    "orientation_error",
    "jitter",
    "frozen_or_duplicate_frames",
    "tracking_or_identity_switch",
    "scale_or_translation_error",
    "other",
}
DECISION_COLUMNS = (
    "export_id",
    "review_id",
    "dataset",
    "sample_id",
    "duration_sec",
    "decision",
    "issue_codes",
    "reviewer",
    "notes",
)
REVIEW_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Maps source Z-up coordinates (x, y, z) to GENMO review Y-up coordinates
# (x, z, -y).  This is the same rigid rotation used by the CoMPAS3D converter.
Z_UP_TO_Y_UP = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=torch.float32,
)


def safe_torch_load(path: str | Path) -> Any:
    """Load a trusted local conversion artifact across Torch versions."""
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_motion(pose: torch.Tensor, transl: torch.Tensor, betas: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for name, value in (("pose", pose), ("transl", transl), ("betas", betas)):
        array = np.ascontiguousarray(value.detach().cpu().numpy().astype(np.float32, copy=False))
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def git_commit(repo_root: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as file:
        temporary = Path(file.name)
        file.write(text)
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    atomic_write_text(path, text)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], columns=DECISION_COLUMNS) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent,
        prefix=f".{path.name}.", delete=False,
    ) as file:
        temporary = Path(file.name)
        writer = csv.DictWriter(file, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: CSV header is missing")
        return list(reader.fieldnames), [dict(row) for row in reader]


def atomic_torch_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_save_npz(path: str | Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def make_review_id(dataset: str, sample_id: str) -> str:
    if dataset not in DATASET_ORDER:
        raise ValueError(f"unsupported dataset: {dataset}")
    if not sample_id or not REVIEW_ID_RE.fullmatch(sample_id):
        raise ValueError(
            f"sample_id must contain only letters, digits, dot, underscore or dash: {sample_id!r}"
        )
    return f"{dataset}__{sample_id}"


def resolve_relative(root: str | Path, relative: str, field: str) -> Path:
    root = Path(root).expanduser().resolve()
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes dataset root: {relative}") from exc
    return path


def canonical_motion(payload: Any, source: str | Path) -> dict[str, torch.Tensor]:
    """Normalize an existing converted payload to the body-only review contract."""
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: motion payload must be a dict")
    if "pose" in payload:
        pose = torch.as_tensor(payload["pose"]).detach().cpu().float()
        if pose.ndim != 2 or pose.shape[1] != 66:
            raise ValueError(f"{source}: pose must have shape [T,66], got {tuple(pose.shape)}")
        global_orient = pose[:, :3]
        body_pose = pose[:, 3:66]
    else:
        try:
            global_orient = torch.as_tensor(payload["global_orient"]).detach().cpu().float()
            body_pose = torch.as_tensor(payload["body_pose"]).detach().cpu().float()
        except KeyError as exc:
            raise ValueError(f"{source}: missing canonical field {exc.args[0]}") from exc
        pose = torch.cat((global_orient, body_pose), dim=-1)
    try:
        transl = torch.as_tensor(payload["transl"]).detach().cpu().float()
    except KeyError as exc:
        raise ValueError(f"{source}: missing canonical field transl") from exc
    frames = int(pose.shape[0]) if pose.ndim >= 1 else 0
    raw_betas = payload.get("betas", torch.zeros(frames, 10))
    betas = torch.as_tensor(raw_betas).detach().cpu().float()
    if betas.ndim == 1 and betas.shape[0] == 10:
        betas = betas.unsqueeze(0).repeat(frames, 1)
    elif betas.ndim == 2 and betas.shape == (1, 10):
        betas = betas.repeat(frames, 1)
    result = {
        "pose": pose.contiguous(),
        "global_orient": global_orient.contiguous(),
        "body_pose": body_pose.contiguous(),
        "transl": transl.contiguous(),
        "betas": betas.contiguous(),
    }
    validate_canonical_motion(result, source)
    return result


def validate_canonical_motion(motion: Mapping[str, torch.Tensor], source: str | Path) -> int:
    expected = {"pose": 66, "global_orient": 3, "body_pose": 63, "transl": 3, "betas": 10}
    lengths: set[int] = set()
    for field, width in expected.items():
        value = motion[field]
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"{source}: {field} must have shape [T,{width}], got {tuple(value.shape)}")
        if value.dtype != torch.float32:
            raise ValueError(f"{source}: {field} must be float32, got {value.dtype}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{source}: {field} contains NaN or Inf")
        lengths.add(int(value.shape[0]))
    if len(lengths) != 1 or next(iter(lengths), 0) <= 0:
        raise ValueError(f"{source}: motion fields have inconsistent or empty lengths")
    if not torch.equal(motion["pose"][:, :3], motion["global_orient"]):
        raise ValueError(f"{source}: pose[:, :3] differs from global_orient")
    if not torch.equal(motion["pose"][:, 3:66], motion["body_pose"]):
        raise ValueError(f"{source}: pose[:, 3:66] differs from body_pose")
    return next(iter(lengths))


def transform_z_up_to_y_up(
    motion: Mapping[str, torch.Tensor], pelvis: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Rigidly rotate an SMPL-X motion while preserving the physical pelvis path."""
    validate_canonical_motion(motion, "Z-up source motion")
    pelvis = torch.as_tensor(pelvis, dtype=torch.float32).reshape(3)
    source_rotation = axis_angle_to_matrix(motion["global_orient"])
    target_rotation = Z_UP_TO_Y_UP @ source_rotation
    global_orient = matrix_to_axis_angle(target_rotation).float().contiguous()
    transl = (
        (Z_UP_TO_Y_UP @ (motion["transl"] + pelvis).unsqueeze(-1)).squeeze(-1) - pelvis
    ).float().contiguous()
    body_pose = motion["body_pose"].clone().contiguous()
    result = {
        "pose": torch.cat((global_orient, body_pose), dim=-1).contiguous(),
        "global_orient": global_orient,
        "body_pose": body_pose,
        "transl": transl,
        "betas": motion["betas"].clone().contiguous(),
    }
    validate_canonical_motion(result, "Y-up review motion")
    return result


def validate_review_npz(path: str | Path, master: Mapping[str, Any] | None = None) -> int:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {"pose", "transl", "betas", "fps", "num_frames", "review_id"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{path}: missing NPZ fields {sorted(missing)}")
        pose = payload["pose"]
        transl = payload["transl"]
        betas = payload["betas"]
        frames = int(np.asarray(payload["num_frames"]).item())
        fps = float(np.asarray(payload["fps"]).item())
        review_id = str(np.asarray(payload["review_id"]).item())
        if pose.shape != (frames, 66) or transl.shape != (frames, 3) or betas.shape != (frames, 10):
            raise ValueError(
                f"{path}: expected pose/transl/betas [(T,66),(T,3),(T,10)], got "
                f"{pose.shape}, {transl.shape}, {betas.shape} with T={frames}"
            )
        if any(value.dtype != np.float32 for value in (pose, transl, betas)):
            raise ValueError(f"{path}: pose/transl/betas must all be float32")
        if not all(np.isfinite(value).all() for value in (pose, transl, betas)):
            raise ValueError(f"{path}: motion contains NaN or Inf")
        if not np.isclose(fps, 30.0):
            raise ValueError(f"{path}: fps must be 30, got {fps}")
        if frames <= 0:
            raise ValueError(f"{path}: num_frames must be positive")
        if master is not None:
            if review_id != master["review_id"]:
                raise ValueError(f"{path}: review_id differs from master index")
            if frames != int(master["num_frames"]):
                raise ValueError(f"{path}: num_frames differs from master index")
    return frames


def link_file(source: str | Path, target: str | Path) -> str:
    """Materialize a file as a hardlink, or copy it across filesystems.

    The dataset loaders deliberately reject symlinks that resolve outside the
    configured root.  Hardlinks preserve that path-safety invariant and do not
    duplicate data when source and curated roots share a filesystem.
    """
    source = Path(source).resolve()
    target = Path(target)
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def link_directory(source: str | Path, target: str | Path) -> None:
    source = Path(source).resolve()
    target = Path(target)
    if not source.is_dir():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)


def resolve_decisions_path(export_root: str | Path, value: str | Path) -> Path:
    value = Path(value).expanduser()
    candidates = [value]
    if not value.is_absolute():
        candidates.extend([Path(export_root) / value, Path(export_root) / "review" / value])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"decision CSV does not exist; checked: {candidates}")
