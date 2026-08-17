# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Ground-estimate sidecar contract for SMPL music-dance training.

The sidecar deliberately contains no contact labels.  It records one robust
ground height per complete source motion so random crops can express that
height relative to their first root translation.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

GROUND_CONTRACT_VERSION = "genmo.smpl437_ground.v1"
SOLE_PROXY_VERSION = "smplx_v437_sole_v1"
LEFT_SOLE_V437_INDICES = (228, 220, 229, 406)
RIGHT_SOLE_V437_INDICES = (385, 377, 386, 393)
SOLE_V437_INDICES = LEFT_SOLE_V437_INDICES + RIGHT_SOLE_V437_INDICES
LEFT_SOLE_SMPLX_VERTEX_IDS = (5907, 5774, 5911, 8920)
RIGHT_SOLE_SMPLX_VERTEX_IDS = (8601, 8468, 8605, 8708)
SOLE_SMPLX_VERTEX_IDS = LEFT_SOLE_SMPLX_VERTEX_IDS + RIGHT_SOLE_SMPLX_VERTEX_IDS

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sole_proxy_contract() -> dict[str, Any]:
    """Return the immutable sole-proxy portion of every sidecar record."""
    return {
        "version": SOLE_PROXY_VERSION,
        "left_v437_indices": list(LEFT_SOLE_V437_INDICES),
        "right_v437_indices": list(RIGHT_SOLE_V437_INDICES),
        "left_smplx_vertex_ids": list(LEFT_SOLE_SMPLX_VERTEX_IDS),
        "right_smplx_vertex_ids": list(RIGHT_SOLE_SMPLX_VERTEX_IDS),
    }


def sha256_file(path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash a source motion without loading the complete artifact into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while True:
            chunk = file.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def estimate_ground_height(
    sole_positions: torch.Tensor,
    *,
    fps: float = 30.0,
    max_candidate_speed: float = 0.20,
    quantile: float = 0.05,
    cluster_half_width: float = 0.02,
    min_candidates: int = 30,
    min_cluster_candidates: int = 8,
    max_cluster_mad: float = 0.01,
) -> dict[str, Any]:
    """Estimate ground Y from full-sequence sole proxy positions.

    A frame/proxy is a candidate only if both adjacent transitions (where
    present) are no faster than ``max_candidate_speed``.  The low cluster is
    centred at the candidate 5th percentile.  Insufficient or dispersed low
    clusters are invalid and never fall back to Y=0.
    """
    positions = torch.as_tensor(sole_positions).detach().cpu().float()
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            "sole_positions must have shape [T,P,3], got "
            f"{tuple(positions.shape)}"
        )
    if positions.shape[1] != len(SOLE_V437_INDICES):
        raise ValueError(
            f"sole_positions must contain {len(SOLE_V437_INDICES)} proxy points"
        )
    if positions.shape[0] < 2 or not torch.isfinite(positions).all():
        return {
            "ground_y": None,
            "ground_valid": False,
            "candidate_count": 0,
            "low_cluster_count": 0,
            "invalid_reason": "too_short_or_nonfinite",
        }
    if fps <= 0 or max_candidate_speed <= 0:
        raise ValueError("fps and max_candidate_speed must be positive")

    transition_speed = torch.linalg.vector_norm(
        positions[1:] - positions[:-1], dim=-1
    ) * float(fps)
    frame_speed = torch.empty_like(positions[..., 0])
    frame_speed[0] = transition_speed[0]
    frame_speed[-1] = transition_speed[-1]
    if positions.shape[0] > 2:
        frame_speed[1:-1] = torch.maximum(
            transition_speed[:-1], transition_speed[1:]
        )
    candidates = positions[..., 1][frame_speed <= float(max_candidate_speed)]
    candidate_count = int(candidates.numel())
    if candidate_count < int(min_candidates):
        return {
            "ground_y": None,
            "ground_valid": False,
            "candidate_count": candidate_count,
            "low_cluster_count": 0,
            "invalid_reason": "insufficient_stationary_candidates",
        }

    low_quantile = torch.quantile(candidates, float(quantile))
    in_cluster = (candidates - low_quantile).abs() <= float(cluster_half_width)
    cluster = candidates[in_cluster]
    cluster_count = int(cluster.numel())
    if cluster_count < int(min_cluster_candidates):
        return {
            "ground_y": None,
            "ground_valid": False,
            "candidate_count": candidate_count,
            "low_cluster_count": cluster_count,
            "invalid_reason": "insufficient_low_cluster",
        }

    ground_y = cluster.median()
    mad = (cluster - ground_y).abs().median()
    if not torch.isfinite(ground_y) or float(mad) > float(max_cluster_mad):
        return {
            "ground_y": None,
            "ground_valid": False,
            "candidate_count": candidate_count,
            "low_cluster_count": cluster_count,
            "cluster_mad_m": float(mad),
            "invalid_reason": "unstable_low_cluster",
        }
    return {
        "ground_y": float(ground_y),
        "ground_valid": True,
        "candidate_count": candidate_count,
        "low_cluster_count": cluster_count,
        "cluster_mad_m": float(mad),
        "invalid_reason": None,
    }


def validate_ground_record(record: dict[str, Any], *, source: str = "sidecar") -> None:
    """Validate one JSON-compatible ground record against contract v1."""
    required = {
        "contract_version",
        "sample_id",
        "source_motion_sha256",
        "num_frames",
        "fps",
        "ground_y",
        "ground_valid",
        "sole_proxy",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"{source}: missing ground fields {sorted(missing)}")
    if record["contract_version"] != GROUND_CONTRACT_VERSION:
        raise ValueError(
            f"{source}: unsupported contract_version={record['contract_version']!r}"
        )
    if not str(record["sample_id"]):
        raise ValueError(f"{source}: sample_id must be non-empty")
    if not _SHA256_RE.fullmatch(str(record["source_motion_sha256"])):
        raise ValueError(f"{source}: source_motion_sha256 must be lowercase SHA256")
    if not isinstance(record["num_frames"], int) or record["num_frames"] <= 0:
        raise ValueError(f"{source}: num_frames must be a positive integer")
    if abs(float(record["fps"]) - 30.0) > 1e-6:
        raise ValueError(f"{source}: physics-v1 requires fps=30")
    if not isinstance(record["ground_valid"], bool):
        raise ValueError(f"{source}: ground_valid must be boolean")
    if record["ground_valid"]:
        ground_y = record["ground_y"]
        if ground_y is None or not torch.isfinite(torch.tensor(float(ground_y))):
            raise ValueError(f"{source}: valid ground requires finite ground_y")
    elif record["ground_y"] is not None:
        raise ValueError(f"{source}: invalid ground must use ground_y=null")
    if record["sole_proxy"] != sole_proxy_contract():
        raise ValueError(f"{source}: sole proxy contract differs from {SOLE_PROXY_VERSION}")


def load_ground_sidecar(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate a JSONL ground sidecar, indexed by sample ID."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ground sidecar does not exist: {path}")
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            validate_ground_record(record, source=f"{path}:{line_number}")
            sample_id = str(record["sample_id"])
            if sample_id in records:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id={sample_id}")
            records[sample_id] = record
    if not records:
        raise ValueError(f"ground sidecar is empty: {path}")
    return records


def make_ground_record(
    *,
    sample_id: str,
    source_motion_sha256: str,
    num_frames: int,
    fps: float,
    estimate: dict[str, Any],
) -> dict[str, Any]:
    """Create and validate one serialisable v1 record."""
    record = {
        "contract_version": GROUND_CONTRACT_VERSION,
        "sample_id": str(sample_id),
        "source_motion_sha256": str(source_motion_sha256),
        "num_frames": int(num_frames),
        "fps": float(fps),
        "ground_y": estimate.get("ground_y"),
        "ground_valid": bool(estimate.get("ground_valid", False)),
        "sole_proxy": sole_proxy_contract(),
        "estimator": {
            "max_candidate_speed_mps": 0.20,
            "height_quantile": 0.05,
            "cluster_half_width_m": 0.02,
            "candidate_count": int(estimate.get("candidate_count", 0)),
            "low_cluster_count": int(estimate.get("low_cluster_count", 0)),
            "cluster_mad_m": estimate.get("cluster_mad_m"),
            "invalid_reason": estimate.get("invalid_reason"),
        },
    }
    validate_ground_record(record, source=f"sample_id={sample_id}")
    return record


__all__ = [
    "GROUND_CONTRACT_VERSION",
    "LEFT_SOLE_SMPLX_VERTEX_IDS",
    "LEFT_SOLE_V437_INDICES",
    "RIGHT_SOLE_SMPLX_VERTEX_IDS",
    "RIGHT_SOLE_V437_INDICES",
    "SOLE_PROXY_VERSION",
    "SOLE_SMPLX_VERTEX_IDS",
    "SOLE_V437_INDICES",
    "estimate_ground_height",
    "load_ground_sidecar",
    "make_ground_record",
    "sha256_file",
    "sole_proxy_contract",
    "validate_ground_record",
]
