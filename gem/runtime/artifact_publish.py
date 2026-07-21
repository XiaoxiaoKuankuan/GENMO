# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Crash-safe publication helpers for generated motion artifacts.

Generators write into a hidden directory on the same filesystem.  The
directory is flushed and atomically renamed before the ``READY`` marker is
created, so runtime watchers never consume a partially written generation.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 ``Z`` form."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_generation_prefix(value: str, *, fallback: str = "motion", limit: int = 80) -> str:
    """Convert a user-controlled label into a bounded directory-name prefix."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip()).strip("_-").lower()
    return (cleaned[:limit].rstrip("_-") or fallback) if limit > 0 else fallback


def make_unique_output_paths(output_root: Path, prefix: str) -> tuple[Path, Path]:
    """Return same-filesystem temporary and unique final generation paths."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    clean_prefix = safe_generation_prefix(prefix)
    return root / f".tmp_{token}", root / f"{clean_prefix}_{timestamp}_{token[:8]}"


def fsync_file(path: Path) -> None:
    """Flush one closed regular file to stable storage."""
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    """Flush directory metadata to stable storage."""
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_ready_directory(temporary: Path, final: Path, completed_at: str) -> None:
    """Atomically publish ``temporary`` and create ``READY`` strictly last."""
    temporary = Path(temporary)
    final = Path(final)
    if not temporary.is_dir():
        raise FileNotFoundError(f"Temporary generation directory does not exist: {temporary}")
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {final}")
    if (temporary / "READY").exists():
        raise RuntimeError("READY must not exist in the temporary generation directory")

    for path in temporary.rglob("*"):
        if path.is_file():
            fsync_file(path)
    fsync_directory(temporary)
    os.replace(temporary, final)
    fsync_directory(final.parent)

    ready = final / "READY"
    with ready.open("x", encoding="utf-8") as handle:
        handle.write(f"completed_at={completed_at}\n")
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(final)
    fsync_directory(final.parent)


def enforce_zero_shape(
    body_params: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Replace global and in-camera SMPL-X betas with exact zero tensors."""
    for group_name in ("body_params_global", "body_params_incam"):
        group = body_params.get(group_name)
        if not isinstance(group, dict) or "betas" not in group:
            raise RuntimeError(f"Cannot apply shape_mode=zero: missing {group_name}.betas")
        betas = group["betas"]
        if not isinstance(betas, torch.Tensor) or betas.ndim != 2 or betas.shape[-1] != 10:
            raise RuntimeError(
                f"Cannot apply shape_mode=zero to {group_name}.betas with shape "
                f"{getattr(betas, 'shape', None)}"
            )
        group["betas"] = torch.zeros_like(betas)
        if torch.count_nonzero(group["betas"]).item() != 0:
            raise AssertionError(f"{group_name}.betas is not zero after shape policy")
    return body_params
