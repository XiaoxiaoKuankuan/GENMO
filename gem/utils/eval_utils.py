# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema

from gem.utils.pylogger import Log


class InvalidMetricInputError(RuntimeError):
    """Raised when one evaluation sequence cannot produce trustworthy metrics."""

    def __init__(self, message: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def tensor_metric_diagnostics(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    """Return finite-value diagnostics without modifying ``tensor`` or its graph."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    value = tensor.detach()
    finite = torch.isfinite(value)
    nan_mask = torch.isnan(value)
    posinf_mask = torch.isposinf(value)
    neginf_mask = torch.isneginf(value)
    nonfinite = ~finite

    if value.numel() == 0:
        bad_frames: list[int] = []
    elif value.ndim == 0:
        bad_frames = [0] if bool(nonfinite.item()) else []
    else:
        per_frame = nonfinite.reshape(value.shape[0], -1).any(dim=1)
        bad_frames = per_frame.nonzero(as_tuple=False).flatten()[:20].cpu().tolist()

    finite_values = value[finite]
    if finite_values.numel() > 0:
        finite_min = float(finite_values.min().item())
        finite_max = float(finite_values.max().item())
        finite_absmax = float(finite_values.abs().max().item())
    else:
        finite_min = finite_max = finite_absmax = None

    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "nan_count": int(nan_mask.sum().item()),
        "posinf_count": int(posinf_mask.sum().item()),
        "neginf_count": int(neginf_mask.sum().item()),
        "nonfinite_count": int(nonfinite.sum().item()),
        "bad_frames": bad_frames,
        "finite_min": finite_min,
        "finite_max": finite_max,
        "finite_absmax": finite_absmax,
    }


def _format_tensor_diagnostic(diagnostic: dict[str, Any]) -> str:
    return (
        f"tensor={diagnostic['name']} shape={tuple(diagnostic['shape'])} "
        f"NaN={diagnostic['nan_count']} +Inf={diagnostic['posinf_count']} "
        f"-Inf={diagnostic['neginf_count']} "
        f"nonfinite={diagnostic['nonfinite_count']} "
        f"bad_frames={diagnostic['bad_frames']}"
    )


def check_finite_metric_inputs(
    tensors: dict[str, torch.Tensor], *, sequence_id: str = "unknown"
) -> None:
    """Reject a sequence when any named metric input contains NaN or Inf."""

    diagnostics = {
        name: tensor_metric_diagnostics(name, tensor) for name, tensor in tensors.items()
    }
    invalid = [item for item in diagnostics.values() if item["nonfinite_count"] > 0]
    if invalid:
        details = "; ".join(_format_tensor_diagnostic(item) for item in invalid)
        raise InvalidMetricInputError(
            f"sequence_id={sequence_id}: non-finite metric input: {details}",
            diagnostics=diagnostics,
        )


def check_non_degenerate_points(
    pred_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    sequence_id: str = "unknown",
    eps: float = 1e-12,
) -> None:
    """Reject frames whose spatial point cloud has essentially zero energy."""

    if pred_points.ndim != 3 or target_points.ndim != 3:
        raise ValueError(
            "pred_points and target_points must have shape [frames, points, coordinates]"
        )
    if pred_points.shape != target_points.shape or pred_points.shape[-1] not in (2, 3):
        raise ValueError(
            f"point-cloud shape mismatch: pred={tuple(pred_points.shape)}, "
            f"target={tuple(target_points.shape)}"
        )
    check_finite_metric_inputs(
        {"pred_points": pred_points, "target_points": target_points},
        sequence_id=sequence_id,
    )

    diagnostics: dict[str, Any] = {}
    messages = []
    for name, points in (("pred_points", pred_points), ("target_points", target_points)):
        detached = points.detach()
        centered = detached - detached.mean(dim=1, keepdim=True)
        energy = centered.square().sum(dim=(1, 2))
        invalid = (~torch.isfinite(energy)) | (energy <= eps)
        bad_frames = invalid.nonzero(as_tuple=False).flatten()[:20].cpu().tolist()
        diagnostics[f"{name}_energy"] = {
            "eps": float(eps),
            "bad_frames": bad_frames,
            "minimum": float(energy.min().item()) if energy.numel() else None,
        }
        if bad_frames:
            messages.append(f"{name} degenerate bad_frames={bad_frames} eps={eps}")
    if messages:
        raise InvalidMetricInputError(
            f"sequence_id={sequence_id}: degenerate point cloud: {'; '.join(messages)}",
            diagnostics=diagnostics,
        )


def validate_invalid_policy(invalid_policy: str) -> str:
    """Validate and return a metric callback invalid-sequence policy."""

    if invalid_policy not in {"skip", "raise"}:
        raise ValueError(
            f"invalid_policy must be one of {{'skip', 'raise'}}, got {invalid_policy!r}"
        )
    return invalid_policy


def apply_invalid_policy(invalid_policy: str, error: InvalidMetricInputError) -> None:
    """Re-raise an invalid metric input in strict mode; skip mode returns normally."""

    validate_invalid_policy(invalid_policy)
    if invalid_policy == "raise":
        raise error


def _safe_filename_component(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return text or "unknown"


def _resolve_eval_dump_run_dir(trainer: object) -> Path:
    for callback in getattr(trainer, "checkpoint_callbacks", None) or []:
        dirpath = getattr(callback, "dirpath", None)
        if dirpath:
            return Path(dirpath).expanduser().resolve(strict=False).parent
    for attribute in ("log_dir", "default_root_dir"):
        value = getattr(trainer, attribute, None)
        if value:
            return Path(value).expanduser().resolve(strict=False)
    return Path.cwd()


def _cpu_prediction_parameters(outputs: object) -> dict[str, dict[str, torch.Tensor]]:
    saved: dict[str, dict[str, torch.Tensor]] = {}
    if not isinstance(outputs, Mapping):
        return saved
    for group_name in ("pred_body_params_incam", "pred_body_params_global"):
        group = outputs.get(group_name)
        if not isinstance(group, Mapping):
            continue
        tensors = {
            str(name): value.detach().float().cpu().contiguous().clone()
            for name, value in group.items()
            if isinstance(value, torch.Tensor)
        }
        if tensors:
            saved[group_name] = tensors
    return saved


def _safe_batch_metadata(batch: object) -> dict[str, Any]:
    if not isinstance(batch, Mapping):
        return {}
    metadata: dict[str, Any] = {}
    length = batch.get("length")
    if isinstance(length, torch.Tensor) and length.numel() <= 32:
        metadata["length"] = length.detach().cpu().clone()
    gender = batch.get("gender")
    if isinstance(gender, (str, int, float, bool)):
        metadata["gender"] = gender
    elif isinstance(gender, (list, tuple)):
        metadata["gender"] = [str(value) for value in gender[:8]]
    meta = batch.get("meta")
    if isinstance(meta, (list, tuple)) and meta and isinstance(meta[0], Mapping):
        for name in ("dataset_id", "vid"):
            if name in meta[0]:
                metadata[name] = str(meta[0][name])
    return metadata


def dump_invalid_eval_sample(
    trainer: object,
    dataset_id: str,
    sequence_id: str,
    reason: str,
    outputs: object,
    *,
    batch: object | None = None,
    tensor_diagnostics: dict[str, Any] | None = None,
) -> Path | None:
    """Atomically save a compact invalid-sequence artifact without full meshes."""

    tmp_path: Path | None = None
    try:
        output_dir = _resolve_eval_dump_run_dir(trainer) / "eval_nonfinite"
        output_dir.mkdir(parents=True, exist_ok=True)
        step = int(getattr(trainer, "global_step", 0))
        epoch = int(getattr(trainer, "current_epoch", 0))
        rank = int(getattr(trainer, "global_rank", 0))
        unique = uuid.uuid4().hex[:8]
        filename = (
            f"step{step:08d}_epoch{epoch:04d}_rank{rank}_"
            f"{_safe_filename_component(dataset_id)}_"
            f"{_safe_filename_component(sequence_id)}_{unique}.pt"
        )
        final_path = output_dir / filename
        tmp_path = output_dir / f".{filename}.tmp"
        payload = {
            "dataset_id": str(dataset_id),
            "sequence_id": str(sequence_id),
            "global_step": step,
            "epoch": epoch,
            "global_rank": rank,
            "reason": str(reason),
            "tensor_diagnostics": tensor_diagnostics or {},
            "predictions": _cpu_prediction_parameters(outputs),
            "batch_metadata": _safe_batch_metadata(batch),
        }
        with tmp_path.open("wb") as file:
            torch.save(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, final_path)
        return final_path
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        Log.warning(
            f"[EvalGuard] failed to dump invalid sequence "
            f"dataset={dataset_id} vid={sequence_id}: {error}"
        )
        return None


def _detach_cpu_tensors(batch: dict[str, torch.Tensor], names: tuple[str, ...]):
    return {name: batch[name].detach().cpu() for name in names}


def _apply_frame_mask(
    tensors: dict[str, torch.Tensor],
    mask: torch.Tensor | None,
    *,
    sequence_id: str,
) -> dict[str, torch.Tensor]:
    if mask is None:
        return tensors
    if not isinstance(mask, torch.Tensor) or mask.ndim != 1:
        raise ValueError("metric mask must be a one-dimensional torch.Tensor")
    mask = mask.detach().cpu()
    if mask.dtype != torch.bool:
        raise ValueError(f"metric mask must be boolean, got {mask.dtype}")
    for name, tensor in tensors.items():
        if tensor.ndim == 0 or tensor.shape[0] != mask.shape[0]:
            raise ValueError(
                f"mask length {mask.shape[0]} does not match {name} shape {tuple(tensor.shape)}"
            )
    return {name: tensor[mask].clone() for name, tensor in tensors.items()}


def _check_minimum_frames(
    tensors: dict[str, torch.Tensor], minimum: int, *, sequence_id: str, metric_name: str
) -> None:
    frames = next(iter(tensors.values())).shape[0]
    if frames < minimum:
        raise InvalidMetricInputError(
            f"sequence_id={sequence_id}: {metric_name} requires at least {minimum} "
            f"valid frames, got {frames}",
            diagnostics={"valid_frame_count": int(frames), "required": int(minimum)},
        )


def _check_finite_metric_results(results: dict[str, Any], *, sequence_id: str) -> None:
    tensors = {f"metric.{name}": torch.as_tensor(value) for name, value in results.items()}
    check_finite_metric_inputs(tensors, sequence_id=sequence_id)


@torch.no_grad()
def compute_camcoord_metrics(batch, pelvis_idxs=None, fps=30, mask=None, sequence_id="unknown"):
    """
    Args:
        batch (dict): {
            "pred_j3d": (..., J, 3) tensor
            "target_j3d":
            "pred_verts":
            "target_verts":
        }
    Returns:
        cam_coord_metrics (dict): {
            "pa_mpjpe": (..., ) numpy array
            "mpjpe":
            "pve":
            "accel":
        }
    """
    if pelvis_idxs is None:
        pelvis_idxs = [1, 2]
    names = ("pred_j3d", "target_j3d", "pred_verts", "target_verts")
    tensors = _detach_cpu_tensors(batch, names)
    tensors = _apply_frame_mask(tensors, mask, sequence_id=sequence_id)
    _check_minimum_frames(tensors, 3, sequence_id=sequence_id, metric_name="camera metrics")
    check_finite_metric_inputs(tensors, sequence_id=sequence_id)
    pred_j3d, target_j3d, pred_verts, target_verts = (tensors[name] for name in names)
    assert "mask" not in batch

    # Align by pelvis
    pred_j3d, target_j3d, pred_verts, target_verts = batch_align_by_pelvis(
        [pred_j3d, target_j3d, pred_verts, target_verts], pelvis_idxs=pelvis_idxs
    )
    aligned = {
        "aligned_pred_j3d": pred_j3d,
        "aligned_target_j3d": target_j3d,
        "aligned_pred_verts": pred_verts,
        "aligned_target_verts": target_verts,
    }
    check_finite_metric_inputs(aligned, sequence_id=sequence_id)
    check_non_degenerate_points(pred_j3d, target_j3d, sequence_id=sequence_id)

    # Metrics
    m2mm = 1000
    S1_hat = batch_compute_similarity_transform_torch(pred_j3d, target_j3d, sequence_id=sequence_id)
    pa_mpjpe = compute_jpe(S1_hat, target_j3d) * m2mm
    mpjpe = compute_jpe(pred_j3d, target_j3d) * m2mm
    pve = compute_jpe(pred_verts, target_verts) * m2mm
    accel = compute_error_accel(joints_pred=pred_j3d, joints_gt=target_j3d, fps=fps)

    camcoord_metrics = {
        "pa_mpjpe": pa_mpjpe,
        "mpjpe": mpjpe,
        "pve": pve,
        "accel": accel,
    }
    _check_finite_metric_results(camcoord_metrics, sequence_id=sequence_id)
    return camcoord_metrics


@torch.no_grad()
def compute_music_metrics(batch, mask=None):
    """
    Args:
        batch (dict): {
            "pred_j3d": (..., J, 3) tensor
            "target_j3d":
            "music_beats": (T,) numpy array
        }
    Returns:
        music_metrics (dict): {
            "PFC":
        }
    """
    # All data is in global coordinates
    pred_j3d_glob = batch["pred_j3d_glob"].cpu().numpy()  # (..., J, 3)
    # pred_j3d_glob = batch["target_j3d_glob"].cpu().numpy()  # (..., J, 3)
    up_dir = 1  # y is up
    flat_dirs = [i for i in range(3) if i != up_dir]

    DT = 1 / 30
    assert pred_j3d_glob.ndim == 3

    root_v = (pred_j3d_glob[1:, 0, :] - pred_j3d_glob[:-1, 0, :]) / DT  # root velocity (T-1, 3)
    root_a = (root_v[1:, :] - root_v[:-1, :]) / DT  # root acceleration (T-2, 3)

    # clamp the up-direction of root acceleration
    root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)  # (T-2, 3)
    # l2 norm
    root_a = np.linalg.norm(root_a, axis=-1)  # (T-2,)
    scaling = root_a.max()
    root_a = root_a / scaling

    foot_idx = [7, 10, 8, 11]
    feet = pred_j3d_glob[:, foot_idx, :]  # (T, 4, 3)
    foot_v = np.linalg.norm(
        feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1
    )  # horizontal velocity (T-2, 4)
    foot_mins = np.zeros((len(foot_v), 2))
    foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])
    foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])
    foot_v = np.maximum(foot_mins, 0)

    foot_loss = foot_mins[:, 0] * foot_mins[:, 1] * root_a  # min leftv * min rightv * root_a (T-2,)
    pfc = foot_loss.mean() * 10000

    # compute Beat Align Score
    motion_beats = compute_motion_beats(pred_j3d_glob)[0]
    music_beats = compute_music_beats(batch["music_beats"])
    ba = 0
    for bb in music_beats:
        ba += np.exp(-np.min((motion_beats - bb) ** 2) / 2 / 9)
    bas = ba / len(music_beats)
    return {
        "PFC": pfc,
        "BAS": bas,
    }


@torch.no_grad()
def compute_global_metrics(batch, mask=None, sequence_id="unknown"):
    """Follow WHAM, the input has skipped invalid frames
    Args:
        batch (dict): {
            "pred_j3d_glob": (F, J, 3) tensor
            "target_j3d_glob":
            "pred_verts_glob":
            "target_verts_glob":
        }
    Returns:
        global_metrics (dict): {
            "wa2_mpjpe": (F, ) numpy array
            "waa_mpjpe":
            "rte":
            "jitter":
            "fs":
        }
    """
    names = (
        "pred_j3d_glob",
        "target_j3d_glob",
        "pred_verts_glob",
        "target_verts_glob",
    )
    tensors = _detach_cpu_tensors(batch, names)
    tensors = _apply_frame_mask(tensors, mask, sequence_id=sequence_id)
    _check_minimum_frames(tensors, 4, sequence_id=sequence_id, metric_name="global metrics")
    check_finite_metric_inputs(tensors, sequence_id=sequence_id)
    pred_j3d_glob, target_j3d_glob, pred_verts_glob, target_verts_glob = (
        tensors[name] for name in names
    )
    assert "mask" not in batch
    check_non_degenerate_points(pred_j3d_glob, target_j3d_glob, sequence_id=sequence_id)

    seq_length = pred_j3d_glob.shape[0]

    # Use chunk to compare
    chunk_length = 100
    wa2_mpjpe, waa_mpjpe = [], []
    for start in range(0, seq_length, chunk_length):
        end = min(seq_length, start + chunk_length)

        target_j3d = target_j3d_glob[start:end].clone().cpu()
        pred_j3d = pred_j3d_glob[start:end].clone().cpu()

        w_j3d = first_align_joints(target_j3d, pred_j3d)
        wa_j3d = global_align_joints(target_j3d, pred_j3d)

        wa2_mpjpe.append(compute_jpe(target_j3d, w_j3d))
        waa_mpjpe.append(compute_jpe(target_j3d, wa_j3d))

    # Metrics
    m2mm = 1000
    wa2_mpjpe = np.concatenate(wa2_mpjpe) * m2mm
    waa_mpjpe = np.concatenate(waa_mpjpe) * m2mm

    # Additional Metrics
    rte = compute_rte(target_j3d_glob[:, 0].cpu(), pred_j3d_glob[:, 0].cpu()) * 1e2
    jitter = compute_jitter(pred_j3d_glob, fps=30)
    foot_sliding = compute_foot_sliding(target_verts_glob, pred_verts_glob) * m2mm

    global_metrics = {
        "wa2_mpjpe": wa2_mpjpe,
        "waa_mpjpe": waa_mpjpe,
        "rte": rte,
        "jitter": jitter,
        "fs": foot_sliding,
    }
    _check_finite_metric_results(global_metrics, sequence_id=sequence_id)
    return global_metrics


@torch.no_grad()
def compute_camcoord_perjoint_metrics(batch, pelvis_idxs=None):
    """
    Args:
        batch (dict): {
            "pred_j3d": (..., J, 3) tensor
            "target_j3d":
        }
    Returns:
        cam_coord_metrics (dict): {
            "pa_mpjpe": (..., ) numpy array
            "mpjpe":
            "pve":
            "accel":
        }
    """
    if pelvis_idxs is None:
        pelvis_idxs = [1, 2]
    # All data is in camera coordinates
    pred_j3d = batch["pred_j3d"].cpu()  # (..., J, 3)
    target_j3d = batch["target_j3d"].cpu()
    pred_verts = batch["pred_verts"].cpu()
    target_verts = batch["target_verts"].cpu()

    # Align by pelvis
    pred_j3d, target_j3d, pred_verts, target_verts = batch_align_by_pelvis(
        [pred_j3d, target_j3d, pred_verts, target_verts], pelvis_idxs=pelvis_idxs
    )
    # Metrics
    m2mm = 1000
    perjoint_mpjpe = compute_perjoint_jpe(pred_j3d, target_j3d) * m2mm

    camcoord_perjoint_metrics = {
        "mpjpe": perjoint_mpjpe,
    }
    return camcoord_perjoint_metrics


# ===== Utilities =====


def compute_jpe(S1, S2):
    return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).mean(dim=-1).numpy()


def compute_perjoint_jpe(S1, S2):
    return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).numpy()


def batch_align_by_pelvis(data_list, pelvis_idxs=None):
    """
    Assumes data is given as [pred_j3d, target_j3d, pred_verts, target_verts].
    Each data is in shape of (frames, num_points, 3)
    Pelvis is notated as one / two joints indices.
    Align all data to the corresponding pelvis location.
    """
    if pelvis_idxs is None:
        pelvis_idxs = [1, 2]

    pred_j3d, target_j3d, pred_verts, target_verts = data_list

    pred_pelvis = pred_j3d[:, pelvis_idxs].mean(dim=1, keepdims=True).clone()
    target_pelvis = target_j3d[:, pelvis_idxs].mean(dim=1, keepdims=True).clone()

    # Align to the pelvis
    pred_j3d = pred_j3d - pred_pelvis
    target_j3d = target_j3d - target_pelvis
    pred_verts = pred_verts - pred_pelvis
    target_verts = target_verts - target_pelvis

    return (pred_j3d, target_j3d, pred_verts, target_verts)


def batch_compute_similarity_transform_torch(S1, S2, sequence_id="unknown", eps=1e-12):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    check_finite_metric_inputs({"S1": S1, "S2": S2}, sequence_id=sequence_id)
    if S1.ndim != 3 or S2.ndim != 3 or S1.shape != S2.shape:
        raise ValueError(
            f"S1 and S2 must have matching rank-3 shapes, got {S1.shape} and {S2.shape}"
        )
    if S1.device != S2.device:
        raise ValueError(f"S1 and S2 must share a device, got {S1.device} and {S2.device}")
    original_dtype = S1.dtype
    original_device = S1.device

    transposed = False
    if S1.shape[-1] in (2, 3):
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    elif S1.shape[1] not in (2, 3):
        raise ValueError(f"S1 and S2 must contain 2D or 3D points, got shape {S1.shape}")
    S1 = S1.to(dtype=torch.float64)
    S2 = S2.to(dtype=torch.float64)

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)
    invalid_var = (~torch.isfinite(var1)) | (var1 <= eps)
    if invalid_var.any():
        bad_frames = invalid_var.nonzero(as_tuple=False).flatten()[:20].cpu().tolist()
        raise InvalidMetricInputError(
            f"sequence_id={sequence_id}: invalid Procrustes variance; "
            f"bad_frames={bad_frames} eps={eps}",
            diagnostics={
                "var1": tensor_metric_diagnostics("var1", var1),
                "bad_frames": bad_frames,
                "eps": float(eps),
            },
        )

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))
    check_finite_metric_inputs({"Procrustes_K": K}, sequence_id=sequence_id)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    try:
        U, singular_values, Vh = torch.linalg.svd(K, full_matrices=False)
    except RuntimeError as error:
        raise InvalidMetricInputError(
            f"sequence_id={sequence_id}: finite Procrustes SVD failed: {error}",
            diagnostics={"Procrustes_K": tensor_metric_diagnostics("Procrustes_K", K)},
        ) from error
    check_finite_metric_inputs(
        {"svd_U": U, "svd_singular_values": singular_values, "svd_Vh": Vh},
        sequence_id=sequence_id,
    )
    V = Vh.transpose(1, 2)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device, dtype=S1.dtype).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))
    check_finite_metric_inputs({"Procrustes_R": R}, sequence_id=sequence_id)

    # 5. Recover scale.
    scale = torch.diagonal(R.bmm(K), dim1=-2, dim2=-1).sum(dim=-1) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t
    check_finite_metric_inputs(
        {"Procrustes_scale": scale, "Procrustes_t": t, "S1_hat_float64": S1_hat},
        sequence_id=sequence_id,
    )

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    S1_hat = S1_hat.to(device=original_device, dtype=original_dtype)
    check_finite_metric_inputs({"S1_hat": S1_hat}, sequence_id=sequence_id)
    return S1_hat


def batch_compute_scale_trans_torch(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
    assert S2.shape[1] == S1.shape[1]

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    return scale, t, R


def compute_error_accel(joints_gt, joints_pred, valid_mask=None, fps=None):
    r"""
    Use [i-1, i, i+1] to compute acc at frame_i. The acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries(-1, 0, +1) in the
    acceleration error will be zero'd out.
    Args:
        joints_gt : (F, J, 3)
        joints_pred : (F, J, 3)
        valid_mask : (F)
    Returns:
        error_accel (F-2) when valid_mask is None, else (F'), F' <= F-2
    """
    # (F, J, 3) -> (F-2) per-joint
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]
    normed = np.linalg.norm(accel_pred - accel_gt, axis=-1).mean(axis=-1)
    if fps is not None:
        normed = normed * fps**2

    if valid_mask is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(valid_mask)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)
        if new_vis.sum() == 0:
            print("Warning!!! no valid acceleration error to compute.")

    return normed[new_vis]


def compute_rte(target_trans, pred_trans):
    # Compute the global alignment
    _, rot, trans = align_pcl(target_trans[None, :], pred_trans[None, :], fixed_scale=True)
    pred_trans_hat = (torch.einsum("tij,tnj->tni", rot, pred_trans[None, :]) + trans[None, :])[0]

    # Compute the entire displacement of ground truth trajectory
    disps, disp = [], 0
    for p1, p2 in zip(target_trans, target_trans[1:]):
        delta = (p2 - p1).norm(2, dim=-1)
        disp += delta
        disps.append(disp)

    # Compute absolute root-translation-error (RTE)
    rte = torch.norm(target_trans - pred_trans_hat, 2, dim=-1)

    # Normalize it to the displacement
    return (rte / disp).numpy()


def compute_jitter(joints, fps=30):
    """compute jitter of the motion
    Args:
        joints (N, J, 3).
        fps (float).
    Returns:
        jitter (N-3).
    """
    pred_jitter = torch.norm(
        (joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3]) * (fps**3),
        dim=2,
    ).mean(dim=-1)

    return pred_jitter.cpu().numpy() / 10.0


def compute_foot_sliding(target_verts, pred_verts, thr=1e-2):
    """compute foot sliding error
    The foot ground contact label is computed by the threshold of 1 cm/frame
    Args:
        target_verts (N, 6890, 3).
        pred_verts (N, 6890, 3).
    Returns:
        error (N frames in contact).
    """
    assert target_verts.shape == pred_verts.shape
    assert target_verts.shape[-2] == 6890

    # Foot vertices idxs
    foot_idxs = [3216, 3387, 6617, 6787]

    # Compute contact label
    foot_loc = target_verts[:, foot_idxs]
    foot_disp = (foot_loc[1:] - foot_loc[:-1]).norm(2, dim=-1)
    contact = foot_disp[:] < thr

    pred_feet_loc = pred_verts[:, foot_idxs]
    pred_disp = (pred_feet_loc[1:] - pred_feet_loc[:-1]).norm(2, dim=-1)

    error = pred_disp[contact]

    return error.cpu().numpy()


def convert_joints22_to_24(joints22, ratio2220=0.3438, ratio2321=0.3345):
    joints24 = torch.zeros(*joints22.shape[:-2], 24, 3).to(joints22.device)
    joints24[..., :22, :] = joints22
    joints24[..., 22, :] = joints22[..., 20, :] + ratio2220 * (
        joints22[..., 20, :] - joints22[..., 18, :]
    )
    joints24[..., 23, :] = joints22[..., 21, :] + ratio2321 * (
        joints22[..., 21, :] - joints22[..., 19, :]
    )
    return joints24


def align_pcl(Y, X, weight=None, fixed_scale=False):
    """align similarity transform to align X with Y using umeyama method
    X' = s * R * X + t is aligned with Y
    :param Y (*, N, 3) first trajectory
    :param X (*, N, 3) second trajectory
    :param weight (*, N, 1) optional weight of valid correspondences
    :returns s (*, 1), R (*, 3, 3), t (*, 3)
    """
    *dims, N, _ = Y.shape
    N = torch.ones(*dims, 1, 1) * N

    if weight is not None:
        Y = Y * weight
        X = X * weight
        N = weight.sum(dim=-2, keepdim=True)  # (*, 1, 1)

    # subtract mean
    my = Y.sum(dim=-2) / N[..., 0]  # (*, 3)
    mx = X.sum(dim=-2) / N[..., 0]
    y0 = Y - my[..., None, :]  # (*, N, 3)
    x0 = X - mx[..., None, :]

    if weight is not None:
        y0 = y0 * weight
        x0 = x0 * weight

    # correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N  # (*, 3, 3)
    U, D, Vh = torch.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

    S = torch.eye(3).reshape(*(1,) * (len(dims)), 3, 3).repeat(*dims, 1, 1)
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S[neg, 2, 2] = -1

    R = torch.matmul(U, torch.matmul(S, Vh))  # (*, 3, 3)

    D = torch.diag_embed(D)  # (*, 3, 3)
    if fixed_scale:
        s = torch.ones(*dims, 1, device=Y.device, dtype=torch.float32)
    else:
        var = torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True) / N  # (*, 1, 1)
        s = (
            torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
            / var[..., 0]
        )  # (*, 1)

    t = my - s * torch.matmul(R, mx[..., None])[..., 0]  # (*, 3)

    return s, R, t


def global_align_joints(gt_joints, pred_joints):
    """
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    s_glob, R_glob, t_glob = align_pcl(gt_joints.reshape(-1, 3), pred_joints.reshape(-1, 3))
    pred_glob = s_glob * torch.einsum("ij,tnj->tni", R_glob, pred_joints) + t_glob[None, None]
    return pred_glob


def first_align_joints(gt_joints, pred_joints):
    """
    align the first two frames
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    # (1, 1), (1, 3, 3), (1, 3)
    s_first, R_first, t_first = align_pcl(
        gt_joints[:2].reshape(1, -1, 3), pred_joints[:2].reshape(1, -1, 3)
    )
    pred_first = s_first * torch.einsum("tij,tnj->tni", R_first, pred_joints) + t_first[:, None]
    return pred_first


def rearrange_by_mask(x, mask):
    """
    x (L, *)
    mask (M,), M >= L
    """
    M = mask.size(0)
    L = x.size(0)
    if M == L:
        return x
    assert M > L
    assert mask.sum() == L
    x_rearranged = torch.zeros((M, *x.size()[1:]), dtype=x.dtype, device=x.device)
    x_rearranged[mask] = x
    return x_rearranged


def as_np_array(d):
    if isinstance(d, torch.Tensor):
        return d.cpu().numpy()
    elif isinstance(d, np.ndarray):
        return d
    else:
        return np.array(d)


def compute_motion_beats(keypoints):
    keypoints = keypoints.reshape(-1, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = gaussian_filter(kinetic_vel, sigma=5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats


def compute_music_beats(beats):
    beats = beats.astype(bool)
    beat_axis = np.arange(len(beats))
    beat_axis = beat_axis[beats]

    return beat_axis
