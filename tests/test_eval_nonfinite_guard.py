"""CPU tests for evaluation NaN/Inf, degeneration, SVD and dump guards."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gem.utils.eval_utils import (
    InvalidMetricInputError,
    apply_invalid_policy,
    batch_compute_similarity_transform_torch,
    check_finite_metric_inputs,
    compute_camcoord_metrics,
    dump_invalid_eval_sample,
    validate_invalid_policy,
)


def _metric_batch(frames: int = 7) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260804)
    target_j3d = torch.randn(frames, 14, 3, generator=generator)
    target_verts = torch.randn(frames, 32, 3, generator=generator)
    return {
        "pred_j3d": target_j3d + 0.01 * torch.randn(frames, 14, 3, generator=generator),
        "target_j3d": target_j3d,
        "pred_verts": target_verts + 0.01 * torch.randn(frames, 32, 3, generator=generator),
        "target_verts": target_verts,
    }


def _old_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """Minimal pre-guard implementation retained only for finite-value regression."""

    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)
    K = X1.bmm(X2.permute(0, 2, 1))
    U, _, V = torch.svd(K)
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0).repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))
    scale = torch.stack([torch.trace(value) for value in R.bmm(K)]) / var1
    translation = mu2 - scale[:, None, None] * R.bmm(mu1)
    aligned = scale[:, None, None] * R.bmm(S1) + translation
    return aligned.permute(0, 2, 1) if transposed else aligned


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def test_compute_camcoord_metrics_accepts_finite_input() -> None:
    metrics = compute_camcoord_metrics(_metric_batch(), sequence_id="finite-sequence")
    assert set(metrics) == {"pa_mpjpe", "mpjpe", "pve", "accel"}
    assert metrics["pa_mpjpe"].shape == (7,)
    assert metrics["mpjpe"].shape == (7,)
    assert metrics["pve"].shape == (7,)
    assert metrics["accel"].shape == (5,)
    assert all(np.isfinite(value).all() for value in metrics.values())


def test_nan_reports_tensor_sequence_and_frame() -> None:
    batch = _metric_batch()
    batch["pred_j3d"][3, 0, 0] = float("nan")
    with pytest.raises(InvalidMetricInputError) as caught:
        compute_camcoord_metrics(batch, sequence_id="nan-sequence")
    message = str(caught.value)
    assert "nan-sequence" in message
    assert "pred_j3d" in message
    assert "bad_frames=[3]" in message
    assert "NaN=1" in message


@pytest.mark.parametrize("value,label", [(float("inf"), "+Inf=1"), (-float("inf"), "-Inf=1")])
def test_inf_reports_tensor_frame_and_sign(value: float, label: str) -> None:
    batch = _metric_batch()
    batch["pred_verts"][4, 2, 1] = value
    with pytest.raises(InvalidMetricInputError) as caught:
        compute_camcoord_metrics(batch, sequence_id="inf-sequence")
    message = str(caught.value)
    assert "pred_verts" in message
    assert "bad_frames=[4]" in message
    assert label in message


def test_degenerate_human_frame_is_rejected_before_svd() -> None:
    batch = _metric_batch()
    batch["pred_j3d"][2] = torch.tensor([2.0, -1.0, 0.5])
    with pytest.raises(InvalidMetricInputError, match="degenerate") as caught:
        compute_camcoord_metrics(batch, sequence_id="collapsed-human")
    assert "bad_frames=[2]" in str(caught.value)


def test_procrustes_recovers_known_similarity_transform() -> None:
    generator = torch.Generator().manual_seed(17)
    source = torch.randn(5, 20, 3, generator=generator)
    angle = torch.tensor(0.7)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = 1.7 * torch.einsum("ij,bnj->bni", rotation, source)
    target = target + torch.tensor([0.3, -0.4, 1.2])
    aligned = batch_compute_similarity_transform_torch(
        source, target, sequence_id="known-transform"
    )
    torch.testing.assert_close(aligned, target, rtol=1e-5, atol=1e-5)


def test_procrustes_zero_variance_is_rejected_before_svd() -> None:
    source = torch.ones(4, 12, 3)
    target = torch.randn(4, 12, 3)
    with pytest.raises(InvalidMetricInputError, match="variance") as caught:
        batch_compute_similarity_transform_torch(source, target, sequence_id="zero-variance")
    assert "bad_frames=[0, 1, 2, 3]" in str(caught.value)


def test_finite_procrustes_matches_previous_implementation() -> None:
    generator = torch.Generator().manual_seed(1234)
    source = torch.randn(8, 17, 3, generator=generator)
    target = torch.randn(8, 17, 3, generator=generator)
    reference = _old_similarity_transform(source.clone(), target.clone())
    guarded = batch_compute_similarity_transform_torch(source, target, sequence_id="regression")
    torch.testing.assert_close(guarded, reference, rtol=1e-4, atol=1e-5)


def test_finite_checker_handles_scalar_and_empty_tensors() -> None:
    check_finite_metric_inputs(
        {"scalar": torch.tensor(1.0), "empty": torch.empty(0, 3)},
        sequence_id="edge-shapes",
    )
    with pytest.raises(InvalidMetricInputError, match=r"bad_frames=\[0\]"):
        check_finite_metric_inputs({"scalar": torch.tensor(float("nan"))}, sequence_id="bad-scalar")


def test_dump_is_compact_atomic_sanitized_and_unique(tmp_path) -> None:
    trainer = SimpleNamespace(
        checkpoint_callbacks=[SimpleNamespace(dirpath=tmp_path / "run" / "checkpoints")],
        log_dir=None,
        default_root_dir=tmp_path / "fallback",
        global_step=3000,
        current_epoch=18,
        global_rank=3,
    )
    outputs = {
        "pred_body_params_incam": {
            "body_pose": torch.zeros(6, 63),
            "global_orient": torch.zeros(6, 3),
            "transl": torch.zeros(6, 3),
            "betas": torch.zeros(6, 10),
        },
        "pred_body_params_global": {"transl": torch.ones(6, 3)},
        "pred_verts": torch.zeros(6, 10475, 3),
    }
    batch = {
        "length": torch.tensor([6]),
        "gender": ["neutral"],
        "meta": [{"dataset_id": "EMDB_1", "vid": "P8/64 outdoor skateboard"}],
    }
    first = dump_invalid_eval_sample(
        trainer,
        "EMDB_1",
        "P8/64 outdoor skateboard",
        "non-finite prediction",
        outputs,
        batch=batch,
        tensor_diagnostics={"pred": {"bad_frames": [3]}},
    )
    second = dump_invalid_eval_sample(
        trainer,
        "EMDB_1",
        "P8/64 outdoor skateboard",
        "non-finite prediction",
        outputs,
        batch=batch,
    )
    assert first is not None and second is not None and first != second
    assert first.is_file() and second.is_file()
    assert "P8_64_outdoor_skateboard" in first.name
    assert not list(first.parent.glob("*.tmp"))
    assert not list(first.parent.glob(".*.tmp"))
    payload = _torch_load(first)
    assert payload["dataset_id"] == "EMDB_1"
    assert payload["batch_metadata"]["vid"] == "P8/64 outdoor skateboard"
    assert "pred_body_params_incam" in payload["predictions"]
    assert "body_pose" in payload["predictions"]["pred_body_params_incam"]
    assert "pred_verts" not in payload
    assert "pred_verts" not in payload["predictions"]


def test_invalid_policy_skip_raise_and_validation() -> None:
    error = InvalidMetricInputError("bad sequence")
    assert validate_invalid_policy("skip") == "skip"
    assert validate_invalid_policy("raise") == "raise"
    assert apply_invalid_policy("skip", error) is None
    with pytest.raises(InvalidMetricInputError, match="bad sequence"):
        apply_invalid_policy("raise", error)
    with pytest.raises(ValueError, match="invalid_policy"):
        validate_invalid_policy("ignore")
