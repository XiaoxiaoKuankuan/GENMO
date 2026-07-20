"""Tests for the unified webcam rendering/GMR SMPL-X shape policy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import scripts.demo.demo_webcam as demo_webcam
from gem.smplx_gmr_reference import BetaStabilizer
from scripts.demo.demo_webcam import (
    WebcamGEMSMPLDemo,
    parse_args,
    resolve_effective_betas,
)


def test_resolve_effective_betas_zero_returns_same_shape() -> None:
    raw = torch.randn(1, 10, dtype=torch.float32)
    result = resolve_effective_betas(
        raw,
        BetaStabilizer("zero"),
        reference=torch.zeros(1, 63),
    )
    assert result.shape == (1, 10)
    assert result.dtype == raw.dtype
    assert result.device == raw.device
    assert torch.count_nonzero(result) == 0


def test_resolve_effective_betas_zero_without_raw_uses_reference() -> None:
    reference = torch.zeros(2, 3, 63, dtype=torch.float64)
    result = resolve_effective_betas(
        None,
        BetaStabilizer("zero"),
        reference=reference,
    )
    assert result.shape == (2, 3, 10)
    assert result.dtype == reference.dtype
    assert result.device == reference.device
    assert torch.count_nonzero(result) == 0


@pytest.mark.parametrize("mode", ["first", "mean", "ema", "per_frame"])
def test_resolve_effective_betas_missing_raw_rejected_for_predicted_modes(mode: str) -> None:
    with pytest.raises(ValueError, match=rf"shape_mode={mode} requires predicted betas"):
        resolve_effective_betas(
            None,
            BetaStabilizer(mode),
            reference=torch.zeros(1, 63),
        )


@pytest.mark.parametrize("shape", [(9,), (1, 9), (1, 2, 11), ()])
def test_resolve_effective_betas_rejects_invalid_last_dimension(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="last dimension 10"):
        resolve_effective_betas(
            torch.zeros(shape),
            BetaStabilizer("zero"),
            reference=torch.zeros(1, 63),
        )


def test_resolve_effective_betas_rejects_nonfinite_result() -> None:
    class NonfiniteStabilizer:
        mode = "zero"

        @staticmethod
        def update(value: torch.Tensor) -> torch.Tensor:
            return torch.full_like(value, float("nan"))

    with pytest.raises(ValueError, match="effective SMPL-X betas contains NaN or Inf"):
        resolve_effective_betas(
            torch.ones(1, 10),
            NonfiniteStabilizer(),
            reference=torch.zeros(1, 63),
        )


def test_shape_cli_default_and_both_names() -> None:
    assert parse_args([]).shape_mode == "zero"
    assert parse_args(["--shape_mode", "zero"]).shape_mode == "zero"
    assert parse_args(["--gmr_shape_mode", "mean"]).shape_mode == "mean"
    args = parse_args(["--gmr_shape_warmup", "7"])
    assert args.shape_warmup == 7
    assert not hasattr(args, "gmr_shape_mode")
    assert not hasattr(args, "gmr_shape_warmup")


def test_backend_zero_shape_is_shared_by_results_and_gmr_fk(monkeypatch) -> None:
    """Exercise the backend handoff without loading camera or ONNX models."""
    device = demo_webcam._DEVICE
    frame_count = 2

    class FakeEnDecoder:
        def __init__(self) -> None:
            self.fk_betas = None

        def decode(self, _pred_x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "body_pose": torch.zeros(1, frame_count, 63, device=device),
                "global_orient": torch.zeros(1, frame_count, 3, device=device),
                "global_orient_gv": torch.zeros(1, frame_count, 6, device=device),
                "local_transl_vel": torch.zeros(1, frame_count, 3, device=device),
                "betas": torch.full((1, frame_count, 10), 3.0, device=device),
            }

        def fk_v2(self, *, betas: torch.Tensor, **_kwargs):
            self.fk_betas = betas.detach().clone()
            joints = torch.zeros(1, 1, 22, 3, device=betas.device)
            fk_mat = torch.eye(4, device=betas.device).reshape(1, 1, 1, 4, 4)
            fk_mat = fk_mat.repeat(1, 1, 22, 1, 1)
            return joints, None, fk_mat

    class FakeBridge:
        sequence = 0

        def __init__(self) -> None:
            self.sent = False

        def send_smplx_targets(self, _targets, *, source_stamp_ns: int) -> None:
            assert source_stamp_ns > 0
            self.sent = True

    class FakeAdapter:
        @staticmethod
        def adapt(*_args, **_kwargs):
            return SimpleNamespace(scaled_targets={})

    monkeypatch.setattr(
        demo_webcam,
        "run_denoiser",
        lambda *_args, **_kwargs: (
            torch.zeros(1, frame_count, 151, device=device),
            torch.zeros(1, frame_count, 3, device=device),
        ),
    )
    monkeypatch.setattr(
        demo_webcam,
        "compute_transl_full_cam",
        lambda *_args, **_kwargs: torch.zeros(1, 1, 3, device=device),
    )
    monkeypatch.setattr(demo_webcam, "init_rollout_w_Rt_state", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        demo_webcam,
        "rollout_step_w_Rt",
        lambda state, **_kwargs: (
            {
                "global_orient": torch.zeros(1, 3, device=device),
                "transl": torch.zeros(1, 3, device=device),
            },
            state,
        ),
    )

    backend = WebcamGEMSMPLDemo.__new__(WebcamGEMSMPLDemo)
    backend.K_fullimg = torch.eye(3)
    backend._cam_angvel_static = torch.zeros(frame_count, 6)
    backend._f_imgseq_zeros = torch.zeros(1, frame_count, 1024)
    backend.no_imgfeat = True
    backend.denoiser_runner = object()
    backend.denoiser_backend = "fake"
    backend.endecoder = FakeEnDecoder()
    backend.rollout_state = None
    backend.shape_stabilizer = BetaStabilizer("zero")
    backend._shape_logged = False
    backend.gmr_bridge = FakeBridge()
    backend.gmr_adapter = FakeAdapter()
    backend._last_gmr_error_log = 0.0

    result = backend._run_backend(
        torch.ones(frame_count, 3),
        torch.zeros(frame_count, 17, 3),
        torch.zeros(frame_count, 1024),
        torch.zeros(17, 3),
    )

    incam_betas = result["body_params_incam"]["betas"]
    global_betas = result["body_params_global"]["betas"]
    assert incam_betas.shape == (1, 10)
    assert global_betas.shape == (1, 10)
    assert torch.count_nonzero(incam_betas) == 0
    assert torch.count_nonzero(global_betas) == 0
    assert backend.endecoder.fk_betas.shape == (1, 1, 10)
    assert torch.count_nonzero(backend.endecoder.fk_betas) == 0
    assert backend.gmr_bridge.sent
