"""CPU-facing tests for the persistent GMR streamer CLI and FK boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gem.runtime.motion_streamer import SMPLFrame
from gem.smplx_gmr_reference import SMPLXGMRReference
from scripts.demo import stream_smpl_params_to_gmr as streamer
from scripts.tools.extract_smpl_idle_pose import extract_idle_pose


def write_motion(path: Path, length: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "body_params_global": {
                "body_pose": torch.zeros(length, 63),
                "global_orient": torch.zeros(length, 3),
                "transl": torch.zeros(length, 3),
                "betas": torch.randn(length, 10),
            },
            "fps": 30.0,
        },
        path,
    )
    return path


def test_cli_defaults_and_shape_policy() -> None:
    args = streamer.parse_args(["--motion", "example.pt"])
    assert args.gmr_host == "127.0.0.1"
    assert args.gmr_port == 7006
    assert args.publish_fps == 30
    assert args.shape_mode == "zero"
    assert args.mode == "sim"
    assert args.poll_interval == 0.2
    assert args.new_motion_policy == "queue"
    assert not args.loop


def test_cli_requires_source_and_robot_idle() -> None:
    with pytest.raises(ValueError, match="--motion or --watch_dir"):
        streamer.parse_args([])
    with pytest.raises(RuntimeError, match="verified idle"):
        streamer.parse_args(["--motion", "example.pt", "--mode", "robot"])
    with pytest.raises(RuntimeError, match="forbids"):
        streamer.parse_args(
            [
                "--motion",
                "example.pt",
                "--mode",
                "robot",
                "--idle_motion",
                "idle.pt",
                "--new_motion_policy",
                "interrupt",
            ]
        )


def test_dry_run_never_constructs_udp_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    motion = write_motion(tmp_path / "smpl_params.pt")

    def forbidden_bridge(*_args, **_kwargs):
        raise AssertionError("dry-run created a UDP bridge")

    monkeypatch.setattr(streamer, "GMRUDPBridge", forbidden_bridge)
    assert streamer.main(["--motion", str(motion), "--dry_run"]) == 0
    output = capsys.readouterr().out
    assert "Betas norm: 0.000000" in output
    assert "All finite: True" in output
    assert "UDP socket: not created" in output


def test_idle_extractor_saves_one_verified_zero_shape_frame(tmp_path: Path) -> None:
    motion = write_motion(tmp_path / "source.pt", length=3)
    output = extract_idle_pose(motion, 1, tmp_path / "idle.pt")
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["num_frames"] == 1
    assert payload["shape_mode"] == "zero"
    assert payload["body_params_global"]["body_pose"].shape == (1, 63)
    assert torch.count_nonzero(payload["body_params_global"]["betas"]) == 0


class FakeEnDecoder:
    def __init__(self) -> None:
        self.received_betas: torch.Tensor | None = None

    def fk_v2(self, *, body_pose, betas, global_orient, transl, get_intermediate):
        assert body_pose.shape == (1, 1, 63)
        assert global_orient.shape == (1, 1, 3)
        assert transl.shape == (1, 1, 3)
        assert get_intermediate
        self.received_betas = betas.detach().cpu()
        joints = torch.zeros(1, 1, 22, 3, device=body_pose.device)
        fk_mat = torch.eye(4, device=body_pose.device).reshape(1, 1, 1, 4, 4).repeat(1, 1, 22, 1, 1)
        return joints, fk_mat.clone(), fk_mat


class FakeBridge:
    sequence = 11

    def __init__(self) -> None:
        self.calls = []

    def send_smplx_targets(self, targets, source_stamp_ns=None):
        self.calls.append((targets, source_stamp_ns))
        return b"SMP1-test"


def test_mock_bridge_receives_finite_targets_and_fk_gets_zero_betas() -> None:
    frame = SMPLFrame(
        torch.zeros(63),
        torch.zeros(3),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.full((10,), 99.0),
    )
    endecoder = FakeEnDecoder()
    adapter = SMPLXGMRReference()
    bridge = FakeBridge()
    packet = streamer.send_frame_to_gmr(
        frame,
        endecoder,
        adapter,
        bridge,
        device=torch.device("cpu"),
        timestamp_ns=123,
    )
    assert packet == b"SMP1-test"
    assert endecoder.received_betas is not None
    assert torch.count_nonzero(endecoder.received_betas) == 0
    assert len(bridge.calls) == 1
    targets, stamp = bridge.calls[0]
    assert stamp == 123 and len(targets) == 14
    for target in targets.values():
        assert torch.isfinite(torch.as_tensor(target.position_zup)).all()
        assert torch.isfinite(torch.as_tensor(target.rotation_zup)).all()


@pytest.mark.skipif(
    not Path("inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz").is_file(),
    reason="SMPL-X body model is unavailable",
)
def test_real_cpu_fk_output_is_finite_and_zero_shape() -> None:
    endecoder = streamer.load_endecoder(torch.device("cpu"))
    adapter = SMPLXGMRReference()
    adapted = streamer.fk_and_adapt_frame(
        SMPLFrame(torch.zeros(63), torch.zeros(3), torch.zeros(3), torch.ones(10)),
        endecoder,
        adapter,
        device=torch.device("cpu"),
        frame_id=0,
        timestamp_ns=0,
    )
    assert len(adapted.scaled_targets) == 14
    assert all(
        torch.isfinite(torch.as_tensor(target.position_zup)).all()
        for target in adapted.scaled_targets.values()
    )
