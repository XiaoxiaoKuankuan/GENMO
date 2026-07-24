"""CPU tests for the single-output video/generated-motion multiplexer."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from gem.runtime.motion_source_mux import MotionSourceMux, MuxState
from gem.runtime.motion_streamer import PlayerState, SMPLFrame
from gem.utils.rotation_conversions import axis_angle_to_matrix


def frame(value: float = 0.0) -> SMPLFrame:
    return SMPLFrame(
        body_pose=torch.full((63,), value),
        global_orient=torch.zeros(3),
        transl=torch.tensor([value, 0.0, 0.0]),
        betas=torch.ones(10),
    )


def write_motion(root: Path, *, source: str = "text_only") -> Path:
    root.mkdir()
    length = 3
    payload = {
        "body_params_global": {
            "body_pose": torch.stack([torch.full((63,), float(index)) for index in range(length)]),
            "global_orient": torch.zeros(length, 3),
            "transl": torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]),
            "betas": torch.ones(length, 10),
        },
        "fps": 10.0,
        "source": source,
        "metadata": {"source": source},
    }
    torch.save(payload, root / "smpl_params.pt")
    (root / "READY").write_text("ready\n", encoding="utf-8")
    return root


@pytest.fixture()
def clock():
    value = {"now": 0.0}
    return value, lambda: value["now"], lambda: int(value["now"] * 1e9)


def make_mux(clock, **kwargs) -> MotionSourceMux:
    value, now, now_ns = clock
    del value
    return MotionSourceMux(
        dry_run=True,
        endecoder=torch.nn.Identity(),
        adapter_factory=lambda *_args: object(),
        clock=now,
        clock_ns=now_ns,
        blend_seconds=0.1,
        return_seconds=0.1,
        estop_blend_seconds=0.05,
        video_stale_sec=0.5,
        **kwargs,
    )


def test_live_video_keeps_only_latest_and_stale_returns_idle(clock) -> None:
    values, _, _ = clock
    mux = make_mux(clock)
    mux.start_video_mode()
    mux.submit_video_frame(frame(0), timestamp=0.0)
    mux.submit_video_frame(frame(2), timestamp=0.0)
    tick = mux.tick_once(now=0.11)
    assert tick.state == MuxState.VIDEO_LIVE
    assert torch.equal(tick.frame.transl, torch.tensor([2.0, 0.0, 0.0]))
    values["now"] = 1.0
    stale = mux.tick_once(now=1.0)
    assert stale.state == MuxState.IDLE
    assert stale.source == "idle"
    assert torch.equal(stale.frame.transl, torch.tensor([2.0, 0.0, 0.0]))


def test_new_video_source_aligns_root_to_current_world_pose(clock) -> None:
    mux = make_mux(clock)
    current = SMPLFrame(
        body_pose=torch.zeros(63),
        global_orient=torch.tensor([0.0, torch.pi / 2.0, 0.0]),
        transl=torch.tensor([5.0, 0.0, 3.0]),
        betas=torch.zeros(10),
    )
    mux._last_output = current.clone()
    mux.player.current_frame = current.clone()
    mux.start_video_mode()

    source_first = SMPLFrame(
        body_pose=torch.ones(63),
        global_orient=torch.tensor([0.0, torch.pi, 0.0]),
        transl=torch.zeros(3),
        betas=torch.ones(10),
    )
    mux.submit_video_frame(source_first, timestamp=0.0)
    first = mux.tick_once(now=0.0)
    assert torch.allclose(first.frame.transl, current.transl)
    assert torch.allclose(
        axis_angle_to_matrix(first.frame.global_orient),
        axis_angle_to_matrix(current.global_orient),
        atol=1e-5,
    )

    source_next = SMPLFrame(
        body_pose=torch.ones(63),
        global_orient=source_first.global_orient,
        transl=torch.tensor([1.0, 0.0, 0.0]),
        betas=torch.ones(10),
    )
    mux.submit_video_frame(source_next, timestamp=0.05)
    aligned = mux.tick_once(now=0.11)
    assert aligned.state == MuxState.VIDEO_LIVE
    assert torch.allclose(
        axis_angle_to_matrix(aligned.frame.global_orient),
        axis_angle_to_matrix(current.global_orient),
        atol=1e-5,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(aligned.frame.transl - current.transl),
        torch.tensor(1.0),
        atol=1e-5,
    )
    assert not torch.allclose(aligned.frame.transl, source_next.transl)
    assert mux.status()["video_root_aligned"] is True
    assert mux.status()["video_alignment_count"] == 1


def test_generated_clip_blends_plays_returns_and_requests_video_reset(
    clock, tmp_path: Path
) -> None:
    values, _, _ = clock
    resets = []
    mux = make_mux(clock)
    mux.on_video_resume_reset = lambda: resets.append(True)
    mux.start_video_mode()
    mux.submit_video_frame(frame(0.5), timestamp=0)
    mux.tick_once(now=0)
    motion = mux.submit_generated_motion(write_motion(tmp_path / "motion"))
    assert torch.count_nonzero(motion.betas) == 0

    assert mux.tick_once(now=0).state == MuxState.CLIP_PENDING
    values["now"] = 0.11
    assert mux.tick_once(now=0.11).state == MuxState.CLIP_PLAYING
    assert mux.player.state == PlayerState.PLAYING
    values["now"] = 0.42
    mux.tick_once(now=0.42)
    assert mux.player.state == PlayerState.RETURNING
    values["now"] = 0.53
    returned = mux.tick_once(now=0.53)
    assert mux.player.state == PlayerState.HOLDING
    assert resets == [True]
    assert returned.source == "generated_clip"
    assert mux.tick_once(now=0.54).state == MuxState.IDLE


def test_clip_does_not_hold_dangerous_last_frame(clock, tmp_path: Path) -> None:
    values, _, _ = clock
    mux = make_mux(clock)
    mux.submit_generated_motion(write_motion(tmp_path / "motion"))
    mux.tick_once(now=0)
    values["now"] = 0.11
    mux.tick_once(now=0.11)
    values["now"] = 0.42
    last = mux.tick_once(now=0.42)
    assert torch.linalg.vector_norm(last.frame.body_pose) > 0
    values["now"] = 0.53
    returned = mux.tick_once(now=0.53)
    assert torch.linalg.vector_norm(returned.frame.body_pose) < torch.linalg.vector_norm(
        last.frame.body_pose
    )
    values["now"] = 0.6
    held = mux.tick_once(now=0.6)
    assert held.state == MuxState.IDLE


def test_clip_ending_at_180_returns_smoothly_without_idle_snap(clock, tmp_path: Path) -> None:
    values, _, _ = clock
    motion_dir = write_motion(tmp_path / "turn")
    payload = torch.load(
        motion_dir / "smpl_params.pt",
        map_location="cpu",
        weights_only=False,
    )
    payload["body_params_global"]["global_orient"][-1, 1] = torch.pi
    torch.save(payload, motion_dir / "smpl_params.pt")

    mux = make_mux(clock)
    initial = mux.tick_once(now=0.0).frame
    mux.submit_generated_motion(motion_dir)
    mux.tick_once(now=0.0)
    values["now"] = 0.11
    mux.tick_once(now=0.11)
    values["now"] = 0.42
    turned = mux.tick_once(now=0.42)
    assert mux.player.state == PlayerState.RETURNING
    assert not torch.allclose(
        axis_angle_to_matrix(turned.frame.global_orient),
        axis_angle_to_matrix(initial.global_orient),
        atol=1e-3,
    )

    values["now"] = 0.53
    returned = mux.tick_once(now=0.53)
    assert mux.player.state == PlayerState.HOLDING
    values["now"] = 0.54
    held = mux.tick_once(now=0.54)
    assert held.state == MuxState.IDLE
    assert torch.allclose(
        axis_angle_to_matrix(returned.frame.global_orient),
        axis_angle_to_matrix(initial.global_orient),
        atol=1e-5,
    )
    assert torch.allclose(
        axis_angle_to_matrix(held.frame.global_orient),
        axis_angle_to_matrix(returned.frame.global_orient),
        atol=1e-6,
    )

    mux.submit_generated_motion(motion_dir)
    restarted = mux.tick_once(now=0.54)
    assert torch.allclose(
        axis_angle_to_matrix(restarted.frame.global_orient),
        axis_angle_to_matrix(held.frame.global_orient),
        atol=1e-6,
    )


def test_estop_has_priority_and_all_output_betas_are_zero(clock) -> None:
    mux = make_mux(clock)
    mux.start_video_mode()
    mux.submit_video_frame(frame(3), timestamp=0)
    assert torch.count_nonzero(mux.tick_once(now=0).frame.betas) == 0
    mux.estop()
    estop = mux.tick_once(now=0.1)
    assert estop.state == MuxState.ESTOP
    assert torch.count_nonzero(estop.frame.betas) == 0
    mux.clear_estop()
    assert mux.status()["estop"] is False


def test_queue_latest_and_robot_interrupt_safety(clock, tmp_path: Path) -> None:
    mux = make_mux(clock)
    first = write_motion(tmp_path / "first")
    second = write_motion(tmp_path / "second")
    third = write_motion(tmp_path / "third")
    mux.submit_generated_motion(first, policy="queue")
    mux.submit_generated_motion(second, policy="queue")
    assert len(mux.player.queue) == 1
    mux.submit_generated_motion(third, policy="latest")
    assert len(mux.player.queue) == 1
    assert mux.player.queue.pop().source_path.parent == third.resolve()

    with pytest.raises(RuntimeError, match="verified idle"):
        make_mux(clock, mode="robot")


def test_unpublished_temporary_directory_is_rejected(clock, tmp_path: Path) -> None:
    mux = make_mux(clock)
    temporary = tmp_path / ".tmp_partial"
    temporary.mkdir()
    with pytest.raises(RuntimeError, match="READY"):
        mux.submit_generated_motion(temporary)


def test_start_creates_exactly_one_bridge_and_close_releases_it() -> None:
    created = []
    sent = []

    class Bridge:
        sequence = 0

        def close(self):
            self.closed = True

    def bridge_factory(_host, _port, _debug):
        bridge = Bridge()
        created.append(bridge)
        return bridge

    mux = MotionSourceMux(
        publish_fps=100,
        endecoder=torch.nn.Identity(),
        adapter_factory=lambda *_args: object(),
        bridge_factory=bridge_factory,
        frame_sender=lambda *args, **kwargs: sent.append((args, kwargs)),
    )
    mux.start()
    mux.start()
    time.sleep(0.04)
    mux.close()
    assert len(created) == 1
    assert created[0].closed
    assert sent
    assert mux.status()["gmr_sender_instances"] == 0
