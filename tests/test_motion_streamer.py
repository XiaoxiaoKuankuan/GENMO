"""CPU tests for validated, continuous SMPL-X motion playback."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from gem.runtime.motion_streamer import (
    LEFT_SHOULDER_BODY_INDEX,
    RIGHT_SHOULDER_BODY_INDEX,
    SYNTHETIC_IDLE_ARM_ANGLE_RAD,
    MonotonicDeadline,
    MotionPlayer,
    MotionQueue,
    MotionWatcher,
    PlayerState,
    SMPLFrame,
    align_motion_root_yaw,
    align_motion_to_frame,
    interpolate_axis_angle,
    interpolate_frames,
    load_smpl_motion,
    sample_motion_at,
    smoothstep,
    synthetic_idle_motion,
)
from gem.utils.rotation_conversions import axis_angle_to_matrix


def write_motion(
    path: Path,
    *,
    length: int = 4,
    fps: float = 30.0,
    body_pose: torch.Tensor | None = None,
    global_orient: torch.Tensor | None = None,
    transl: torch.Tensor | None = None,
    betas: torch.Tensor | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body_pose = torch.zeros(length, 63) if body_pose is None else body_pose
    global_orient = torch.zeros(length, 3) if global_orient is None else global_orient
    transl = torch.zeros(length, 3) if transl is None else transl
    betas = torch.ones(length, 10) if betas is None else betas
    torch.save(
        {
            "body_params_global": {
                "body_pose": body_pose,
                "global_orient": global_orient,
                "transl": transl,
                "betas": betas,
            },
            "fps": fps,
            "source": "test",
        },
        path,
    )
    return path


def test_load_valid_motion_converts_cpu_float32_and_zero_shape(tmp_path: Path) -> None:
    path = write_motion(tmp_path / "smpl_params.pt")
    motion = load_smpl_motion(path)
    assert motion.num_frames == 4
    assert motion.body_pose.dtype == torch.float32
    assert motion.body_pose.device.type == "cpu"
    assert torch.count_nonzero(motion.betas) == 0
    assert motion.metadata["shape_mode"] == "zero"


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("body_pose", torch.zeros(4, 62), "body_pose must have shape"),
        ("global_orient", torch.zeros(3, 3), "first dimension"),
        ("transl", torch.zeros(4, 4), "transl must have shape"),
        ("betas", torch.zeros(4, 9), "betas must have shape"),
    ],
)
def test_load_rejects_invalid_shapes(
    tmp_path: Path, field: str, replacement: torch.Tensor, message: str
) -> None:
    values = {field: replacement}
    path = write_motion(tmp_path / f"{field}.pt", **values)
    with pytest.raises(ValueError, match=message):
        load_smpl_motion(path)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_load_rejects_nonfinite(tmp_path: Path, bad_value: float) -> None:
    pose = torch.zeros(4, 63)
    pose[1, 2] = bad_value
    path = write_motion(tmp_path / "bad.pt", body_pose=pose)
    with pytest.raises(ValueError, match="NaN or Inf"):
        load_smpl_motion(path)


def test_load_rejects_empty_and_bad_fps(tmp_path: Path) -> None:
    empty = write_motion(tmp_path / "empty.pt", length=0)
    with pytest.raises(ValueError, match="at least 2"):
        load_smpl_motion(empty)
    bad_fps = write_motion(tmp_path / "fps.pt", fps=0)
    with pytest.raises(ValueError, match="fps"):
        load_smpl_motion(bad_fps)


def test_ready_watcher_ignores_incomplete_and_deduplicates(tmp_path: Path) -> None:
    watcher = MotionWatcher(tmp_path)
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    assert watcher.scan() == []

    ready = tmp_path / "complete"
    ready.mkdir()
    (ready / "metadata.json").write_text(
        json.dumps({"completed_at": "2026-01-01T00:00:00Z"}), encoding="utf-8"
    )
    (ready / "READY").write_text("ok\n", encoding="utf-8")
    assert watcher.scan() == [ready]
    watcher.mark_consumed(ready)
    assert watcher.scan() == []
    restarted = MotionWatcher(tmp_path)
    assert restarted.scan() == []


def test_watcher_does_not_replay_preexisting_by_default(tmp_path: Path) -> None:
    ready = tmp_path / "old"
    ready.mkdir()
    (ready / "READY").touch()
    assert MotionWatcher(tmp_path).scan() == []
    assert MotionWatcher(tmp_path, replay_existing=True).scan() == [ready]


def _ready_source(root: Path, name: str, source: str | None) -> Path:
    path = root / name
    path.mkdir()
    if source is not None:
        (path / "metadata.json").write_text(
            json.dumps({"source": source, "completed_at": "2026-07-20T00:00:00Z"}),
            encoding="utf-8",
        )
    (path / "READY").touch()
    return path


def test_watcher_source_filters_do_not_consume_mismatches(tmp_path: Path) -> None:
    music = _ready_source(tmp_path, "music", "music_only")
    text = _ready_source(tmp_path, "text", "text_only")
    assert MotionWatcher(
        tmp_path, replay_existing=True, source_filter="music_only", logger=None
    ).scan() == [music]
    text_watcher = MotionWatcher(
        tmp_path, replay_existing=True, source_filter="text_only", logger=None
    )
    assert text_watcher.scan() == [text]
    assert str(music.resolve()) not in text_watcher.consumed
    any_paths = MotionWatcher(
        tmp_path, replay_existing=True, source_filter="any", logger=None
    ).scan()
    assert any_paths == [music, text]


def test_filtered_watcher_warns_and_ignores_bad_metadata(tmp_path: Path) -> None:
    missing = _ready_source(tmp_path, "missing", None)
    broken = _ready_source(tmp_path, "broken", "music_only")
    (broken / "metadata.json").write_text("not-json", encoding="utf-8")
    logs: list[str] = []
    watcher = MotionWatcher(
        tmp_path, replay_existing=True, source_filter="music_only", logger=logs.append
    )
    assert watcher.scan() == []
    assert len(logs) == 2
    assert str(missing.resolve()) not in watcher.consumed
    assert str(broken.resolve()) not in watcher.consumed


def test_music_metadata_is_preserved_by_loader(tmp_path: Path) -> None:
    path = write_motion(tmp_path / "music.pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.update(
        {
            "source": "music_only",
            "audio_path": "/tmp/song.wav",
            "audio_start_sec": 1.0,
            "audio_duration_sec": 2.0,
            "feature_type": "edge_baseline35",
            "feature_fps": 30,
            "estimated_bpm": 120.0,
            "bpm_source": "estimated",
            "sample_index": 3,
            "guidance_scale": 2.5,
            "ddim_steps": 50,
        }
    )
    torch.save(payload, path)
    metadata = load_smpl_motion(path).metadata
    assert metadata["source"] == "music_only"
    assert metadata["audio_path"] == "/tmp/song.wav"
    assert metadata["estimated_bpm"] == 120.0
    assert metadata["sample_index"] == 3


def test_queue_latest_and_interrupt_policies(tmp_path: Path) -> None:
    first = load_smpl_motion(write_motion(tmp_path / "first.pt"))
    second = load_smpl_motion(write_motion(tmp_path / "second.pt"))
    queue = MotionQueue()
    queue.add(first, "queue")
    queue.add(second, "queue")
    assert len(queue) == 2
    queue.add(first, "latest")
    assert len(queue) == 1 and queue.pop().source_path == first.source_path

    player = MotionPlayer(synthetic_idle_motion(), logger=None)
    player.start(0.0)
    player.enqueue(first, policy="queue", now=0.0)
    assert player.state == PlayerState.BLENDING
    player.tick(0.8)
    assert player.state == PlayerState.PLAYING
    player.enqueue(second, policy="interrupt", now=0.9)
    assert player.state == PlayerState.BLENDING
    assert player.active_motion is not None
    assert player.active_motion.source_path == second.source_path


def test_holding_continuously_returns_static_frames() -> None:
    player = MotionPlayer(synthetic_idle_motion(), logger=None)
    first = player.tick(0.0)
    second = player.tick(10.0)
    assert player.state == PlayerState.HOLDING
    assert torch.equal(first.body_pose, second.body_pose)
    assert torch.equal(first.transl, second.transl)
    pose = first.body_pose.reshape(21, 3)
    assert pose[LEFT_SHOULDER_BODY_INDEX, 2].item() == pytest.approx(-SYNTHETIC_IDLE_ARM_ANGLE_RAD)
    assert pose[RIGHT_SHOULDER_BODY_INDEX, 2].item() == pytest.approx(SYNTHETIC_IDLE_ARM_ANGLE_RAD)
    assert torch.count_nonzero(pose).item() == 2


def test_synthetic_idle_supports_small_symmetric_arm_opening() -> None:
    idle = synthetic_idle_motion(arm_open_degrees=10.0)
    pose = idle.body_pose.reshape(21, 3)
    expected_rotation = math.radians(80.0)
    assert pose[LEFT_SHOULDER_BODY_INDEX, 2].item() == pytest.approx(
        -expected_rotation
    )
    assert pose[RIGHT_SHOULDER_BODY_INDEX, 2].item() == pytest.approx(
        expected_rotation
    )
    assert idle.metadata["arm_open_degrees"] == 10.0
    with pytest.raises(ValueError, match="within"):
        synthetic_idle_motion(arm_open_degrees=45.1)


def test_motion_end_returns_then_holds(tmp_path: Path) -> None:
    transl = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    motion = load_smpl_motion(write_motion(tmp_path / "short.pt", length=2, fps=1, transl=transl))
    player = MotionPlayer(
        synthetic_idle_motion(), blend_seconds=0.5, return_seconds=1.0, logger=None
    )
    player.start(0.0)
    player.enqueue(motion, policy="queue", now=0.0)
    player.tick(0.5)
    assert player.state == PlayerState.PLAYING
    player.tick(2.5)
    assert player.state == PlayerState.RETURNING
    final_action_root = player.current_frame.transl.clone()
    player.tick(3.0)
    assert player.state == PlayerState.RETURNING
    player.tick(3.5)
    assert player.state == PlayerState.HOLDING
    # Idle root remains under the current feet instead of jumping to origin.
    assert torch.allclose(player.current_frame.transl, final_action_root)


def test_axis_angle_slerp_and_smooth_translation() -> None:
    zero = torch.zeros(3)
    right_angle = torch.tensor([0.0, math.pi / 2.0, 0.0])
    midpoint = interpolate_axis_angle(zero, right_angle, 0.5)
    expected = axis_angle_to_matrix(torch.tensor([0.0, math.pi / 4.0, 0.0]))
    assert torch.allclose(axis_angle_to_matrix(midpoint), expected, atol=1e-5)

    frame0 = SMPLFrame(torch.zeros(63), zero, torch.zeros(3), torch.zeros(10))
    frame1 = SMPLFrame(torch.zeros(63), zero, torch.tensor([8.0, 0.0, 0.0]), torch.zeros(10))
    frame = interpolate_frames(frame0, frame1, 0.25)
    assert smoothstep(0.25) == pytest.approx(0.15625)
    assert frame.transl[0].item() == pytest.approx(1.25)


def test_root_translation_and_yaw_alignment(tmp_path: Path) -> None:
    orient = torch.zeros(2, 3)
    transl = torch.tensor([[4.0, 0.0, 9.0], [4.0, 0.0, 10.0]])
    current_orient = torch.tensor([0.0, math.pi / 2.0, 0.0])
    current_transl = torch.tensor([2.0, 0.0, 3.0])
    aligned_orient, aligned_transl = align_motion_root_yaw(
        orient, transl, current_orient, current_transl
    )
    assert torch.allclose(aligned_transl[0], current_transl, atol=1e-5)
    assert torch.allclose(aligned_transl[1], torch.tensor([3.0, 0.0, 3.0]), atol=1e-5)
    assert torch.allclose(
        axis_angle_to_matrix(aligned_orient[0]),
        axis_angle_to_matrix(current_orient),
        atol=1e-5,
    )

    motion = load_smpl_motion(
        write_motion(tmp_path / "align.pt", length=2, global_orient=orient, transl=transl)
    )
    aligned = align_motion_to_frame(
        motion,
        SMPLFrame(torch.zeros(63), current_orient, current_transl, torch.zeros(10)),
    )
    assert torch.allclose(aligned.transl[0], current_transl, atol=1e-5)
    assert torch.count_nonzero(aligned.betas) == 0


def test_time_resampling_uses_source_fps(tmp_path: Path) -> None:
    transl = torch.zeros(3, 3)
    transl[:, 0] = torch.tensor([0.0, 1.0, 2.0])
    motion = load_smpl_motion(write_motion(tmp_path / "fps.pt", length=3, fps=2, transl=transl))
    frame, frame_float, finished = sample_motion_at(motion, 0.25)
    assert frame_float == pytest.approx(0.5)
    assert frame.transl[0].item() == pytest.approx(0.5)
    assert not finished
    _, _, finished = sample_motion_at(motion, 1.0)
    assert not finished
    _, _, finished = sample_motion_at(motion, 1.5)
    assert finished


def test_deadline_skips_stale_sends_without_burst() -> None:
    deadline = MonotonicDeadline(10.0, 0.0)
    assert deadline.advance(0.0) == 0
    skipped = deadline.advance(0.55)
    assert skipped == 4
    assert deadline.next_deadline == pytest.approx(0.6)
    assert deadline.seconds_until(0.55) == pytest.approx(0.05)


def test_estop_is_latched_and_blends_to_idle(tmp_path: Path) -> None:
    transl = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    motion = load_smpl_motion(write_motion(tmp_path / "move.pt", length=2, fps=1, transl=transl))
    player = MotionPlayer(
        synthetic_idle_motion(), blend_seconds=0.2, estop_blend_seconds=0.3, logger=None
    )
    player.start(0.0)
    player.enqueue(motion, policy="queue", now=0.0)
    player.tick(0.2)
    player.tick(0.7)
    before = player.current_frame.transl.clone()
    player.tick(0.7, estop=True)
    assert player.state == PlayerState.ESTOP and player.estop_latched
    after = player.tick(1.0)
    assert torch.allclose(after.transl, before)
    # File removal alone is represented by estop=False and does not unlatch.
    player.tick(2.0, estop=False)
    assert player.state == PlayerState.ESTOP
    player.reset_estop(2.0)
    assert player.state == PlayerState.HOLDING and not player.estop_latched
