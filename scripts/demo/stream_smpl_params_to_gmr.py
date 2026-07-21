# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Persistently stream validated SMPL-X motions to GMR-CPP over SMP1 UDP.

The generator and player are intentionally decoupled: this process maintains a
fixed publish rate and emits either a buffered motion, a safe return, or an idle
pose even while GEM is generating the next sequence.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gem.gmr_udp_bridge import GMRUDPBridge
from gem.runtime.motion_streamer import (
    MonotonicDeadline,
    MotionPlayer,
    MotionWatcher,
    PlayerState,
    SMPLFrame,
    SMPLMotion,
    load_smpl_motion,
    sample_motion_at,
    synthetic_idle_motion,
)
from gem.smplx_gmr_reference import SMPLXGMRReference


class FFplayAudioController:
    """Best-effort, non-blocking source-audio playback tied to player states."""

    def __init__(
        self,
        mode: str = "off",
        *,
        offset_sec: float = 0.0,
        which=shutil.which,
        popen=subprocess.Popen,
        logger=print,
    ) -> None:
        if mode not in {"off", "ffplay"}:
            raise ValueError("audio playback mode must be off or ffplay")
        if not math.isfinite(offset_sec):
            raise ValueError("audio_offset_sec must be finite")
        self.mode = mode
        self.offset_sec = float(offset_sec)
        self._which = which
        self._popen = popen
        self._logger = logger
        self.process: Any | None = None
        self._last_state: PlayerState | None = None
        self._warned_missing = False

    def _warn(self, message: str) -> None:
        if self._logger is not None:
            self._logger(f"[Audio WARNING] {message}")

    def start(self, motion: SMPLMotion | None) -> bool:
        """Start ffplay for one music motion without blocking the send loop."""
        if self.mode == "off" or motion is None:
            return False
        metadata = motion.metadata
        if metadata.get("source") != "music_only":
            return False
        audio_value = metadata.get("audio_path")
        if not isinstance(audio_value, str) or not audio_value:
            self._warn("music motion has no audio_path metadata")
            return False
        audio_path = Path(audio_value).expanduser()
        if not audio_path.is_file():
            self._warn(f"source audio does not exist: {audio_path}")
            return False
        ffplay = self._which("ffplay")
        if ffplay is None:
            if not self._warned_missing:
                self._warn("ffplay is unavailable; motion streaming continues without audio")
                self._warned_missing = True
            return False
        try:
            start_sec = float(metadata.get("audio_start_sec", 0.0)) + self.offset_sec
            duration_sec = float(metadata.get("audio_duration_sec", motion.duration))
        except (TypeError, ValueError):
            self._warn("audio timing metadata is invalid")
            return False
        if not math.isfinite(start_sec) or not math.isfinite(duration_sec) or duration_sec <= 0:
            self._warn("audio timing metadata is non-finite or non-positive")
            return False
        start_sec = max(0.0, start_sec)
        command = [
            ffplay,
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.9f}",
            "-t",
            f"{duration_sec:.9f}",
            str(audio_path),
        ]
        try:
            self.process = self._popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            self.process = None
            self._warn(f"ffplay failed to start: {exc}; motion streaming continues")
            return False
        if self._logger is not None:
            self._logger(f"[Audio] Playing {audio_path} from {start_sec:.3f}s")
        return True

    def stop(self, reason: str = "state change") -> None:
        """Terminate a running ffplay child without failing motion streaming."""
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.2)
        except (OSError, ProcessLookupError, subprocess.SubprocessError) as exc:
            self._warn(f"unable to stop ffplay cleanly ({reason}): {exc}")

    def update(self, state: PlayerState, motion: SMPLMotion | None) -> None:
        """Start only on entry to PLAYING and stop on every safety transition."""
        if state == PlayerState.PLAYING and self._last_state != PlayerState.PLAYING:
            self.start(motion)
        elif state != PlayerState.PLAYING and self.process is not None:
            self.stop(state.value)
        self._last_state = state

    def close(self) -> None:
        """Release the optional subprocess during normal or exceptional exit."""
        self.stop("program exit")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface without creating runtime resources."""
    parser = argparse.ArgumentParser(
        description="Persistent SMPL-X motion player for GMR-CPP (SMP1 UDP)"
    )
    parser.add_argument("--watch_dir", type=Path)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--gmr_host", default="127.0.0.1")
    parser.add_argument("--gmr_port", type=int, default=7006)
    parser.add_argument("--publish_fps", type=float, default=30.0)
    parser.add_argument("--shape_mode", choices=["zero"], default="zero")
    parser.add_argument("--idle_motion", type=Path)
    parser.add_argument("--mode", choices=["sim", "robot"], default="sim")
    parser.add_argument("--poll_interval", type=float, default=0.2)
    parser.add_argument("--blend_seconds", type=float, default=0.8)
    parser.add_argument("--return_seconds", type=float, default=1.0)
    parser.add_argument("--estop_blend_seconds", type=float, default=0.3)
    parser.add_argument("--estop_file", type=Path, default=Path("/tmp/genmo_estop"))
    parser.add_argument(
        "--new_motion_policy",
        choices=["queue", "latest", "interrupt"],
        default="queue",
    )
    parser.add_argument("--allow_interrupt_in_robot", action="store_true")
    parser.add_argument("--reset_origin_on_motion", action="store_true")
    parser.add_argument("--replay_existing", action="store_true")
    parser.add_argument(
        "--source_filter",
        choices=["any", "text_only", "music_only"],
        default="any",
        help="Only enqueue READY directories whose metadata source matches.",
    )
    parser.add_argument(
        "--audio_playback",
        choices=["off", "ffplay"],
        default="off",
        help="Best-effort source audio playback for music_only motions.",
    )
    parser.add_argument(
        "--audio_offset_sec",
        type=float,
        default=0.0,
        help="Offset added to the source-audio seek time for best-effort sync.",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--dry_run_fk",
        action="store_true",
        help="Also load SMPL-X and run FK for dry-run sample frames.",
    )
    parser.add_argument("--smplx_yaw_deg", type=float, default=0.0)
    parser.add_argument("--gmr_scale", type=float, default=1.0)
    parser.add_argument("--max_send_errors", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate streamer arguments."""
    args = build_parser().parse_args(argv)
    if args.motion is None and args.watch_dir is None:
        raise ValueError("At least one of --motion or --watch_dir must be provided")
    for name in (
        "publish_fps",
        "poll_interval",
        "blend_seconds",
        "return_seconds",
        "estop_blend_seconds",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name} must be finite and > 0")
    if not 1 <= args.gmr_port <= 65535:
        raise ValueError("--gmr_port must be in [1, 65535]")
    if args.max_send_errors <= 0:
        raise ValueError("--max_send_errors must be > 0")
    if not math.isfinite(args.audio_offset_sec):
        raise ValueError("--audio_offset_sec must be finite")
    if args.once and args.loop:
        raise ValueError("--once and --loop cannot be used together")
    if args.mode == "robot" and args.idle_motion is None:
        raise RuntimeError("Robot mode requires a verified idle SMPL-X motion file.")
    if (
        args.mode == "robot"
        and args.new_motion_policy == "interrupt"
        and not args.allow_interrupt_in_robot
    ):
        raise RuntimeError(
            "Robot mode forbids --new_motion_policy interrupt unless "
            "--allow_interrupt_in_robot is explicitly set."
        )
    return args


def load_endecoder(device: torch.device) -> torch.nn.Module:
    """Load the same SMPL-X FK implementation and configuration as Webcam."""
    from gem.network.endecoder import EnDecoder

    model = EnDecoder(
        stats_name="MM_V1_AMASS_LOCAL_BEDLAM_CAM",
        encode_type="gvhmr",
        feat_dim=151,
        clip_std=True,
    )
    model.build_obs_indices_dict()
    return model.eval().to(device)


def load_idle(args: argparse.Namespace) -> SMPLMotion:
    """Load a verified idle reference or create a simulation-only neutral pose."""
    if args.idle_motion is not None:
        return load_smpl_motion(args.idle_motion, shape_mode="zero", min_frames=1)
    if args.mode == "robot":
        raise RuntimeError("Robot mode requires a verified idle SMPL-X motion file.")
    print(
        "WARNING: using synthetic idle pose; do not use this pose on a real robot\n"
        "without validation."
    )
    idle = synthetic_idle_motion(args.publish_fps)
    print("[Idle] Synthetic simulation pose: standing with both arms down")
    return idle


@torch.inference_mode()
def fk_and_adapt_frame(
    frame: SMPLFrame,
    endecoder: torch.nn.Module,
    adapter: SMPLXGMRReference,
    *,
    device: torch.device,
    frame_id: int,
    timestamp_ns: int,
) -> Any:
    """Run zero-shape SMPL-X FK and convert the result to GMR target semantics."""
    body_pose_fk = frame.body_pose.to(device).reshape(1, 1, 63)
    global_orient_fk = frame.global_orient.to(device).reshape(1, 1, 3)
    transl_fk = frame.transl.to(device).reshape(1, 1, 3)
    # Do not trust or forward source betas: robot FK always uses neutral shape.
    betas_fk = torch.zeros(1, 1, 10, device=device, dtype=body_pose_fk.dtype)
    if torch.count_nonzero(betas_fk).item() != 0:
        raise AssertionError("shape_mode=zero FK betas are not zero")
    for name, tensor in {
        "body_pose": body_pose_fk,
        "global_orient": global_orient_fk,
        "transl": transl_fk,
        "betas": betas_fk,
    }.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf before FK")

    joints, _, fk_mat = endecoder.fk_v2(
        body_pose=body_pose_fk,
        betas=betas_fk,
        global_orient=global_orient_fk,
        transl=transl_fk,
        get_intermediate=True,
    )
    if not torch.isfinite(joints).all() or not torch.isfinite(fk_mat).all():
        raise ValueError("SMPL-X FK output contains NaN or Inf")
    return adapter.adapt(
        joints[0, 0, :22],
        fk_mat[0, 0, :22, :3, :3],
        frame_id=frame_id,
        timestamp_ns=timestamp_ns,
    )


def send_frame_to_gmr(
    frame: SMPLFrame,
    endecoder: torch.nn.Module,
    adapter: SMPLXGMRReference,
    bridge: GMRUDPBridge,
    *,
    device: torch.device,
    timestamp_ns: int,
) -> bytes:
    """Run FK/coordinate adaptation and send exactly one existing SMP1 packet."""
    adapted = fk_and_adapt_frame(
        frame,
        endecoder,
        adapter,
        device=device,
        frame_id=bridge.sequence,
        timestamp_ns=timestamp_ns,
    )
    return bridge.send_smplx_targets(
        adapted.scaled_targets,
        source_stamp_ns=timestamp_ns,
    )


def _frame_summary(label: str, frame: SMPLFrame) -> None:
    print(
        f"{label}: root={frame.transl.tolist()} "
        f"orient_norm={torch.linalg.vector_norm(frame.global_orient).item():.6f} "
        f"pose_norm={torch.linalg.vector_norm(frame.body_pose).item():.6f}"
    )


def run_dry_run(args: argparse.Namespace) -> int:
    """Validate and resample one motion without constructing a UDP bridge."""
    if args.motion is None:
        raise ValueError("--dry_run requires --motion")
    motion = load_smpl_motion(args.motion, shape_mode="zero")
    samples = max(2, int(math.ceil(motion.duration * args.publish_fps)))
    frames = [
        sample_motion_at(motion, min(i / args.publish_fps, motion.duration))[0]
        for i in range(samples)
    ]
    finite = all(
        torch.isfinite(value).all().item() for frame in frames for value in vars(frame).values()
    )
    print(f"Motion frames: {motion.num_frames}")
    print(f"Motion fps: {motion.fps:g}")
    print(f"Duration: {motion.duration:.3f}s")
    print(f"Publish fps: {args.publish_fps:g}")
    print(f"Shape mode: {args.shape_mode}")
    print(f"Betas norm: {torch.linalg.vector_norm(motion.betas).item():.6f}")
    print(f"Initial root: {motion.transl[0].tolist()}")
    print(f"Final root: {motion.transl[-1].tolist()}")
    print(f"All finite: {finite}")
    _frame_summary("First publish frame", frames[0])
    _frame_summary("Middle publish frame", frames[len(frames) // 2])
    _frame_summary("Last publish frame", frames[-1])
    if not finite or torch.count_nonzero(motion.betas).item() != 0:
        raise RuntimeError("Dry-run validation failed")

    if args.dry_run_fk:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        endecoder = load_endecoder(device)
        adapter = SMPLXGMRReference(
            user_yaw_deg=args.smplx_yaw_deg,
            global_scale=args.gmr_scale,
        )
        for index in (0, len(frames) // 2, len(frames) - 1):
            fk_and_adapt_frame(
                frames[index],
                endecoder,
                adapter,
                device=device,
                frame_id=index,
                timestamp_ns=index,
            )
        print("SMPL-X FK: finite for first/middle/last publish frames")
    print("UDP socket: not created (dry-run)")
    return 0


def _banner(args: argparse.Namespace) -> None:
    idle = str(args.idle_motion) if args.idle_motion else "synthetic (simulation only)"
    print("=" * 60)
    print("GENMO SMPL-X Motion Streamer")
    print(f"  Mode:          {args.mode}")
    print(f"  Watch dir:     {args.watch_dir or '-'}")
    print(f"  Direct motion: {args.motion or '-'}")
    print(f"  GMR:           {args.gmr_host}:{args.gmr_port}")
    print(f"  Publish FPS:   {args.publish_fps:g}")
    print(f"  Shape mode:    {args.shape_mode}")
    print(f"  Idle motion:   {idle}")
    print(f"  New policy:    {args.new_motion_policy}")
    print(f"  Source filter: {args.source_filter}")
    print(f"  Audio:         {args.audio_playback} (offset={args.audio_offset_sec:g}s)")
    print("=" * 60)


def _load_future_result(
    future: Future[SMPLMotion],
    source_dir: Path,
    watcher: MotionWatcher | None,
) -> SMPLMotion | None:
    try:
        motion = future.result()
    except Exception as exc:
        print(f"[Load ERROR] {source_dir}: {type(exc).__name__}: {exc}")
        return None
    _log_loaded_motion(motion)
    if watcher is not None:
        watcher.mark_consumed(source_dir)
    return motion


def _log_loaded_motion(motion: SMPLMotion) -> None:
    """Log common and optional music provenance without requiring metadata fields."""
    source = motion.metadata.get("source", "unknown")
    print(f"[Load] source={source}")
    if source == "music_only":
        print(f"[Load] audio={motion.metadata.get('audio_path', '-')}")
        print(f"[Load] bpm={motion.metadata.get('estimated_bpm', '-')}")
    print(f"[Load] motion={motion.num_frames} frames @ {motion.fps:g} FPS")


def run_stream(args: argparse.Namespace) -> int:
    """Run the persistent fixed-rate streamer until interrupted or --once completes."""
    _banner(args)
    idle = load_idle(args)
    direct_motion = (
        load_smpl_motion(args.motion, shape_mode="zero") if args.motion is not None else None
    )
    if direct_motion is not None:
        _log_loaded_motion(direct_motion)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Loading SMPL-X FK on {device}")
    endecoder = load_endecoder(device)
    adapter = SMPLXGMRReference(
        user_yaw_deg=args.smplx_yaw_deg,
        global_scale=args.gmr_scale,
    )
    bridge = GMRUDPBridge(args.gmr_host, args.gmr_port, debug=args.verbose)
    audio_controller = FFplayAudioController(
        args.audio_playback,
        offset_sec=args.audio_offset_sec,
    )
    player = MotionPlayer(
        idle,
        blend_seconds=args.blend_seconds,
        return_seconds=args.return_seconds,
        estop_blend_seconds=args.estop_blend_seconds,
        loop=args.loop,
    )
    now = time.monotonic()
    player.start(now)

    watcher = (
        MotionWatcher(
            args.watch_dir,
            replay_existing=args.replay_existing,
            source_filter=args.source_filter,
        )
        if args.watch_dir is not None
        else None
    )
    pending_paths: deque[Path] = deque()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="smpl-loader")
    load_future: Future[SMPLMotion] | None = None
    load_source: Path | None = None
    last_poll = -math.inf
    deadline = MonotonicDeadline(args.publish_fps, now)
    stats_started = now
    stats_sends = 0
    consecutive_errors = 0
    observed_motion_start = player.motion_started_count

    if direct_motion is not None:
        player.enqueue(direct_motion, policy=args.new_motion_policy, now=now)
        print(f"[Queue] Added motion: {direct_motion.source_path}")

    try:
        while True:
            now = time.monotonic()
            delay = deadline.seconds_until(now)
            if delay > 0.0:
                time.sleep(delay)
                now = time.monotonic()
            skipped = deadline.advance(now)
            if skipped and args.verbose:
                print(f"[Stream] skipped {skipped} stale deadline(s)")

            if watcher is not None and now - last_poll >= args.poll_interval:
                new_paths = watcher.scan()
                for path in new_paths:
                    print(f"[Watch] New motion detected: {path}")
                    pending_paths.append(path)
                last_poll = now

            if load_future is None and pending_paths:
                load_source = pending_paths.popleft()
                player.begin_loading()
                load_future = executor.submit(
                    load_smpl_motion,
                    load_source / "smpl_params.pt",
                    shape_mode="zero",
                )
            if load_future is not None and load_future.done():
                assert load_source is not None
                motion = _load_future_result(load_future, load_source, watcher)
                load_future = None
                load_source = None
                player.finish_loading()
                if motion is not None:
                    # Deleting the ESTOP file alone does not resume old work.  A
                    # successfully loaded new action is the explicit reset event.
                    if player.estop_latched and not args.estop_file.exists():
                        player.reset_estop(now)
                    if not player.estop_latched:
                        player.enqueue(motion, policy=args.new_motion_policy, now=now)
                        print(f"[Queue] Added motion: {motion.source_path}")
                    else:
                        print("[ESTOP] New action ignored while ESTOP file exists")

            frame = player.tick(now, estop=args.estop_file.exists())
            if player.motion_started_count != observed_motion_start:
                observed_motion_start = player.motion_started_count
                if args.reset_origin_on_motion:
                    adapter.reset()

            stamp_ns = time.monotonic_ns()
            try:
                send_frame_to_gmr(
                    frame,
                    endecoder,
                    adapter,
                    bridge,
                    device=device,
                    timestamp_ns=stamp_ns,
                )
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                print(
                    f"[Stream ERROR] {type(exc).__name__}: {exc} "
                    f"({consecutive_errors}/{args.max_send_errors})"
                )
                if consecutive_errors >= args.max_send_errors and player.state not in {
                    PlayerState.ERROR,
                    PlayerState.ESTOP,
                }:
                    player.enter_error(now)
                    consecutive_errors = 0
            audio_controller.update(player.state, player.active_motion)
            stats_sends += 1

            elapsed_stats = now - stats_started
            if elapsed_stats >= 5.0:
                active_frames = player.active_motion.num_frames if player.active_motion else 0
                print(
                    f"[Stream] send={stats_sends / elapsed_stats:.1f}Hz "
                    f"state={player.state.value} "
                    f"frame={int(player.frame_float)}/{active_frames} "
                    f"queue={len(player.queue)}"
                )
                stats_started = now
                stats_sends = 0

            if (
                args.once
                and player.motion_started_count >= 1
                and player.completed_count >= 1
                and player.state == PlayerState.HOLDING
            ):
                print("[Done] One motion completed and returned to safe idle")
                return 0
    except KeyboardInterrupt:
        print("\n[Interrupted] closing streamer")
        return 0
    finally:
        audio_controller.close()
        bridge.close()
        executor.shutdown(wait=False, cancel_futures=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    if args.dry_run:
        return run_dry_run(args)
    return run_stream(args)


if __name__ == "__main__":
    raise SystemExit(main())
