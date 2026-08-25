#!/usr/bin/env python3
"""从 BUMI ONNX/TensorRT 音乐模型生成长轨迹并安全发布到 GMT。

该入口完成部署全链路：WAV→30 Hz EDGE35→120/30 独立 DDIM+几何感知 overlap-add→
连续世界系 qpos28→
带 CRC/revision/模型指纹的 BUMI 安全流→跨分块物理边界检查→30→50 Hz SLERP→GMT
110×55 ``trajectory_v1`` 滚动窗口→Redis→GMT ACK。BUMI 已直接输出机器人关节轨迹，
所以此链路明确绕过 SMPL 和 GMR，避免对机器人动作重复重定向。

默认是只生成、检查并保存计划的 dry-run，不会写 Redis。实际机器人执行必须同时传入
``--execute`` 和 ``--confirm-robot-motion``，并在运动开始前收到匹配 stream/revision/plan
的 ACK；执行中 ACK 超时/陈旧、紧急停止文件、安全门失败或 Redis 异常都会终止发布。
输出 NPZ 同时保存 30 Hz 模型原始 qpos、足底锁定 qpos、contact logits、50 Hz 播放 qpos
和 GMT frames，便于确认后处理只改 root XY 而没有掩盖根旋转或关节错误。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.robots.bumi.endecoder import BumiEndecoder  # noqa: E402
from gem.robots.bumi.feature_codec import (  # noqa: E402
    BUMI_REPRESENTATION_CONTRACT_VERSION,
)
from gem.runtime.bumi_gmt_plan import BumiIncrementalGmtPlanBuilder  # noqa: E402
from gem.runtime.bumi_music_deploy import (  # noqa: E402
    BUMI_SLIDING_QPOS_CONTRACT_VERSION,
    BumiOrtStepRunner,
    BumiSlidingQposGenerator,
    BumiTensorRTStepRunner,
)
from gem.runtime.bumi_music_onnx import BUMI_ONNX_CONTRACT_VERSION  # noqa: E402
from gem.runtime.bumi_robot_stream import (  # noqa: E402
    BUMI_QPOS_STREAM_CONTRACT,
    BumiQposChunk,
    BumiQposRevisionTracker,
    BumiQposSafetyGate,
    bumi_joint_order_sha256,
)
from gem.runtime.gmt_trajectory import (  # noqa: E402
    FLAG_AUDIO,
    FLAG_FIXED_IDLE,
    FLAG_TRANSITION,
    GmtPolicyContract,
    RedisTrajectoryPublisher,
)
from gem.runtime.motion_streamer import MonotonicDeadline  # noqa: E402
from gem.runtime.music_only_trt import (  # noqa: E402
    exact_motion_frame_count,
    sha256_file,
)
from gem.utils.music_features import extract_edge_baseline35  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--onnx-metadata", type=Path)
    parser.add_argument("--backend", choices=("onnx", "tensorrt"), default="onnx")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--onnx-provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--gmt-policy", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument(
        "--no-foot-lock",
        action="store_true",
        help="诊断时保留原始 root XY；实际机器人默认启用 contact-gated FK 足底锁定",
    )
    parser.add_argument("--blend-seconds", type=float, default=0.8)
    parser.add_argument("--return-seconds", type=float, default=1.0)
    parser.add_argument("--joint-limit-tolerance-rad", type=float, default=0.05)
    parser.add_argument("--max-joint-velocity-radps", type=float, default=18.0)
    parser.add_argument("--max-root-linear-velocity-mps", type=float, default=4.0)
    parser.add_argument("--max-root-angular-velocity-radps", type=float, default=8.0)
    parser.add_argument("--min-root-height-m", type=float, default=0.25)
    parser.add_argument("--max-root-height-m", type=float, default=1.20)
    parser.add_argument("--output", type=Path, default=Path("outputs/bumi_onnx_gmt/plan.npz"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-robot-motion", action="store_true")
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--redis-key", default="gmt_online_frame_bumi")
    parser.add_argument("--redis-ttl-ms", type=int, default=250)
    parser.add_argument("--revision", type=int, default=1)
    parser.add_argument("--request-id")
    parser.add_argument("--ack-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--ack-stale-seconds", type=float, default=1.0)
    parser.add_argument("--estop-file", type=Path, default=Path("/tmp/genmo_estop"))
    parser.add_argument("--audio-playback", choices=("off", "ffplay"), default="off")
    return parser


def _validate_onnx_identity(
    metadata_path: Path,
    *,
    checkpoint: Path,
    kinematics: Path,
    stats: Path,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("contract_version") != BUMI_ONNX_CONTRACT_VERSION:
        raise ValueError("ONNX metadata is not the BUMI guided denoiser contract")
    if metadata.get("representation_contract_version") != BUMI_REPRESENTATION_CONTRACT_VERSION:
        raise ValueError("ONNX metadata is not the current BUMI qpos30 representation")
    if int(metadata.get("sequence_length", -1)) != 120:
        raise ValueError("BUMI GMT runtime requires a fixed 120-frame ONNX export")
    expected = {
        "checkpoint": (metadata.get("checkpoint") or {}).get("sha256"),
        "kinematics": (metadata.get("kinematics") or {}).get("sha256"),
        "stats": (metadata.get("stats") or {}).get("sha256"),
    }
    actual = {
        "checkpoint": sha256_file(checkpoint),
        "kinematics": sha256_file(kinematics),
        "stats": sha256_file(stats),
    }
    for name in expected:
        if expected[name] != actual[name]:
            raise ValueError(
                f"BUMI ONNX {name} identity mismatch: metadata={expected[name]}, actual={actual[name]}"
            )
    return metadata


def _start_audio(
    path: Path, start_sec: float, duration_sec: float
) -> subprocess.Popen[bytes] | None:
    ffplay = shutil.which("ffplay")
    if ffplay is None:
        raise RuntimeError("--audio-playback=ffplay requires ffplay")
    return subprocess.Popen(
        [
            ffplay,
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.9f}",
            "-t",
            f"{duration_sec:.9f}",
            str(path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _execute_plan(
    args: argparse.Namespace,
    *,
    snapshot: Any,
    contract: GmtPolicyContract,
) -> dict[str, Any]:
    import redis

    if args.estop_file.expanduser().exists():
        raise RuntimeError(f"emergency stop is active: {args.estop_file}")
    client = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        socket_timeout=0.2,
        socket_connect_timeout=1.0,
    )
    client.ping()
    publisher = RedisTrajectoryPublisher(client, key=args.redis_key, ttl_ms=args.redis_ttl_ms)
    cursor = 0
    last_ack_sequence = -1
    submitted = time.monotonic()
    last_ack = submitted
    acked = False
    audio: subprocess.Popen[bytes] | None = None
    audio_started = False
    ticks = 0
    skipped_total = 0
    deadline = MonotonicDeadline(50.0, time.monotonic())
    try:
        while cursor < len(snapshot.frames) - 100:
            now = time.monotonic()
            delay = deadline.seconds_until(now)
            if delay > 0.0:
                time.sleep(delay)
                now = time.monotonic()
            skipped = deadline.advance(now)
            skipped_total += skipped
            if args.estop_file.expanduser().exists():
                raise RuntimeError("ESTOP became active during BUMI GMT playback")
            flags = (
                FLAG_AUDIO
                if snapshot.audio_start_frame <= cursor < snapshot.audio_end_frame
                else FLAG_TRANSITION
            )
            publisher.publish(
                snapshot.frames,
                cursor,
                fps=50.0,
                joint_order_hash=contract.joint_order_hash,
                command_revision=args.revision,
                plan_id=args.revision,
                flags=flags,
            )
            ack = publisher.matching_ack()
            after_publish = time.monotonic()
            if ack is not None and ack.sequence > last_ack_sequence:
                last_ack_sequence = ack.sequence
                last_ack = after_publish
                acked = True
            if not acked:
                if after_publish - submitted > args.ack_timeout_seconds:
                    raise RuntimeError("GMT ACK timeout before motion start")
            elif after_publish - last_ack > args.ack_stale_seconds:
                raise RuntimeError("GMT ACK became stale during motion")
            else:
                if cursor >= snapshot.audio_start_frame and not audio_started:
                    if args.audio_playback == "ffplay":
                        audio = _start_audio(
                            args.audio.expanduser().resolve(strict=True),
                            args.start_sec,
                            snapshot.audio_end_frame / 50.0 - snapshot.audio_start_frame / 50.0,
                        )
                    audio_started = True
                cursor = min(cursor + 1 + skipped, len(snapshot.frames) - 100)
            ticks += 1
        assert snapshot.terminal_idle_frames is not None
        for _ in range(50):
            publisher.publish(
                snapshot.terminal_idle_frames,
                0,
                fps=50.0,
                joint_order_hash=contract.joint_order_hash,
                command_revision=args.revision,
                plan_id=args.revision,
                flags=FLAG_FIXED_IDLE,
            )
            time.sleep(0.02)
    finally:
        if audio is not None and audio.poll() is None:
            audio.terminate()
    return {
        "redis_key": args.redis_key,
        "stream_id": publisher.stream_id,
        "published_ticks": ticks,
        "skipped_deadlines": skipped_total,
        "last_ack_sequence": last_ack_sequence,
        "completed": cursor >= len(snapshot.frames) - 100,
    }


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute != args.confirm_robot_motion:
        raise ValueError("physical execution requires both --execute and --confirm-robot-motion")
    if args.backend == "tensorrt" and args.engine is None:
        raise ValueError("--backend=tensorrt requires --engine")
    if not 2 <= args.ddim_steps <= 1000:
        raise ValueError("--ddim-steps must be in 2..1000")
    if args.start_sec < 0.0 or not math.isfinite(args.start_sec):
        raise ValueError("--start-sec must be finite and >= 0")
    if args.duration_sec is not None and (
        args.duration_sec <= 0.0 or not math.isfinite(args.duration_sec)
    ):
        raise ValueError("--duration-sec must be finite and > 0")
    if args.redis_ttl_ms <= 40:
        raise ValueError("--redis-ttl-ms must exceed two 50 Hz periods")
    if args.revision < 0:
        raise ValueError("--revision must be >= 0")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    audio_path = args.audio.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    onnx_path = args.onnx.expanduser().resolve(strict=True)
    kinematics_path = args.kinematics.expanduser().resolve(strict=True)
    stats_path = args.stats.expanduser().resolve(strict=True)
    metadata_path = (
        args.onnx_metadata.expanduser().resolve(strict=True)
        if args.onnx_metadata is not None
        else onnx_path.with_suffix(onnx_path.suffix + ".json").resolve(strict=True)
    )
    metadata = _validate_onnx_identity(
        metadata_path,
        checkpoint=checkpoint,
        kinematics=kinematics_path,
        stats=stats_path,
    )
    features, feature_metadata = extract_edge_baseline35(
        audio_path,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        target_fps=30,
    )
    total_frames = exact_motion_frame_count(len(features), args.duration_sec, fps=30)
    features = features[:total_frames].contiguous()
    endecoder = BumiEndecoder(
        kinematics_path=kinematics_path,
        stats_path=stats_path,
        enable_contact_targets=False,
    ).to(device)
    engine_path = None
    if args.backend == "onnx":
        step = BumiOrtStepRunner(onnx_path, device=device, provider=args.onnx_provider)
        engine_sha = sha256_file(onnx_path)
    else:
        engine_path = args.engine.expanduser().resolve(strict=True)
        step = BumiTensorRTStepRunner(engine_path, device=device)
        if step.manifest is None:
            raise RuntimeError("BUMI physical TensorRT runtime requires engine.json")
        if step.manifest.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise ValueError("TensorRT manifest checkpoint SHA does not match --checkpoint")
        if step.manifest.get("onnx_sha256") != sha256_file(onnx_path):
            raise ValueError("TensorRT manifest ONNX SHA does not match --onnx")
        engine_sha = sha256_file(engine_path)
    generated = BumiSlidingQposGenerator(
        step,
        endecoder,
        device=device,
        steps=args.ddim_steps,
        guidance_scale=args.cfg_scale,
        apply_foot_lock=not args.no_foot_lock,
    ).generate(features, seed=args.seed)

    contract = GmtPolicyContract.from_onnx(args.gmt_policy)
    native_names = endecoder.kinematics.joint_order
    native_to_gmt = contract.native_to_gmt_indices(native_names)
    idle_qpos = endecoder.kinematics.default_qpos.detach().cpu().numpy().copy()
    idle_qpos[7:] = contract.default_in_native_order(native_names)
    builder = BumiIncrementalGmtPlanBuilder(
        idle_qpos,
        native_to_gmt,
        blend_seconds=args.blend_seconds,
        return_seconds=args.return_seconds,
    )
    safety = BumiQposSafetyGate(
        endecoder.kinematics,
        joint_limit_tolerance_rad=args.joint_limit_tolerance_rad,
        max_joint_velocity_radps=args.max_joint_velocity_radps,
        max_root_linear_velocity_mps=args.max_root_linear_velocity_mps,
        max_root_angular_velocity_radps=args.max_root_angular_velocity_radps,
        min_root_height_m=args.min_root_height_m,
        max_root_height_m=args.max_root_height_m,
    )
    request_id = args.request_id or str(uuid.uuid4())
    tracker = BumiQposRevisionTracker()
    tracker.begin(request_id, args.revision, total_frames)
    checkpoint_sha = sha256_file(checkpoint)
    kinematics_sha = endecoder.kinematics.kinematics_sha256
    order_sha = bumi_joint_order_sha256(list(native_names))
    snapshot = None
    wire_bytes = 0
    for index, generated_chunk in enumerate(generated.chunks):
        chunk = BumiQposChunk.from_qpos(
            generated_chunk.qpos.numpy(),
            request_id=request_id,
            revision=args.revision,
            chunk_index=index,
            absolute_start_frame=generated_chunk.absolute_start_frame,
            total_frames=total_frames,
            is_last=generated_chunk.is_last,
            checkpoint_sha256=checkpoint_sha,
            engine_sha256=engine_sha,
            kinematics_sha256=kinematics_sha,
            joint_order_sha256=order_sha,
        )
        parts = chunk.multipart()
        decoded = BumiQposChunk.from_multipart(parts)
        tracker.accept(decoded)
        safe_qpos = safety.validate(decoded.qpos())
        snapshot = builder.append(safe_qpos, is_last=decoded.is_last)
        wire_bytes += sum(len(part) for part in parts)
    if snapshot is None or not snapshot.action_complete:
        raise RuntimeError("BUMI qpos stream did not produce a complete GMT plan")

    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("--output must use the .npz suffix")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        qpos_30hz=generated.qpos.numpy().astype(np.float32),
        qpos_raw_30hz=generated.qpos_raw.numpy().astype(np.float32),
        foot_contact_logits_30hz=generated.foot_contact_logits.numpy().astype(np.float32),
        foot_lock_correction_xy_30hz=generated.foot_lock_correction_xy.numpy().astype(np.float32),
        foot_lock_active_contact_30hz=generated.foot_lock_active_contact.numpy(),
        qpos_50hz=snapshot.qpos.astype(np.float32),
        gmt_frames_50hz=snapshot.frames.astype(np.float32),
        native_to_gmt=native_to_gmt.astype(np.int64),
    )
    execution = None
    if args.execute:
        execution = _execute_plan(args, snapshot=snapshot, contract=contract)
    report = {
        "contract_version": "genmo.bumi_onnx_gmt_demo.qpos30_contact.v3",
        "mode": "execute" if args.execute else "dry_run",
        "source_stream_contract": BUMI_QPOS_STREAM_CONTRACT,
        "sliding_qpos_contract_version": BUMI_SLIDING_QPOS_CONTRACT_VERSION,
        "request_id": request_id,
        "revision": args.revision,
        "backend": args.backend,
        "audio": str(audio_path),
        "feature_metadata": feature_metadata,
        "frames_30hz": total_frames,
        "frames_50hz_with_transitions": len(snapshot.qpos),
        "audio_start_frame_50hz": snapshot.audio_start_frame,
        "audio_end_frame_50hz": snapshot.audio_end_frame,
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "onnx": {"path": str(onnx_path), "sha256": sha256_file(onnx_path)},
        "onnx_metadata": str(metadata_path),
        "engine": None if engine_path is None else {"path": str(engine_path), "sha256": engine_sha},
        "kinematics": {"path": str(kinematics_path), "sha256": kinematics_sha},
        "stats": {"path": str(stats_path), "sha256": sha256_file(stats_path)},
        "gmt_policy": str(contract.path),
        "qpos_chunks": len(generated.chunks),
        "foot_lock_applied": not args.no_foot_lock,
        "foot_lock_contract_version": generated.foot_lock_contract_version,
        "foot_lock_max_abs_correction_m": float(generated.foot_lock_correction_xy.abs().max()),
        "wire_bytes": wire_bytes,
        "output": str(output),
        "execution": execution,
        "safety": {
            "joint_limit_tolerance_rad": args.joint_limit_tolerance_rad,
            "max_joint_velocity_radps": args.max_joint_velocity_radps,
            "max_root_linear_velocity_mps": args.max_root_linear_velocity_mps,
            "max_root_angular_velocity_radps": args.max_root_angular_velocity_radps,
            "root_height_m": [args.min_root_height_m, args.max_root_height_m],
        },
        "identity_verified": bool(metadata),
    }
    report_path = output.with_suffix(output.suffix + ".json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
