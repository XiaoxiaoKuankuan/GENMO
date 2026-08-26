#!/usr/bin/env python3
"""批量验证 physics_v3 SMPL 音乐模型，并生成 SMPL-X 与 BUMI3 对照视频。

本工具把一次正式视觉验证拆成可独立续跑的三个阶段。``generate`` 阶段读取冻结的
高质量四库曲目清单，逐条调用仓库标准 ``demo_music_only.py``，读取由原始 WAV 冻结的
EDGE35 条件并保存 30 Hz 全局 SMPL-X 参数；该阶段支持按条目编号分片，适合在服务器
1 上用八张 GPU 并行执行。``render`` 阶段在本地加载指定 GMR-CPP 仓库的真实
``smplx_bumi3_batch_server``、IK JSON 与 BUMI3 MJCF，逐帧执行 SMPL-X FK、SMP1 编码
和同步 C++ IK，得到 MuJoCo 原生顺序的 ``qpos[T,28]``。随后分别流式渲染完整
SMPL-X 表面网格和 BUMI3 MuJoCo 机器人，不在内存中堆积整段 RGB，并为两段视频封装
同一个验证音频。程序还生成横向对照视频、每条 sidecar、全局 summary 与 HTML 索引。

所有正式边界均为硬校验：清单数量、音频哈希、checkpoint step、SMPL 字段形状、GMR
根四元数、MJCF 关节顺序以及最终 H.264/AAC/30 FPS/时长不一致时都会失败。生成结果先
写入独立临时目录后再发布；已完成且身份匹配的条目会直接复用，因此服务器中断、网络
回传或本地渲染失败后可以安全重跑同一命令。脚本不会修改 GMR 仓库中的源码、配置、
XML 或构建目录，实际使用的 checkpoint、batch server、IK 与 MJCF SHA256 都会写入结果。
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

mujoco: Any | None = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.gmr_udp_bridge import SMP1PacketEncoder  # noqa: E402
from gem.runtime.motion_streamer import SMPLMotion, load_smpl_motion  # noqa: E402
from gem.runtime.robot_stream import GMRBatchClient  # noqa: E402
from gem.smplx_gmr_reference import SMPLXGMRReference  # noqa: E402
from scripts.build_smpl_hq4_full_comparison import (  # noqa: E402
    load_generated_body,
    mux_audio,
    probe_final_video,
    render_smpl_mesh,
)
from scripts.demo.stream_smpl_params_to_gmr import load_endecoder  # noqa: E402

EXPECTED_COUNTS = {
    "aistpp": 5,
    "aioz_gdance": 20,
    "finedance": 30,
    "compas3d": 5,
}
DEFAULT_OUTPUT = Path("outputs/smpl_physics_v3_s100000_hq4_validation_20260826")
DEFAULT_CHECKPOINT = Path(
    "outputs/gem_smpl_music_only_4set_manual_q1_physics_v3_100k/"
    "version_0/checkpoints/s100000.ckpt"
)
DEFAULT_GMR_ROOT = Path("/home/weili/GMR-CPP_e1jump_lowdpi")
RENDER_CONTRACT = "smplx_bumi3_comparison.v1.camera_distance_2p2"


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    """流式计算大文件 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    """在目标目录原子发布 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    """先完整保存 PyTorch artifact，再原子替换最终文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    """构造服务器生成、本地 GMR 渲染和汇总共用的命令行。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate", "render", "summarize"), required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--exp", default="gem_smpl_music_only_4set_manual_q1_physics_v3_100k"
    )
    parser.add_argument("--checkpoint-step", type=int, default=100000)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--only-id")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--batch-server", type=Path)
    parser.add_argument("--ik-config", type=Path)
    parser.add_argument("--robot-xml", type=Path)
    parser.add_argument("--ground-clearance", type=float, default=0.05)
    parser.add_argument("--reset-iterations", type=int, default=1000)
    parser.add_argument("--fk-chunk-frames", type=int, default=128)
    return parser


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """解析路径并拒绝会改变验证合同的异常参数。"""

    args.output_root = args.output_root.expanduser().resolve()
    args.selection = (
        args.selection.expanduser().resolve()
        if args.selection is not None
        else args.output_root / "selection.json"
    )
    args.ckpt = args.ckpt.expanduser().resolve()
    if not args.selection.is_file():
        raise FileNotFoundError(args.selection)
    if args.stage == "generate" and not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index 必须位于 [0, shard-count)")
    if args.only_id is not None and args.shard_count != 1:
        raise ValueError("--only-id 不能与多分片同时使用")
    if args.ddim_steps < 2 or args.checkpoint_step < 0:
        raise ValueError("DDIM step 至少为 2，checkpoint step 不能为负")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("渲染宽高必须为正")
    if args.reset_iterations <= 0 or args.fk_chunk_frames <= 0:
        raise ValueError("GMR reset iteration 和 FK chunk 必须为正")
    if not np.isfinite(args.ground_clearance) or args.ground_clearance < 0:
        raise ValueError("GMR ground clearance 必须是有限非负数")
    args.output_root.mkdir(parents=True, exist_ok=True)
    return args


def load_selection(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """验证冻结清单、四库数量、ID 和本次 20 秒音频。"""

    payload = json.loads(args.selection.read_text(encoding="utf-8"))
    if payload.get("contract_version") != "genmo.smpl.physics_v3_hq4_visual_validation.v1":
        raise ValueError("selection contract_version 不匹配")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("selection.items 必须为列表")
    counts = Counter(str(item.get("dataset")) for item in items)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"四库数量不符合 {EXPECTED_COUNTS}，实际为 {dict(counts)}")
    identifiers = [str(item.get("id")) for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("selection 出现重复 id")
    for item in items:
        if item.get("quality_decision") != "keep":
            raise ValueError(f"{item.get('id')} 未通过人工高质量 keep 门")
        if int(item.get("validation_frames", -1)) != 600:
            raise ValueError(f"{item.get('id')} 不是固定 600 帧验证")
        audio = validation_audio(args, item)
        if not audio.is_file():
            raise FileNotFoundError(audio)
        expected_hash = str(item.get("validation_audio_sha256"))
        if sha256_file(audio) != expected_hash:
            raise ValueError(f"{item.get('id')} 的 20 秒验证音频 SHA256 不匹配")
        feature = validation_feature(args, item)
        if not feature.is_file():
            raise FileNotFoundError(feature)
        if sha256_file(feature) != str(item.get("validation_music_feature_sha256")):
            raise ValueError(f"{item.get('id')} 的冻结 EDGE35 SHA256 不匹配")
    if args.only_id is not None:
        requested = [item for item in items if item["id"] == args.only_id]
        if not requested:
            raise ValueError(f"selection 中不存在 --only-id={args.only_id}")
    else:
        requested = items[args.shard_index :: args.shard_count]
    return payload, requested


def validation_audio(args: argparse.Namespace, item: dict[str, Any]) -> Path:
    """按输出根目录解析可跨服务器搬运的验证 WAV。"""

    return args.output_root / "audio" / item["dataset"] / f"{item['id']}.wav"


def validation_feature(args: argparse.Namespace, item: dict[str, Any]) -> Path:
    """解析由同一 20 秒 WAV 预先冻结、可跨机器复用的 EDGE35 tensor。"""

    return args.output_root / "music_features" / item["dataset"] / f"{item['id']}.pt"


def artifact_dir(args: argparse.Namespace, item: dict[str, Any]) -> Path:
    return args.output_root / "artifacts" / item["dataset"] / item["id"]


def generation_is_complete(
    args: argparse.Namespace, item: dict[str, Any], directory: Path
) -> bool:
    """只复用 checkpoint、种子、帧数和音频身份均匹配的生成结果。"""

    report_path = directory / "demo_report.json"
    body_path = directory / "pred_body_params_global.pt"
    wrapper_path = directory / "smpl_params.pt"
    sidecar_path = directory / "generation.json"
    if not all(path.is_file() for path in (report_path, body_path, wrapper_path, sidecar_path)):
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        return (
            int(report.get("checkpoint_global_step", -1)) == args.checkpoint_step
            and int(report.get("seed", -1)) == int(item["seed"])
            and report.get("music_shape") == [600, 35]
            and report.get("generated_shape") == [600, 151]
            and sidecar.get("validation_audio_sha256")
            == item["validation_audio_sha256"]
            and sidecar.get("validation_music_feature_sha256")
            == item["validation_music_feature_sha256"]
            and sidecar.get("checkpoint_sha256") == args.checkpoint_sha256
        )
    except (OSError, ValueError, TypeError, RuntimeError):
        return False


def publish_smpl_wrapper(
    directory: Path, item: dict[str, Any], report: dict[str, Any], audio: Path
) -> None:
    """把 demo 的纯 body 字典封装为 GMR 严格加载器使用的 SMPL artifact。"""

    body = load_generated_body(directory / "pred_body_params_global.pt")
    if int(body["body_pose"].shape[0]) != 600:
        raise ValueError(f"{item['id']} 的 SMPL 帧数不是 600")
    payload = {
        "body_params_global": body,
        "fps": 30.0,
        "num_frames": 600,
        "duration_sec": 20.0,
        "source": "music_only_physics_v3_s100000",
        "shape_mode": "zero",
        "audio_path": str(audio),
        "seed": int(item["seed"]),
        "guidance_scale": report.get("cfg_scale"),
        "ddim_steps": report.get("ddim_steps"),
        "metadata": {
            "selection_id": item["id"],
            "dataset": item["dataset"],
            "music_key": item["music_key"],
            "fps": 30.0,
            "checkpoint_global_step": report.get("checkpoint_global_step"),
            "motion_sanity": report.get("motion_sanity"),
        },
    }
    atomic_torch_save(directory / "smpl_params.pt", payload)


def run_generation(args: argparse.Namespace, item: dict[str, Any]) -> None:
    """在隔离 staging 目录运行一次 600 帧 PyTorch DDIM 生成。"""

    final = artifact_dir(args, item)
    if generation_is_complete(args, item, final):
        print(f"[生成] 复用 {item['id']}", flush=True)
        return
    staging_root = args.output_root / "_generation_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{item['id']}-", dir=staging_root))
    audio = validation_audio(args, item)
    feature = validation_feature(args, item)
    log_dir = args.output_root / "logs" / "generation"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{item['id']}.log"
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "scripts/demo_music_only.py"),
        "--music-embed",
        str(feature),
        "--num-frames",
        "600",
        "--max-frames",
        "600",
        "--chunk-frames",
        "600",
        "--chunk-overlap-frames",
        "120",
        "--ckpt",
        str(args.ckpt),
        "--exp",
        args.exp,
        "--output-dir",
        str(staging),
        "--cfg-scale",
        str(args.cfg_scale),
        "--ddim-steps",
        str(args.ddim_steps),
        "--seed",
        str(item["seed"]),
        "--postproc",
        "--no-render",
        "--allow-physically-invalid",
    ]
    started = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"{item['id']} 生成失败 code={result.returncode}，见 {log_path}")
        report_path = staging / "demo_report.json"
        if not report_path.is_file():
            raise RuntimeError(f"{item['id']} 未生成 demo_report.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("checkpoint_global_step", -1)) != args.checkpoint_step:
            raise RuntimeError(f"{item['id']} checkpoint step 不匹配")
        publish_smpl_wrapper(staging, item, report, audio)
        atomic_json(
            staging / "generation.json",
            {
                "selection_id": item["id"],
                "dataset": item["dataset"],
                "music_key": item["music_key"],
                "validation_audio_sha256": item["validation_audio_sha256"],
                "validation_music_feature": str(feature),
                "validation_music_feature_sha256": item[
                    "validation_music_feature_sha256"
                ],
                "checkpoint": str(args.ckpt),
                "checkpoint_sha256": args.checkpoint_sha256,
                "checkpoint_global_step": args.checkpoint_step,
                "experiment": args.exp,
                "seed": item["seed"],
                "ddim_steps": args.ddim_steps,
                "cfg_scale": args.cfg_scale,
                "frames": 600,
                "fps": 30,
                "motion_sanity": report.get("motion_sanity"),
                "final_pass": report.get("final_pass"),
                "wall_seconds": time.monotonic() - started,
            },
        )
        if final.exists():
            backup = final.with_name(f"{final.name}.invalid-{int(time.time())}")
            final.rename(backup)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        print(f"[生成] 完成 {item['id']}，耗时 {time.monotonic() - started:.1f}s", flush=True)
    except BaseException:
        print(f"[生成] 保留失败 staging 供诊断：{staging}", flush=True)
        raise


def resolve_gmr_paths(args: argparse.Namespace) -> dict[str, Path]:
    """绑定用户指定 GMR 仓库中的 batch server、IK、MJCF 与关节 preset。"""

    root = args.gmr_root.expanduser().resolve(strict=True)
    values = {
        "root": root,
        "batch_server": (
            args.batch_server
            if args.batch_server is not None
            else root / "build/smplx_bumi3_batch_server"
        ),
        "ik": (
            args.ik_config
            if args.ik_config is not None
            else root / "config/ik_configs/smplx_to_bumi3_auto.json"
        ),
        "xml": (
            args.robot_xml
            if args.robot_xml is not None
            else root / "assets/bumi3/mjcf/bumi3.xml"
        ),
        "preset": root / "config/robot_presets/bumi3.json",
    }
    for name in ("batch_server", "ik", "xml", "preset"):
        values[name] = values[name].expanduser().resolve(strict=True)
    if not os.access(values["batch_server"], os.X_OK):
        raise PermissionError(f"GMR batch server 不可执行：{values['batch_server']}")
    return values


class OfflineGMRRunner:
    """复用 SMPL-X FK 和 C++ batch 进程，按动作显式 reset 后输出 30 Hz qpos。"""

    def __init__(self, args: argparse.Namespace, paths: dict[str, Path]):
        self.args = args
        self.paths = paths
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("本地 GMR 请求 CUDA，但 torch.cuda.is_available() 为 false")
        self.endecoder = load_endecoder(self.device)
        self.encoder = SMP1PacketEncoder(debug=False)
        command = [
            str(paths["batch_server"]),
            "--xml",
            str(paths["xml"]),
            "--ik-config",
            str(paths["ik"]),
            "--ground-clearance",
            f"{args.ground_clearance:.9g}",
            "--offset-to-ground",
        ]
        self.command = command
        self.client = GMRBatchClient(command, cwd=paths["root"])

    def close(self) -> None:
        self.client.close()

    @torch.inference_mode()
    def run(self, motion: SMPLMotion) -> tuple[np.ndarray, list[int]]:
        """逐帧执行真实 GMR，动作之间同时重置参考原点和 IK 状态。"""

        if not np.isclose(motion.fps, 30.0):
            raise ValueError(f"GMR 可视化输入必须为 30 FPS，实际 {motion.fps}")
        adapter = SMPLXGMRReference(user_yaw_deg=0.0, global_scale=1.0)
        qpos = np.empty((motion.num_frames, 28), dtype=np.float32)
        elapsed_us: list[int] = []
        for start in range(0, motion.num_frames, self.args.fk_chunk_frames):
            end = min(start + self.args.fk_chunk_frames, motion.num_frames)
            body_pose = motion.body_pose[start:end].to(self.device).unsqueeze(0)
            global_orient = motion.global_orient[start:end].to(self.device).unsqueeze(0)
            transl = motion.transl[start:end].to(self.device).unsqueeze(0)
            betas = torch.zeros(1, end - start, 10, device=self.device)
            joints, _, fk_mat = self.endecoder.fk_v2(
                body_pose=body_pose,
                betas=betas,
                global_orient=global_orient,
                transl=transl,
                get_intermediate=True,
            )
            if not torch.isfinite(joints).all() or not torch.isfinite(fk_mat).all():
                raise RuntimeError(f"SMPL-X FK 在 [{start}:{end}) 产生 NaN/Inf")
            for local_index in range(end - start):
                frame_index = start + local_index
                timestamp_ns = int(round(frame_index / motion.fps * 1e9))
                adapted = adapter.adapt(
                    joints[0, local_index, :22],
                    fk_mat[0, local_index, :22, :3, :3],
                    frame_id=frame_index,
                    timestamp_ns=timestamp_ns,
                )
                packet = self.encoder.pack_smplx_targets(
                    adapted.scaled_targets, source_stamp_ns=timestamp_ns
                )
                if frame_index == 0:
                    frame_qpos, solve_us = self.client.reset(
                        packet, iterations=self.args.reset_iterations
                    )
                else:
                    frame_qpos, solve_us = self.client.frame(packet)
                qpos[frame_index] = frame_qpos
                elapsed_us.append(int(solve_us))
        if not np.isfinite(qpos).all():
            raise RuntimeError("GMR qpos 含 NaN/Inf")
        quaternion_error = np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1.0)
        if float(quaternion_error.max()) > 1e-4:
            raise RuntimeError("GMR 根节点 wxyz 四元数不是单位四元数")
        return qpos, elapsed_us


def mujoco_joint_order(model: mujoco.MjModel) -> list[str]:
    """从 MJCF qpos address 读取 21 个原生关节顺序。"""

    ids = [
        joint_id
        for joint_id in range(model.njnt)
        if int(model.jnt_qposadr[joint_id]) >= 7
    ]
    ids.sort(key=lambda value: int(model.jnt_qposadr[value]))
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in ids
    ]
    if len(names) != 21 or any(name is None for name in names):
        raise RuntimeError("BUMI3 MJCF 未提供完整 21 关节原生顺序")
    return [str(name) for name in names]


def load_or_run_gmr(
    args: argparse.Namespace,
    item: dict[str, Any],
    runner: OfflineGMRRunner,
    joint_names: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """复用身份一致的 qpos artifact，否则运行同步 GMR 并原子保存。"""

    directory = artifact_dir(args, item)
    source = directory / "smpl_params.pt"
    output = directory / "bumi_qpos30.pt"
    source_hash = sha256_file(source)
    asset_hashes = {
        "gmr_batch_server_sha256": sha256_file(runner.paths["batch_server"]),
        "gmr_ik_config_sha256": sha256_file(runner.paths["ik"]),
        "gmr_xml_sha256": sha256_file(runner.paths["xml"]),
    }
    if output.is_file():
        payload = torch.load(output, map_location="cpu", weights_only=False)
        if (
            payload.get("source_motion_sha256") == source_hash
            and all(payload.get(key) == value for key, value in asset_hashes.items())
            and payload.get("joint_names") == joint_names
        ):
            return torch.as_tensor(payload["qpos"]).numpy(), payload
    motion = load_smpl_motion(source, shape_mode="zero", min_frames=2)
    qpos, solve_times = runner.run(motion)
    payload = {
        "robot_name": "bumi",
        "quaternion_convention": "wxyz",
        "qpos_order": "mujoco_native",
        "qpos": torch.from_numpy(qpos),
        "fps": 30,
        "joint_names": joint_names,
        "source_motion": str(source),
        "source_motion_sha256": source_hash,
        "gmr_mode": "synchronous_batch_no_redis",
        "gmr_command": runner.command,
        "gmr_batch_server": str(runner.paths["batch_server"]),
        "gmr_ik_config": str(runner.paths["ik"]),
        "gmr_xml": str(runner.paths["xml"]),
        **asset_hashes,
        "ground_clearance_m": args.ground_clearance,
        "reset_iterations": args.reset_iterations,
        "solver_elapsed_us": {
            "mean": float(np.mean(solve_times)),
            "p95": float(np.percentile(solve_times, 95)),
            "max": int(max(solve_times)),
        },
    }
    atomic_torch_save(output, payload)
    return qpos, payload


def start_rawvideo_encoder(output: Path, width: int, height: int) -> subprocess.Popen[bytes]:
    """启动接收 RGB24 标准输入的 H.264 ffmpeg 编码器。"""

    output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            "30",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def render_bumi(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    output: Path,
    width: int,
    height: int,
) -> int:
    """用跟随根节点的 MuJoCo 相机把 BUMI3 逐帧流式写入 ffmpeg。"""

    if qpos.shape != (600, 28):
        raise ValueError(f"BUMI qpos 应为 [600,28]，实际 {qpos.shape}")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.2
    camera.azimuth = 135.0
    camera.elevation = -18.0
    encoder = start_rawvideo_encoder(output, width, height)
    try:
        assert encoder.stdin is not None
        for frame_qpos in qpos:
            data.qpos[:] = frame_qpos
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.xpos[1]
            camera.lookat[2] = max(float(camera.lookat[2]), 0.55)
            renderer.update_scene(data, camera=camera)
            frame = np.asarray(renderer.render(), dtype=np.uint8)
            if frame.shape != (height, width, 3):
                raise RuntimeError(f"MuJoCo 返回异常画面形状：{frame.shape}")
            encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
        encoder.stdin.close()
        assert encoder.stderr is not None
        error = encoder.stderr.read().decode("utf-8", errors="replace").strip()
        returncode = encoder.wait()
        if returncode != 0:
            raise RuntimeError(f"BUMI ffmpeg 编码失败（{returncode}）：{error}")
        return len(qpos)
    except BaseException:
        if encoder.stdin is not None and not encoder.stdin.closed:
            encoder.stdin.close()
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
        output.unlink(missing_ok=True)
        raise
    finally:
        renderer.close()


def make_comparison(smpl: Path, bumi: Path, output: Path) -> None:
    """并排编码 SMPL/BUMI 画面，并复用 SMPL 文件中的同源 AAC 音频。"""

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(smpl),
            "-i",
            str(bumi),
            "-filter_complex",
            "[0:v][1:v]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-map",
            "0:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"SMPL/BUMI 对照视频编码失败：{result.stderr.strip()}")


def render_item(
    args: argparse.Namespace,
    item: dict[str, Any],
    body_model: Any,
    robot_model: mujoco.MjModel,
    runner: OfflineGMRRunner,
    joint_names: list[str],
) -> None:
    """完成单条 GMR、两种渲染、音频封装、对照视频和媒体硬校验。"""

    directory = artifact_dir(args, item)
    if not (directory / "generation.json").is_file():
        raise FileNotFoundError(f"{item['id']} 尚未完成服务器 SMPL 生成")
    video_dir = args.output_root / "videos" / item["dataset"]
    video_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = video_dir / f"{item['id']}.json"
    smpl_output = video_dir / f"{item['id']}_smpl.mp4"
    bumi_output = video_dir / f"{item['id']}_bumi.mp4"
    comparison_output = video_dir / f"{item['id']}_comparison.mp4"
    if sidecar_path.is_file() and all(
        path.is_file() for path in (smpl_output, bumi_output, comparison_output)
    ):
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if (
            sidecar.get("selection_id") == item["id"]
            and sidecar.get("render_contract") == RENDER_CONTRACT
        ):
            print(f"[渲染] 复用 {item['id']}", flush=True)
            return
    started = time.monotonic()
    qpos, gmr_payload = load_or_run_gmr(args, item, runner, joint_names)
    generated_body = load_generated_body(directory / "pred_body_params_global.pt")
    work_root = args.output_root / "_render_work"
    work_root.mkdir(parents=True, exist_ok=True)
    audio = validation_audio(args, item)
    with tempfile.TemporaryDirectory(prefix=f"{item['id']}-", dir=work_root) as temporary:
        work = Path(temporary)
        smpl_silent = work / "smpl_silent.mp4"
        bumi_silent = work / "bumi_silent.mp4"
        smpl_candidate = work / "smpl.mp4"
        bumi_candidate = work / "bumi.mp4"
        comparison_candidate = work / "comparison.mp4"
        render_smpl_mesh(
            body_model,
            generated_body,
            smpl_silent,
            args.width,
            args.height,
            work / "smpl_vertices.f32",
            work / "smpl_normals.f32",
        )
        render_bumi(robot_model, qpos, bumi_silent, args.width, args.height)
        mux_audio(smpl_silent, audio, smpl_candidate, 20.0)
        mux_audio(bumi_silent, audio, bumi_candidate, 20.0)
        make_comparison(smpl_candidate, bumi_candidate, comparison_candidate)
        media = {
            "smpl": probe_final_video(smpl_candidate, 600),
            "bumi": probe_final_video(bumi_candidate, 600),
            "comparison": probe_final_video(comparison_candidate, 600),
        }
        os.replace(smpl_candidate, smpl_output)
        os.replace(bumi_candidate, bumi_output)
        os.replace(comparison_candidate, comparison_output)
    sidecar = {
        "selection_id": item["id"],
        "render_contract": RENDER_CONTRACT,
        "dataset": item["dataset"],
        "music_key": item["music_key"],
        "frames": 600,
        "fps": 30,
        "duration_sec": 20.0,
        "smpl_video": str(smpl_output.relative_to(args.output_root)),
        "bumi_video": str(bumi_output.relative_to(args.output_root)),
        "comparison_video": str(comparison_output.relative_to(args.output_root)),
        "media": media,
        "motion_sanity": json.loads(
            (directory / "generation.json").read_text(encoding="utf-8")
        ).get("motion_sanity"),
        "gmr": {
            key: gmr_payload[key]
            for key in (
                "gmr_mode",
                "gmr_command",
                "gmr_batch_server_sha256",
                "gmr_ik_config_sha256",
                "gmr_xml_sha256",
                "ground_clearance_m",
                "reset_iterations",
                "solver_elapsed_us",
            )
        },
        "wall_seconds": time.monotonic() - started,
    }
    for name, path in (
        ("smpl", smpl_output),
        ("bumi", bumi_output),
        ("comparison", comparison_output),
    ):
        sidecar["media"][name]["sha256"] = sha256_file(path)
    atomic_json(sidecar_path, sidecar)
    print(f"[渲染] 完成 {item['id']}，耗时 {sidecar['wall_seconds']:.1f}s", flush=True)


def write_index(args: argparse.Namespace, items: list[dict[str, Any]]) -> None:
    """生成按数据集分组、默认展示横向对照视频的本地 HTML 索引。"""

    groups: list[str] = []
    for dataset in EXPECTED_COUNTS:
        cards: list[str] = []
        for item in [value for value in items if value["dataset"] == dataset]:
            media = f"videos/{quote(dataset)}/{quote(item['id'])}_comparison.mp4"
            cards.append(
                "<article><h3>"
                + html.escape(f"{item['id']} · {item['music_key']}")
                + "</h3><video controls preload='metadata' src='"
                + media
                + "'></video><p>左：SMPL-X　右：BUMI3　划分："
                + html.escape(str(item.get("split")))
                + "</p></article>"
            )
        groups.append(f"<h2>{html.escape(dataset)}</h2><section>{''.join(cards)}</section>")
    page = """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>physics_v3 s100000：SMPL-X / BUMI3 验证</title>
<style>body{font-family:sans-serif;margin:24px;background:#f5f5f5;color:#222}section{display:grid;
grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}article{background:white;padding:12px;
border-radius:10px;box-shadow:0 1px 5px #bbb}video{width:100%;background:#111}h2{margin-top:32px}
p{color:#555}</style></head><body><h1>physics_v3 s100000：SMPL-X / BUMI3</h1>
<p>每条 20 秒、30 FPS；左侧完整 SMPL-X 网格，右侧真实 GMR-CPP 重定向 BUMI3。</p>"""
    page += "".join(groups) + "</body></html>\n"
    (args.output_root / "index.html").write_text(page, encoding="utf-8")


def summarize(args: argparse.Namespace, selection: dict[str, Any]) -> dict[str, Any]:
    """汇总所有独立 sidecar，并严格要求 60 条 SMPL/BUMI/对照视频齐全。"""

    items = selection["items"]
    completed: list[dict[str, Any]] = []
    missing: list[str] = []
    for item in items:
        path = args.output_root / "videos" / item["dataset"] / f"{item['id']}.json"
        if not path.is_file():
            missing.append(item["id"])
            continue
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        if sidecar.get("selection_id") != item["id"]:
            raise ValueError(f"渲染 sidecar 身份不匹配：{path}")
        completed.append({**item, **sidecar})
    payload = {
        "contract_version": "genmo.smpl.physics_v3_hq4_visual_validation.result.v1",
        "selection": str(args.selection),
        "selection_sha256": sha256_file(args.selection),
        "requested_count": len(items),
        "completed_count": len(completed),
        "missing_ids": missing,
        "dataset_counts": dict(Counter(item["dataset"] for item in completed)),
        "items": completed,
    }
    atomic_json(args.output_root / "summary.json", payload)
    write_index(args, items)
    if missing:
        raise RuntimeError(f"仍缺少 {len(missing)} 条渲染：{missing}")
    if payload["dataset_counts"] != EXPECTED_COUNTS:
        raise RuntimeError(f"最终四库数量异常：{payload['dataset_counts']}")
    return payload


def main(argv: list[str] | None = None) -> int:
    """按指定阶段执行可续跑的 60 条正式验证。"""

    args = resolve_args(build_parser().parse_args(argv))
    selection, requested = load_selection(args)
    if args.stage == "generate":
        args.checkpoint_sha256 = sha256_file(args.ckpt)
        for index, item in enumerate(requested, 1):
            print(f"[{index}/{len(requested)}] {item['id']}", flush=True)
            run_generation(args, item)
        return 0
    if args.stage == "summarize":
        result = summarize(args, selection)
        print(f"[汇总] 完成 {result['completed_count']}/{result['requested_count']}")
        return 0

    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("SMPL-X 网格渲染和 FK 请求 CUDA，但当前不可用")
    global mujoco
    import mujoco as mujoco_module

    mujoco = mujoco_module
    from gem.utils.smplx_utils import make_smplx

    gmr_paths = resolve_gmr_paths(args)
    robot_model = mujoco.MjModel.from_xml_path(str(gmr_paths["xml"]))
    if robot_model.nq != 28:
        raise RuntimeError(f"BUMI3 MJCF nq 应为 28，实际为 {robot_model.nq}")
    joint_names = mujoco_joint_order(robot_model)
    preset = json.loads(gmr_paths["preset"].read_text(encoding="utf-8"))
    if joint_names != list(map(str, preset["joint_names_mujoco_qpos_order"])):
        raise RuntimeError("GMR preset 与当前 MJCF 原生 qpos 关节顺序不一致")
    body_model = make_smplx("supermotion").to(args.device).eval()
    runner = OfflineGMRRunner(args, gmr_paths)
    try:
        for index, item in enumerate(requested, 1):
            print(f"[{index}/{len(requested)}] {item['id']}", flush=True)
            render_item(args, item, body_model, robot_model, runner, joint_names)
    finally:
        runner.close()
    if args.shard_count == 1 and args.only_id is None:
        summarize(args, selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
