#!/usr/bin/env python3
"""把 GENMO 的 SMPL-X 动作离线重定向并导出为 SONIC/Isaac-Lab BUMI3 NPZ。

本脚本用于替代旧的 Redis 在线捕获导出流程。输入是 GENMO 标准的
``smpl_params.pt``，程序先在 SMPL-X 参数层把 30 Hz 动作重采样到 50 Hz：根节点
平移使用线性插值，根节点朝向和 21 个 body 旋转使用 SO(3) 最短路径 SLERP。随后
每一个 50 Hz 采样帧都会依次经过真实 SMPL-X FK、SMP1 编码和 GMR-CPP 的同步
``smplx_bumi3_batch_server`` 求解；全程不启动 Redis、不依赖墙钟播放，也不会把
已有的 30 Hz qpos 或 ``gmr_stream_raw.npz`` 仅做格式转换。

GMR 返回的 ``qpos[T,28]`` 是 MuJoCo 原生顺序。脚本读取 GMR BUMI3 preset，把
其中 21 个关节重排为 Isaac-Lab/SONIC 使用的顺序，再用与当前 GMR BUMI3 MJCF
数值一致的 OMG kinematics 计算 22 个刚体的世界系位置和 wxyz 四元数。所有速度
均在最终 50 Hz 时间轴上求世界系数值导数。输出 NPZ 严格只含七个 float32 字段：
``fps``、``joint_pos``、``joint_vel``、``body_pos_w``、``body_quat_w``、
``body_lin_vel_w`` 和 ``body_ang_vel_w``；复现信息保存为同名
``.metadata.json``，不会成为部署 NPZ 的额外键。

典型用法：

.. code-block:: bash

   python scripts/export_smplx_to_bumi3_offline_npz.py \
     --motion outputs/example/smpl_params.pt \
     --output outputs/example_bumi3_50hz.npz \
     --hz 50 \
     --ik-config /home/weili/GMR-CPP_e1jump_lowdpi/config/ik_configs/smplx_to_bumi3_auto.json

``--hz`` 被严格锁定为 50。同步 batch server 自身没有实时 ``--hz`` 发布参数，频率
由送入求解器的采样时间轴决定；因此这里的含义是“实际调用 GMR 求解 T50 次”，而
不是在 GMR 之后把 30 Hz 结果补点到 50 Hz。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GMR_ROOT = Path("/home/weili/GMR-CPP_e1jump_lowdpi")
DEFAULT_OMG_ROOT = Path("/home/weili/OMG")
DEPLOYMENT_FPS = 50.0
NPZ_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.gmr_udp_bridge import SMP1PacketEncoder  # noqa: E402
from gem.runtime.motion_streamer import SMPLMotion, load_smpl_motion  # noqa: E402
from gem.runtime.robot_stream import GMRBatchClient  # noqa: E402
from gem.smplx_gmr_reference import SMPLXGMRReference  # noqa: E402
from scripts.demo.stream_smpl_params_to_gmr import load_endecoder  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构造严格的离线 50 Hz 导出命令行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True, help="GENMO smpl_params.pt")
    parser.add_argument("--output", type=Path, required=True, help="输出 .npz 文件")
    parser.add_argument(
        "--hz",
        type=float,
        default=DEPLOYMENT_FPS,
        help="GMR 输入和部署输出频率；BUMI3 SONIC 固定为 50",
    )
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--ik-config", type=Path)
    parser.add_argument("--robot-xml", type=Path)
    parser.add_argument("--batch-server", type=Path)
    parser.add_argument("--omg-root", type=Path, default=DEFAULT_OMG_ROOT)
    parser.add_argument("--kinematics", type=Path)
    parser.add_argument("--ground-clearance", type=float, default=0.04)
    parser.add_argument("--reset-iterations", type=int, default=1000)
    parser.add_argument("--fk-chunk-frames", type=int, default=256)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="SMPL-X FK 设备；机器人 FK 固定在 CPU，便于确定性导出",
    )
    return parser


def sha256(path: Path) -> str:
    """流式计算文件哈希，写入旁路复现信息。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    """解析并验证 GMR、IK、MJCF、preset 与机器人 FK 资产。"""
    if not math.isclose(float(args.hz), DEPLOYMENT_FPS, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("BUMI3 SONIC 离线 NPZ 要求 --hz 必须是 50，不能使用 30")
    if not math.isfinite(args.ground_clearance) or args.ground_clearance < 0.0:
        raise ValueError("--ground-clearance 必须是有限且非负的数")
    if args.reset_iterations <= 0:
        raise ValueError("--reset-iterations 必须大于 0")
    if args.fk_chunk_frames <= 0:
        raise ValueError("--fk-chunk-frames 必须大于 0")

    gmr_root = args.gmr_root.expanduser().resolve(strict=True)
    omg_root = args.omg_root.expanduser().resolve(strict=True)
    motion = args.motion.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("--output 必须使用 .npz 扩展名")
    paths = {
        "motion": motion,
        "output": output,
        "gmr_root": gmr_root,
        "batch_server": (
            args.batch_server
            if args.batch_server is not None
            else gmr_root / "build-genmo-stream/smplx_bumi3_batch_server"
        )
        .expanduser()
        .resolve(strict=True),
        "ik": (
            args.ik_config
            if args.ik_config is not None
            else gmr_root / "config/ik_configs/smplx_to_bumi3_auto.json"
        )
        .expanduser()
        .resolve(strict=True),
        "xml": (
            args.robot_xml
            if args.robot_xml is not None
            else gmr_root / "assets/bumi3/mjcf/bumi3.xml"
        )
        .expanduser()
        .resolve(strict=True),
        "preset": (gmr_root / "config/robot_presets/bumi3.json").resolve(strict=True),
        "omg_root": omg_root,
        "kinematics": (
            args.kinematics
            if args.kinematics is not None
            else omg_root / "assets/robots/bumi/bumi_kinematics.json"
        )
        .expanduser()
        .resolve(strict=True),
    }
    if not os.access(paths["batch_server"], os.X_OK):
        raise PermissionError(f"GMR batch server 不可执行: {paths['batch_server']}")
    return paths


def _slerp_axis_angle(
    rotations_axis_angle: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """对多条 axis-angle 旋转轨迹逐关节执行最短路径 SLERP。"""
    if rotations_axis_angle.ndim != 3 or rotations_axis_angle.shape[-1] != 3:
        raise ValueError(f"旋转应为 [T,J,3]，实际为 {rotations_axis_angle.shape}")
    output = np.empty(
        (len(target_times), rotations_axis_angle.shape[1], 3), dtype=np.float32
    )
    for joint_index in range(rotations_axis_angle.shape[1]):
        rotations = Rotation.from_rotvec(rotations_axis_angle[:, joint_index])
        output[:, joint_index] = Slerp(source_times, rotations)(target_times).as_rotvec()
    return output


def resample_smpl_motion(motion: SMPLMotion, target_fps: float) -> dict[str, torch.Tensor]:
    """在进入 GMR 之前把完整 SMPL-X 参数轨迹重采样到目标频率。"""
    if motion.num_frames < 2:
        raise ValueError("离线重采样至少需要两个 SMPL-X 帧")
    source_times = np.arange(motion.num_frames, dtype=np.float64) / float(motion.fps)
    duration = float(source_times[-1])
    target_count = int(math.floor(duration * target_fps)) + 1
    target_times = np.arange(target_count, dtype=np.float64) / float(target_fps)
    target_times = np.clip(target_times, 0.0, duration)

    source_transl = motion.transl.numpy().astype(np.float64, copy=False)
    target_transl = np.stack(
        [np.interp(target_times, source_times, source_transl[:, axis]) for axis in range(3)],
        axis=1,
    ).astype(np.float32)
    source_rotations = np.concatenate(
        (
            motion.global_orient.numpy().reshape(motion.num_frames, 1, 3),
            motion.body_pose.numpy().reshape(motion.num_frames, 21, 3),
        ),
        axis=1,
    ).astype(np.float64, copy=False)
    target_rotations = _slerp_axis_angle(source_rotations, source_times, target_times)
    result = {
        "body_pose": torch.from_numpy(target_rotations[:, 1:].reshape(target_count, 63)),
        "global_orient": torch.from_numpy(target_rotations[:, 0]),
        "transl": torch.from_numpy(target_transl),
    }
    for name, value in result.items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise RuntimeError(f"50 Hz SMPL-X 字段 {name} 的类型或有限值校验失败")
    return result


def _device_from_arg(value: str) -> torch.device:
    """按照显式选项或 CUDA 可用性选择 SMPL-X FK 设备。"""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda 已指定，但当前环境没有可用 CUDA")
    return torch.device(value)


@torch.inference_mode()
def run_offline_gmr(
    smpl_50hz: dict[str, torch.Tensor],
    *,
    paths: dict[str, Path],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, list[int]]:
    """逐个 50 Hz 帧调用无 Redis 的 GMR batch server，返回原生 qpos。"""
    endecoder = load_endecoder(device)
    adapter = SMPLXGMRReference(user_yaw_deg=0.0, global_scale=1.0)
    encoder = SMP1PacketEncoder(debug=False)
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
    total_frames = int(smpl_50hz["body_pose"].shape[0])
    qpos = np.empty((total_frames, 28), dtype=np.float32)
    elapsed_us: list[int] = []
    client = GMRBatchClient(command, cwd=paths["gmr_root"])
    try:
        for start in range(0, total_frames, args.fk_chunk_frames):
            end = min(start + args.fk_chunk_frames, total_frames)
            body_pose = smpl_50hz["body_pose"][start:end].to(device).unsqueeze(0)
            global_orient = smpl_50hz["global_orient"][start:end].to(device).unsqueeze(0)
            transl = smpl_50hz["transl"][start:end].to(device).unsqueeze(0)
            betas = torch.zeros(
                1, end - start, 10, device=device, dtype=body_pose.dtype
            )
            joints, _, fk_mat = endecoder.fk_v2(
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
                timestamp_ns = int(round(frame_index / args.hz * 1e9))
                adapted = adapter.adapt(
                    joints[0, local_index, :22],
                    fk_mat[0, local_index, :22, :3, :3],
                    frame_id=frame_index,
                    timestamp_ns=timestamp_ns,
                )
                packet = encoder.pack_smplx_targets(
                    adapted.scaled_targets, source_stamp_ns=timestamp_ns
                )
                if frame_index == 0:
                    frame_qpos, solve_us = client.reset(
                        packet, iterations=args.reset_iterations
                    )
                else:
                    frame_qpos, solve_us = client.frame(packet)
                qpos[frame_index] = frame_qpos
                elapsed_us.append(solve_us)
            print(f"[离线 GMR] 已完成 {end}/{total_frames} 个 50 Hz 帧", flush=True)
    finally:
        client.close()
    if not np.isfinite(qpos).all():
        raise RuntimeError("GMR qpos 含 NaN/Inf")
    quaternion_norms = np.linalg.norm(qpos[:, 3:7], axis=1)
    if not np.allclose(quaternion_norms, 1.0, atol=1e-4):
        raise RuntimeError("GMR 根节点四元数不是单位四元数")
    return qpos, elapsed_us


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    """用中心差分求速度，首尾分别使用前向和后向差分。"""
    array = np.asarray(values, dtype=np.float32)
    if array.shape[0] == 0:
        raise ValueError("速度输入时间轴不能为空")
    output = np.zeros_like(array, dtype=np.float32)
    if len(array) == 1:
        return output
    output[0] = (array[1] - array[0]) * np.float32(fps)
    output[-1] = (array[-1] - array[-2]) * np.float32(fps)
    if len(array) > 2:
        output[1:-1] = (array[2:] - array[:-2]) * np.float32(fps / 2.0)
    return output


def _world_rotvec_between(first_wxyz: np.ndarray, second_wxyz: np.ndarray) -> np.ndarray:
    """返回 first 到 second 的世界系最短相对旋转向量。"""
    first_xyzw = np.asarray(first_wxyz, dtype=np.float64)[..., (1, 2, 3, 0)]
    second_xyzw = np.asarray(second_wxyz, dtype=np.float64)[..., (1, 2, 3, 0)]
    first_rotation = Rotation.from_quat(first_xyzw.reshape(-1, 4))
    second_rotation = Rotation.from_quat(second_xyzw.reshape(-1, 4))
    rotvec = (second_rotation * first_rotation.inv()).as_rotvec()
    return rotvec.reshape(first_wxyz.shape[:-1] + (3,)).astype(np.float32)


def body_angular_velocity_world(quaternions_wxyz: np.ndarray, fps: float) -> np.ndarray:
    """由 wxyz 刚体朝向计算与 Isaac-Lab ``body_ang_vel_w`` 同义的世界系角速度。"""
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float32)
    output = np.zeros(quaternions.shape[:-1] + (3,), dtype=np.float32)
    if len(quaternions) == 1:
        return output
    output[0] = _world_rotvec_between(quaternions[0], quaternions[1]) * np.float32(fps)
    output[-1] = _world_rotvec_between(quaternions[-2], quaternions[-1]) * np.float32(fps)
    if len(quaternions) > 2:
        output[1:-1] = _world_rotvec_between(
            quaternions[:-2], quaternions[2:]
        ) * np.float32(fps / 2.0)
    return output


def build_deployment_arrays(
    qpos_native: np.ndarray,
    *,
    paths: dict[str, Path],
    fps: float,
) -> tuple[dict[str, np.ndarray], tuple[str, ...], tuple[str, ...]]:
    """把原生 qpos 转换为 Isaac-Lab 关节顺序和完整 22-body 世界状态。"""
    omg_src = paths["omg_root"] / "src"
    if str(omg_src) not in sys.path:
        sys.path.insert(0, str(omg_src))
    from omg.robots.generic_kinematics import GenericKinematics

    preset: dict[str, Any] = json.loads(paths["preset"].read_text(encoding="utf-8"))
    native_names = tuple(str(value) for value in preset["joint_names_mujoco_qpos_order"])
    isaac_names = tuple(str(value) for value in preset["joint_names_publish_order"])
    native_indices = np.asarray(preset["joint_ids_map"], dtype=np.int64)
    if len(native_names) != 21 or len(isaac_names) != 21:
        raise RuntimeError("BUMI3 preset 的原生/Isaac-Lab 关节数量不是 21")
    if tuple(native_names[index] for index in native_indices) != isaac_names:
        raise RuntimeError("BUMI3 joint_ids_map 与关节名称顺序不一致")

    kinematics = GenericKinematics(paths["kinematics"]).eval().cpu()
    if tuple(kinematics.joint_order) != native_names:
        raise RuntimeError("OMG kinematics 与 GMR MuJoCo 原生关节顺序不一致")
    body_names = tuple(kinematics.body_order)
    if len(body_names) != 22:
        raise RuntimeError(f"BUMI3 kinematics body 数量应为 22，实际为 {len(body_names)}")
    with torch.inference_mode():
        fk = kinematics.forward_kinematics(torch.from_numpy(qpos_native))
    body_pos_w = fk["body_pos_w"].cpu().numpy().astype(np.float32, copy=False)
    body_quat_w = fk["body_quat_w"].cpu().numpy().astype(np.float32, copy=False)
    joint_pos = qpos_native[:, 7 + native_indices].astype(np.float32, copy=False)
    arrays = {
        "fps": np.asarray([fps], dtype=np.float32),
        "joint_pos": joint_pos,
        "joint_vel": finite_difference(joint_pos, fps),
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": finite_difference(body_pos_w, fps),
        "body_ang_vel_w": body_angular_velocity_world(body_quat_w, fps),
    }
    return arrays, isaac_names, body_names


def validate_arrays(arrays: dict[str, np.ndarray]) -> int:
    """执行部署契约的键、dtype、帧率、形状、有限值和四元数校验。"""
    if tuple(arrays) != NPZ_KEYS:
        raise RuntimeError(f"NPZ 键顺序/集合错误: {tuple(arrays)}")
    frames = int(arrays["joint_pos"].shape[0])
    expected_shapes = {
        "fps": (1,),
        "joint_pos": (frames, 21),
        "joint_vel": (frames, 21),
        "body_pos_w": (frames, 22, 3),
        "body_quat_w": (frames, 22, 4),
        "body_lin_vel_w": (frames, 22, 3),
        "body_ang_vel_w": (frames, 22, 3),
    }
    for name in NPZ_KEYS:
        value = arrays[name]
        if value.shape != expected_shapes[name]:
            raise RuntimeError(
                f"{name} shape 应为 {expected_shapes[name]}，实际为 {value.shape}"
            )
        if value.dtype != np.float32:
            raise RuntimeError(f"{name} dtype 应为 float32，实际为 {value.dtype}")
        if not np.isfinite(value).all():
            raise RuntimeError(f"{name} 含 NaN/Inf")
    if arrays["fps"].tobytes() != np.asarray([50.0], dtype=np.float32).tobytes():
        raise RuntimeError("fps 必须精确保存为 float32 [50.0]")
    quaternion_norms = np.linalg.norm(arrays["body_quat_w"], axis=-1)
    if not np.allclose(quaternion_norms, 1.0, atol=1e-4):
        raise RuntimeError("body_quat_w 含非单位四元数")
    return frames


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """先写同目录临时文件，通过校验后原子替换最终 NPZ。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        with np.load(temporary, allow_pickle=False) as archive:
            loaded = {name: archive[name] for name in archive.files}
        validate_arrays(loaded)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    """执行 50 Hz SMPL-X 重采样、离线 GMR、机器人 FK 与严格 NPZ 保存。"""
    args = build_parser().parse_args(argv)
    paths = resolve_paths(args)
    started = time.monotonic()
    motion = load_smpl_motion(paths["motion"], shape_mode="zero", min_frames=2)
    smpl_50hz = resample_smpl_motion(motion, args.hz)
    device = _device_from_arg(args.device)
    print(
        f"[离线导出] {motion.num_frames} 帧 @ {motion.fps:g} Hz -> "
        f"{len(smpl_50hz['body_pose'])} 帧 @ {args.hz:g} Hz；设备={device}",
        flush=True,
    )
    qpos_native, solve_times_us = run_offline_gmr(
        smpl_50hz, paths=paths, args=args, device=device
    )
    arrays, joint_names, body_names = build_deployment_arrays(
        qpos_native, paths=paths, fps=args.hz
    )
    frames = validate_arrays(arrays)
    atomic_save_npz(paths["output"], arrays)

    batch_command = [
        str(paths["batch_server"]),
        "--xml",
        str(paths["xml"]),
        "--ik-config",
        str(paths["ik"]),
        "--ground-clearance",
        f"{args.ground_clearance:.9g}",
        "--offset-to-ground",
    ]
    metadata = {
        "format": "sonic_isaaclab_bumi3_motion_v1",
        "source_motion": str(paths["motion"]),
        "source_motion_sha256": sha256(paths["motion"]),
        "source_frames": motion.num_frames,
        "source_fps": float(motion.fps),
        "output_npz": str(paths["output"]),
        "output_frames": frames,
        "fps": float(args.hz),
        "fps_dtype": "float32",
        "gmr_mode": "synchronous_batch_no_redis",
        "redis_enabled": False,
        "gmr_input_frames": frames,
        "gmr_input_hz": float(args.hz),
        "gmr_command": batch_command,
        "gmr_batch_server_sha256": sha256(paths["batch_server"]),
        "gmr_xml": str(paths["xml"]),
        "gmr_xml_sha256": sha256(paths["xml"]),
        "gmr_ik_config": str(paths["ik"]),
        "gmr_ik_config_sha256": sha256(paths["ik"]),
        "ground_clearance_m": float(args.ground_clearance),
        "reset_iterations": int(args.reset_iterations),
        "joint_names_isaaclab_order": list(joint_names),
        "body_names_isaaclab_order": list(body_names),
        "quaternion_order": "wxyz",
        "npz_keys": list(NPZ_KEYS),
        "solver_elapsed_us": {
            "mean": float(np.mean(solve_times_us)),
            "p95": float(np.percentile(solve_times_us, 95)),
            "max": int(max(solve_times_us)),
        },
        "wall_seconds": time.monotonic() - started,
    }
    metadata_path = paths["output"].with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[完成] 部署 NPZ: {paths['output']}")
    print(f"[完成] 复现信息: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
