#!/usr/bin/env python3
"""批量把完整音乐生成为 BUMI MuJoCo 原生 qpos28 NPZ。

该入口专门用于“只生成动作数据、不渲染”的离线批处理：进程只创建一次固定形状
BUMI ONNX Runtime 会话和一次 BUMI 编解码器，随后逐首提取 30 Hz EDGE35 音乐特征，
按 120 帧窗口、30 帧重叠、90 帧步长独立执行 DDIM，并保存 overlap-add、四元数
SLERP、世界系根位移单次积分和可选 FK 足底锁定之后的 qpos28。每个 NPZ 同时保留
未做足锁的 qpos、接触 logits、足锁修正量、关节顺序以及 checkpoint/ONNX/stats/
kinematics 指纹，便于后续审计和复现。

脚本不会导入 GMR、SMPL-X、GMT 或任何渲染模块，也不会修改已有单音乐、在线部署和
网页评测入口。输出采用临时文件校验后原子替换；再次执行时只复用身份、音频和生成参数
全部一致的已完成 NPZ。可选 ``--reuse-root`` 会从现有同身份评测目录复用已经完成的
PyTorch 动作 artifact，避免对同一音乐重复扩散推理，但最终仍统一导出为真实 NumPy NPZ。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from gem.robots.bumi.endecoder import BumiEndecoder  # noqa: E402
from gem.robots.bumi.postprocess import BUMI_FOOT_LOCK_CONTRACT_VERSION  # noqa: E402
from gem.runtime.bumi_music_deploy import (  # noqa: E402
    BUMI_SLIDING_QPOS_CONTRACT_VERSION,
    BumiOrtStepRunner,
    BumiSlidingQposGenerator,
)
from gem.runtime.bumi_music_onnx import BUMI_ONNX_CONTRACT_VERSION  # noqa: E402
from gem.runtime.music_only_trt import (  # noqa: E402
    OVERLAP_FRAMES,
    WINDOW_FRAMES,
    plan_sliding_windows,
)
from gem.utils.music_features import extract_edge_baseline35  # noqa: E402

STEP_FRAMES = WINDOW_FRAMES - OVERLAP_FRAMES
NPZ_CONTRACT_VERSION = "genmo.bumi_music_batch_npz.qpos30_contact.v1"
MANIFEST_CONTRACT_VERSION = "genmo.bumi_music_batch_npz_manifest.v1"
NPZ_KEYS = (
    "contract_version",
    "qpos",
    "qpos_raw",
    "foot_contact_logits",
    "foot_lock_correction_xy",
    "foot_lock_active_contact",
    "joint_names",
    "fps",
    "quaternion_convention",
    "qpos_order",
    "audio_path",
    "audio_sha256",
    "feature_metadata_json",
    "checkpoint_sha256",
    "onnx_sha256",
    "kinematics_sha256",
    "stats_sha256",
    "sliding_qpos_contract_version",
    "foot_lock_contract_version",
    "window_frames",
    "overlap_frames",
    "step_frames",
    "ddim_steps",
    "cfg_scale",
    "seed",
)


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    """流式计算文件 SHA256，避免把大 checkpoint 一次读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """以同目录原子替换方式保存 UTF-8 JSON。"""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def scalar_text(value: Any) -> str:
    """把 NPZ 零维字符串安全转换为 Python 字符串。"""

    return str(np.asarray(value).item())


def dataset_key(path: Path) -> str:
    """根据已知数据目录生成稳定且避免跨库重名的输出前缀。"""

    parts = set(path.parts)
    if "mine_active" in parts:
        return "mine_bumi"
    if "finedance" in parts:
        return "finedance"
    if "aioz_gdance" in parts:
        return "aioz_gdance"
    if "compas3d" in parts:
        return "compas3d"
    if "aistpp" in parts:
        return "aistpp"
    return "audio"


def load_onnx_metadata(path: Path) -> dict[str, Any]:
    """读取并检查 BUMI 双输出 ONNX 的伴随身份元数据。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("contract_version") != BUMI_ONNX_CONTRACT_VERSION:
        raise ValueError(f"ONNX metadata 合约不匹配：{path}")
    return payload


def validate_model_identity(
    *,
    checkpoint: Path,
    onnx: Path,
    metadata_path: Path,
    kinematics: Path,
    stats: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    """绑定 checkpoint、ONNX、kinematics 和 stats，拒绝混用资产。"""

    metadata = load_onnx_metadata(metadata_path)
    identities = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "onnx_sha256": sha256_file(onnx),
        "kinematics_sha256": sha256_file(kinematics),
        "stats_sha256": sha256_file(stats),
    }
    expected = {
        "checkpoint_sha256": (metadata.get("checkpoint") or {}).get("sha256"),
        "kinematics_sha256": (metadata.get("kinematics") or {}).get("sha256"),
        "stats_sha256": (metadata.get("stats") or {}).get("sha256"),
    }
    for key, expected_value in expected.items():
        if expected_value is None or identities[key] != str(expected_value):
            raise ValueError(
                f"模型身份不匹配：{key}, actual={identities[key]}, expected={expected_value}"
            )
    return identities, metadata


def build_reuse_index(
    roots: list[Path],
    *,
    identities: dict[str, str],
    ddim_steps: int,
    cfg_scale: float,
    seed: int,
    apply_foot_lock: bool,
) -> dict[str, Path]:
    """索引同一模型与生成设置下已经完成的正式 `.pt` 动作。"""

    indexed: dict[str, Path] = {}
    for raw_root in roots:
        root = raw_root.expanduser().resolve(strict=True)
        selection_path = root / "selection.json"
        if not selection_path.is_file():
            raise FileNotFoundError(f"复用目录缺少 selection.json：{root}")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        model = selection.get("model") or {}
        generation = selection.get("generation") or {}
        expected_foot_lock = BUMI_FOOT_LOCK_CONTRACT_VERSION if apply_foot_lock else None
        compatible = (
            model.get("checkpoint_sha256") == identities["checkpoint_sha256"]
            and model.get("onnx_sha256") == identities["onnx_sha256"]
            and model.get("kinematics_sha256") == identities["kinematics_sha256"]
            and model.get("stats_sha256") == identities["stats_sha256"]
            and generation.get("ddim_steps") == ddim_steps
            and float(generation.get("cfg_scale", -1.0)) == cfg_scale
            and generation.get("seed") == seed
            and generation.get("max_duration_sec") is None
            and generation.get("foot_lock_postprocess") == expected_foot_lock
            and generation.get("sliding_qpos_contract_version")
            == BUMI_SLIDING_QPOS_CONTRACT_VERSION
        )
        if not compatible:
            raise ValueError(f"复用目录的模型或生成参数不匹配：{root}")
        for item in selection.get("items") or []:
            audio = Path(str(item.get("audio", ""))).expanduser().resolve()
            artifact = root / "artifacts" / str(item.get("dataset")) / (
                str(item.get("audio_key")) + ".pt"
            )
            if audio.is_file() and artifact.is_file():
                indexed[str(audio)] = artifact
    return indexed


def tensors_from_pt(
    path: Path,
    *,
    audio: Path,
    identities: dict[str, str],
    ddim_steps: int,
    cfg_scale: float,
    seed: int,
    apply_foot_lock: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], str | None]:
    """严格校验并转换旧评测目录中可复用的动作 artifact。"""

    try:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")
    expected = {
        "audio": str(audio),
        "checkpoint_sha256": identities["checkpoint_sha256"],
        "onnx_sha256": identities["onnx_sha256"],
        "kinematics_sha256": identities["kinematics_sha256"],
        "sliding_qpos_contract_version": BUMI_SLIDING_QPOS_CONTRACT_VERSION,
        "ddim_steps": ddim_steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "max_duration_sec": None,
    }
    if not isinstance(artifact, dict) or any(
        artifact.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"复用 artifact 身份或参数不一致：{path}")
    foot_lock_version = artifact.get("foot_lock_contract_version")
    if apply_foot_lock != (foot_lock_version == BUMI_FOOT_LOCK_CONTRACT_VERSION):
        raise ValueError(f"复用 artifact 足锁设置不一致：{path}")
    tensors = {
        "qpos": torch.as_tensor(artifact["qpos"]),
        "qpos_raw": torch.as_tensor(artifact["qpos_raw"]),
        "foot_contact_logits": torch.as_tensor(artifact["foot_contact_logits"]),
        "foot_lock_correction_xy": torch.as_tensor(artifact["foot_lock_correction_xy"]),
        "foot_lock_active_contact": torch.as_tensor(artifact["foot_lock_active_contact"]),
    }
    return tensors, dict(artifact.get("feature_metadata") or {}), foot_lock_version


def arrays_for_npz(
    *,
    tensors: dict[str, torch.Tensor],
    joint_names: tuple[str, ...],
    audio: Path,
    audio_sha256: str,
    feature_metadata: dict[str, Any],
    identities: dict[str, str],
    foot_lock_contract_version: str | None,
    ddim_steps: int,
    cfg_scale: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """构造键顺序固定、无需 pickle 即可读取的 NPZ 数组字典。"""

    arrays: dict[str, np.ndarray] = {
        "contract_version": np.asarray(NPZ_CONTRACT_VERSION),
        "qpos": tensors["qpos"].detach().cpu().numpy().astype(np.float32),
        "qpos_raw": tensors["qpos_raw"].detach().cpu().numpy().astype(np.float32),
        "foot_contact_logits": tensors["foot_contact_logits"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "foot_lock_correction_xy": tensors["foot_lock_correction_xy"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32),
        "foot_lock_active_contact": tensors["foot_lock_active_contact"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.bool_),
        "joint_names": np.asarray(joint_names),
        "fps": np.asarray(30.0, dtype=np.float32),
        "quaternion_convention": np.asarray("wxyz"),
        "qpos_order": np.asarray("mujoco_native"),
        "audio_path": np.asarray(str(audio)),
        "audio_sha256": np.asarray(audio_sha256),
        "feature_metadata_json": np.asarray(
            json.dumps(feature_metadata, ensure_ascii=False, sort_keys=True, default=str)
        ),
        "checkpoint_sha256": np.asarray(identities["checkpoint_sha256"]),
        "onnx_sha256": np.asarray(identities["onnx_sha256"]),
        "kinematics_sha256": np.asarray(identities["kinematics_sha256"]),
        "stats_sha256": np.asarray(identities["stats_sha256"]),
        "sliding_qpos_contract_version": np.asarray(BUMI_SLIDING_QPOS_CONTRACT_VERSION),
        "foot_lock_contract_version": np.asarray(foot_lock_contract_version or ""),
        "window_frames": np.asarray(WINDOW_FRAMES, dtype=np.int32),
        "overlap_frames": np.asarray(OVERLAP_FRAMES, dtype=np.int32),
        "step_frames": np.asarray(STEP_FRAMES, dtype=np.int32),
        "ddim_steps": np.asarray(ddim_steps, dtype=np.int32),
        "cfg_scale": np.asarray(cfg_scale, dtype=np.float32),
        "seed": np.asarray(seed, dtype=np.int64),
    }
    return arrays


def validate_npz(
    path: Path,
    *,
    audio: Path,
    audio_sha256: str,
    identities: dict[str, str],
    ddim_steps: int,
    cfg_scale: float,
    seed: int,
    apply_foot_lock: bool,
) -> dict[str, Any]:
    """无 pickle 重读 NPZ，并校验形状、数值、四元数和全部身份字段。"""

    with np.load(path, allow_pickle=False) as payload:
        if tuple(payload.files) != NPZ_KEYS:
            raise ValueError(f"NPZ 键集合或顺序错误：{path}")
        qpos = payload["qpos"]
        qpos_raw = payload["qpos_raw"]
        contacts = payload["foot_contact_logits"]
        correction = payload["foot_lock_correction_xy"]
        active = payload["foot_lock_active_contact"]
        frames = len(qpos)
        expected_shapes = {
            "qpos": (frames, 28),
            "qpos_raw": (frames, 28),
            "foot_contact_logits": (frames, 2),
            "foot_lock_correction_xy": (frames, 2),
            "foot_lock_active_contact": (frames, 2),
        }
        for key, shape in expected_shapes.items():
            if payload[key].shape != shape:
                raise ValueError(f"{key} 形状错误：{payload[key].shape} != {shape}")
        if frames <= 0 or any(
            not np.isfinite(payload[key]).all()
            for key in ("qpos", "qpos_raw", "foot_contact_logits", "foot_lock_correction_xy")
        ):
            raise ValueError(f"NPZ 含空轨迹、NaN 或 Inf：{path}")
        if qpos.dtype != np.float32 or qpos_raw.dtype != np.float32:
            raise ValueError(f"qpos 必须为 float32：{path}")
        quaternion_norm = np.linalg.norm(qpos[:, 3:7], axis=1)
        if float(np.max(np.abs(quaternion_norm - 1.0))) > 1.0e-4:
            raise ValueError(f"根四元数未归一化：{path}")
        expected_text = {
            "contract_version": NPZ_CONTRACT_VERSION,
            "audio_path": str(audio),
            "audio_sha256": audio_sha256,
            "checkpoint_sha256": identities["checkpoint_sha256"],
            "onnx_sha256": identities["onnx_sha256"],
            "kinematics_sha256": identities["kinematics_sha256"],
            "stats_sha256": identities["stats_sha256"],
            "sliding_qpos_contract_version": BUMI_SLIDING_QPOS_CONTRACT_VERSION,
            "foot_lock_contract_version": (
                BUMI_FOOT_LOCK_CONTRACT_VERSION if apply_foot_lock else ""
            ),
        }
        for key, value in expected_text.items():
            if scalar_text(payload[key]) != value:
                raise ValueError(f"NPZ 身份字段不匹配：{path}/{key}")
        if (
            int(payload["ddim_steps"]) != ddim_steps
            or not math.isclose(float(payload["cfg_scale"]), cfg_scale, abs_tol=1.0e-6)
            or int(payload["seed"]) != seed
        ):
            raise ValueError(f"NPZ 生成参数不匹配：{path}")
        return {
            "frames": frames,
            "duration_sec": frames / 30.0,
            "windows": len(plan_sliding_windows(frames)),
            "qpos_shape": list(qpos.shape),
            "contact_shape": list(contacts.shape),
            "foot_lock_max_abs_correction_m": float(np.max(np.abs(correction), initial=0.0)),
            "active_contact_frames": int(np.count_nonzero(np.any(active, axis=1))),
            "quaternion_norm_max_error": float(np.max(np.abs(quaternion_norm - 1.0))),
        }


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """压缩写入临时文件，避免中断时留下伪装成完整结果的 NPZ。"""

    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", action="append", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--onnx-metadata", type=Path)
    parser.add_argument("--kinematics", required=True, type=Path)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reuse-root", action="append", default=[], type=Path)
    parser.add_argument("--onnx-provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-foot-lock", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if not 2 <= args.ddim_steps <= 1000:
        raise ValueError("--ddim-steps 必须在 2..1000")
    if not math.isfinite(args.cfg_scale) or args.cfg_scale < 0.0:
        raise ValueError("--cfg-scale 必须是有限非负数")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA device，但当前 PyTorch 不可用 CUDA")

    paths = {
        name: getattr(args, name).expanduser().resolve(strict=True)
        for name in ("checkpoint", "onnx", "kinematics", "stats")
    }
    metadata_path = (
        args.onnx_metadata.expanduser().resolve(strict=True)
        if args.onnx_metadata is not None
        else paths["onnx"].with_suffix(paths["onnx"].suffix + ".json").resolve(strict=True)
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audios = [path.expanduser().resolve(strict=True) for path in args.audio]
    if len(set(audios)) != len(audios):
        raise ValueError("--audio 列表存在重复路径")
    output_paths: list[Path] = []
    for audio in audios:
        if audio.suffix.lower() != ".wav":
            raise ValueError(f"当前批处理只接受 WAV：{audio}")
        output_paths.append(output_dir / f"{dataset_key(audio)}__{audio.stem}.npz")
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("输出文件名冲突，请检查跨库同名音乐")

    identities, onnx_metadata = validate_model_identity(
        checkpoint=paths["checkpoint"],
        onnx=paths["onnx"],
        metadata_path=metadata_path,
        kinematics=paths["kinematics"],
        stats=paths["stats"],
    )
    apply_foot_lock = not args.no_foot_lock
    reuse_index = build_reuse_index(
        args.reuse_root,
        identities=identities,
        ddim_steps=args.ddim_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        apply_foot_lock=apply_foot_lock,
    )
    device = torch.device(args.device)
    endecoder = (
        BumiEndecoder(
            kinematics_path=paths["kinematics"],
            stats_path=paths["stats"],
            enable_contact_targets=False,
        )
        .to(device)
        .eval()
    )
    runner = BumiOrtStepRunner(paths["onnx"], device=device, provider=args.onnx_provider)
    generator = BumiSlidingQposGenerator(
        runner,
        endecoder,
        device=device,
        steps=args.ddim_steps,
        guidance_scale=args.cfg_scale,
        apply_foot_lock=apply_foot_lock,
    )

    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "contract_version": MANIFEST_CONTRACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "output_dir": str(output_dir),
        "requested_audio_count": len(audios),
        "completed_audio_count": 0,
        "failed_audio_count": 0,
        "model": {
            "checkpoint": {"path": str(paths["checkpoint"]), "sha256": identities["checkpoint_sha256"]},
            "onnx": {"path": str(paths["onnx"]), "sha256": identities["onnx_sha256"]},
            "onnx_metadata": str(metadata_path),
            "checkpoint_global_step": (onnx_metadata.get("checkpoint") or {}).get("global_step"),
            "kinematics": {"path": str(paths["kinematics"]), "sha256": identities["kinematics_sha256"]},
            "stats": {"path": str(paths["stats"]), "sha256": identities["stats_sha256"]},
        },
        "generation": {
            "full_audio": True,
            "fps": 30,
            "window_frames": WINDOW_FRAMES,
            "overlap_frames": OVERLAP_FRAMES,
            "step_frames": STEP_FRAMES,
            "ddim_steps": args.ddim_steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "onnx_provider": args.onnx_provider,
            "device": str(device),
            "foot_lock_applied": apply_foot_lock,
            "foot_lock_contract_version": (
                BUMI_FOOT_LOCK_CONTRACT_VERSION if apply_foot_lock else None
            ),
            "sliding_qpos_contract_version": BUMI_SLIDING_QPOS_CONTRACT_VERSION,
            "rendering": False,
        },
        "items": [],
    }
    atomic_json(manifest_path, manifest)

    total = len(audios)
    for index, (audio, output) in enumerate(zip(audios, output_paths, strict=True), start=1):
        started = time.perf_counter()
        audio_hash = sha256_file(audio)
        item: dict[str, Any] = {
            "index": index,
            "dataset": dataset_key(audio),
            "audio": str(audio),
            "audio_sha256": audio_hash,
            "output": str(output),
        }
        try:
            if output.is_file():
                validation = validate_npz(
                    output,
                    audio=audio,
                    audio_sha256=audio_hash,
                    identities=identities,
                    ddim_steps=args.ddim_steps,
                    cfg_scale=args.cfg_scale,
                    seed=args.seed,
                    apply_foot_lock=apply_foot_lock,
                )
                source = "existing_npz"
            else:
                reusable = reuse_index.get(str(audio))
                if reusable is not None:
                    tensors, feature_metadata, foot_lock_version = tensors_from_pt(
                        reusable,
                        audio=audio,
                        identities=identities,
                        ddim_steps=args.ddim_steps,
                        cfg_scale=args.cfg_scale,
                        seed=args.seed,
                        apply_foot_lock=apply_foot_lock,
                    )
                    source = f"reused_pt:{reusable}"
                else:
                    features, feature_metadata = extract_edge_baseline35(audio, target_fps=30)
                    generated = generator.generate(features, seed=args.seed)
                    tensors = {
                        "qpos": generated.qpos,
                        "qpos_raw": generated.qpos_raw,
                        "foot_contact_logits": generated.foot_contact_logits,
                        "foot_lock_correction_xy": generated.foot_lock_correction_xy,
                        "foot_lock_active_contact": generated.foot_lock_active_contact,
                    }
                    foot_lock_version = generated.foot_lock_contract_version
                    source = "generated"
                arrays = arrays_for_npz(
                    tensors=tensors,
                    joint_names=endecoder.kinematics.joint_order,
                    audio=audio,
                    audio_sha256=audio_hash,
                    feature_metadata=feature_metadata,
                    identities=identities,
                    foot_lock_contract_version=foot_lock_version,
                    ddim_steps=args.ddim_steps,
                    cfg_scale=args.cfg_scale,
                    seed=args.seed,
                )
                atomic_save_npz(output, arrays)
                validation = validate_npz(
                    output,
                    audio=audio,
                    audio_sha256=audio_hash,
                    identities=identities,
                    ddim_steps=args.ddim_steps,
                    cfg_scale=args.cfg_scale,
                    seed=args.seed,
                    apply_foot_lock=apply_foot_lock,
                )
            item.update(
                {
                    "status": "passed",
                    "source": source,
                    "output_sha256": sha256_file(output),
                    **validation,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            print(
                f"[{index:02d}/{total:02d}] 完成 {output.name} "
                f"frames={validation['frames']} source={source}",
                flush=True,
            )
        except Exception as exc:
            item.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            print(f"[{index:02d}/{total:02d}] 失败 {audio}: {item['error']}", flush=True)
        manifest["items"].append(item)
        manifest["completed_audio_count"] = sum(
            row["status"] == "passed" for row in manifest["items"]
        )
        manifest["failed_audio_count"] = sum(
            row["status"] == "failed" for row in manifest["items"]
        )
        atomic_json(manifest_path, manifest)

    manifest["status"] = (
        "complete" if manifest["completed_audio_count"] == len(audios) else "failed"
    )
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["total_frames"] = sum(
        int(row.get("frames", 0)) for row in manifest["items"] if row["status"] == "passed"
    )
    manifest["total_duration_sec"] = manifest["total_frames"] / 30.0
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
