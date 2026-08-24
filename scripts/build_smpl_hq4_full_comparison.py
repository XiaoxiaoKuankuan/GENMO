#!/usr/bin/env python3
"""构建四个高质量 SMPLX 音乐舞蹈数据集的完整长度原始/生成动作网页。

本工具面向服务器 1 ``main`` 分支训练出的 GEM-SMPL 模型，不静默截断音频或动作。它从
已经冻结且人工评分为 1 的 AIST++、AIOZ-GDANCE、FineDance、CoMPAS3D 清单中各取
10 条，并使用服务器 1 回传的原始 SMPL 参数。生成端统一调用最新 checkpoint，长序列以
重叠 diffusion 窗口拼接，所有随机种子写入清单以便复现。

为了让数分钟音乐仍能稳定处理，SMPL 原始/生成动作分块计算完整人体表面网格，用 Open3D
逐帧渲染紫色人体和棋盘地面并直接写入 ffmpeg，不在内存中堆积整段 vertices 或 RGB。
最终网页明确区分“原始动作真实片段长度”和“模型覆盖完整音乐的生成长度”，防止跨时长
比较被误解为逐帧复现。脚本会逐条原子更新 manifest/summary/index，并可直接断点续跑。
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import torch

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASET_LABELS = {
    "aistpp": "AIST++",
    "aioz_gdance": "AIOZ-GDANCE",
    "finedance": "FineDance",
    "compas3d": "CoMPAS3D",
}
def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    """流式计算文件 SHA256，避免把 checkpoint/ONNX 整体读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    """在同目录写临时文件后原子替换，防止中断留下半截 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def ffprobe(path: Path) -> dict[str, Any]:
    """读取媒体流和时长元数据。"""

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,codec_type,r_frame_rate:format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def audio_duration(path: Path) -> float:
    """返回音频容器时长。"""

    return float(ffprobe(path)["format"]["duration"])


def probe_final_video(path: Path, expected_frames: int) -> dict[str, Any]:
    """硬校验最终网页视频的 H.264/AAC/30 FPS 和目标时长。"""

    payload = ffprobe(path)
    streams = payload.get("streams", [])
    videos = [value for value in streams if value.get("codec_type") == "video"]
    audios = [value for value in streams if value.get("codec_type") == "audio"]
    if len(videos) != 1 or videos[0].get("codec_name") != "h264":
        raise RuntimeError(f"{path}: 必须且只能包含一条 H.264 视频流")
    if videos[0].get("r_frame_rate") != "30/1":
        raise RuntimeError(f"{path}: 视频帧率不是 30 FPS")
    if len(audios) != 1 or audios[0].get("codec_name") != "aac":
        raise RuntimeError(f"{path}: 必须且只能包含一条 AAC 音频流")
    actual = float(payload["format"]["duration"])
    expected = expected_frames / 30.0
    if abs(actual - expected) > 0.15:
        raise RuntimeError(f"{path}: 时长 {actual:.6f}s 与 {expected:.6f}s 不一致")
    return {
        "duration_sec": actual,
        "size_bytes": int(payload["format"]["size"]),
        "video_codec": "h264",
        "audio_codec": "aac",
        "fps": 30,
    }


def build_selection(
    fourset_selection: Path,
    human_root: Path,
    seed: int,
) -> list[dict[str, Any]]:
    """读取服务器 1 冻结高质量清单，严格得到四库各 10 条。"""

    frozen = json.loads(fourset_selection.read_text(encoding="utf-8"))
    frozen_items = frozen.get("items")
    if not isinstance(frozen_items, list) or len(frozen_items) != 40:
        raise ValueError("冻结四库清单必须恰好包含 40 条")
    normalized_contract = frozen.get("contract_version") == "genmo.smpl_hq4_selection.v1"
    counts = {key: 0 for key in DATASET_LABELS}
    items: list[dict[str, Any]] = []
    for source in frozen_items:
        dataset = str(source["dataset"])
        if dataset not in DATASET_LABELS:
            raise ValueError(f"冻结清单出现异常数据集：{dataset}")
        counts[dataset] += 1
        if normalized_contract:
            if source.get("quality_gate") != "人工评分=1（冻结四库清单）":
                raise ValueError(f"{dataset}/{source.get('sample_id')} 未通过高质量门")
            motion_path = Path(source["original_motion_path"]).expanduser().resolve()
            audio_path = Path(source["audio_path"]).expanduser().resolve()
            audio_key = str(source["audio_key"])
        else:
            if int(source.get("high_quality_motion_count", 0)) < 1:
                raise ValueError(
                    f"{dataset}/{source.get('representative_motion')} 未通过高质量门"
                )
            motion_name = str(source["representative_motion"])
            motion_path = (human_root / dataset / motion_name).resolve()
            audio_path = Path(source["audio"]).expanduser().resolve()
            audio_key = str(source["audio_key"])
        if not motion_path.is_file() or not audio_path.is_file():
            raise FileNotFoundError(f"缺失四库源文件：{motion_path} / {audio_path}")
        items.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "sample_id": motion_path.stem,
                "audio_key": audio_key,
                "audio_path": str(audio_path),
                "original_motion_path": str(motion_path),
                "render_representation": "SMPL-X neutral 完整表面网格（Open3D）",
                "quality_gate": "人工评分=1（冻结四库清单）",
            }
        )

    if counts != {key: 10 for key in DATASET_LABELS}:
        raise ValueError(f"每库必须恰好 10 条，实际 {counts}")
    for number, item in enumerate(items, start=1):
        item["number"] = number
        item["web_id"] = f"{item['dataset']}_{counts_before(items, number):02d}"
        item["seed"] = seed + number - 1
        item["audio_duration_sec"] = audio_duration(Path(item["audio_path"]))
    return items


def counts_before(items: list[dict[str, Any]], number: int) -> int:
    """返回当前条目在所属数据集内的一基序号。"""

    dataset = items[number - 1]["dataset"]
    return sum(value["dataset"] == dataset for value in items[:number])


def load_human_original(path: Path) -> dict[str, torch.Tensor]:
    """把人工高质量 NPZ 转为 GEM 渲染所需的全局 SMPL 参数。"""

    with np.load(path, allow_pickle=False) as source:
        pose = np.ascontiguousarray(source["pose"], dtype=np.float32)
        transl = np.ascontiguousarray(source["transl"], dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] < 66 or transl.shape != (pose.shape[0], 3):
        raise ValueError(f"{path}: pose/transl 形状不符合 SMPL 契约")
    frames = pose.shape[0]
    return {
        "global_orient": torch.from_numpy(pose[:, :3]),
        "body_pose": torch.from_numpy(pose[:, 3:66]),
        "transl": torch.from_numpy(transl),
        "betas": torch.zeros(frames, 10, dtype=torch.float32),
    }


def load_generated_body(path: Path) -> dict[str, torch.Tensor]:
    """硬校验并读取 demo 保存的生成 SMPL 参数。"""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"生成参数不是字典：{path}")
    expected_width = {"global_orient": 3, "body_pose": 63, "transl": 3, "betas": 10}
    result: dict[str, torch.Tensor] = {}
    frames: int | None = None
    for name, width in expected_width.items():
        value = torch.as_tensor(payload.get(name)).detach().cpu().float()
        if value.ndim != 2 or value.shape[1] != width or not torch.isfinite(value).all():
            raise ValueError(f"{path}: {name} 形状/数值异常 {tuple(value.shape)}")
        frames = int(value.shape[0]) if frames is None else frames
        if value.shape[0] != frames:
            raise ValueError(f"{path}: 各参数帧数不一致")
        result[name] = value
    result["betas"] = torch.zeros_like(result["betas"])
    return result


def body_frames(body: dict[str, torch.Tensor]) -> int:
    return int(body["body_pose"].shape[0])


def transform_vertices(
    vertices: np.ndarray,
    offset: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """复现旧 ``normalize_global_verts`` 的平移、落地和首帧朝向归一化。"""

    return (vertices - offset) @ rotation.T + translation


@torch.inference_mode()
def cache_smpl_vertices(
    body_model: Any,
    body: dict[str, torch.Tensor],
    cache_path: Path,
    normal_cache_path: Path,
    chunk: int = 128,
) -> tuple[np.memmap, np.memmap, np.ndarray, np.ndarray, np.ndarray]:
    """分块计算完整 SMPL-X 网格，并返回旧渲染坐标系所需的变换。"""

    from gem.utils.geo_transform import compute_T_ayfz2ay

    frames = body_frames(body)
    cache: np.memmap | None = None
    normal_cache: np.memmap | None = None
    first_joints: np.ndarray | None = None
    minimum_y = float("inf")
    faces = torch.as_tensor(
        np.asarray(body_model.faces), dtype=torch.long, device="cuda"
    )
    for start in range(0, frames, chunk):
        end = min(start + chunk, frames)
        output = body_model(
            body_pose=body["body_pose"][start:end].cuda(non_blocking=True),
            global_orient=body["global_orient"][start:end].cuda(non_blocking=True),
            transl=body["transl"][start:end].cuda(non_blocking=True),
            betas=body["betas"][start:end].cuda(non_blocking=True),
        )
        chunk_vertices = output.vertices.detach().cpu().float().numpy()
        gpu_vertices = output.vertices.detach().float()
        face_normals = torch.cross(
            gpu_vertices[:, faces[:, 1]] - gpu_vertices[:, faces[:, 0]],
            gpu_vertices[:, faces[:, 2]] - gpu_vertices[:, faces[:, 0]],
            dim=-1,
        )
        gpu_normals = torch.zeros_like(gpu_vertices)
        for corner in range(3):
            gpu_normals.index_add_(1, faces[:, corner], face_normals)
        gpu_normals = torch.nn.functional.normalize(gpu_normals, dim=-1)
        chunk_normals = gpu_normals.cpu().numpy()
        if cache is None:
            cache = np.memmap(
                cache_path,
                mode="w+",
                dtype=np.float32,
                shape=(frames, chunk_vertices.shape[1], 3),
            )
            normal_cache = np.memmap(
                normal_cache_path,
                mode="w+",
                dtype=np.float32,
                shape=(frames, chunk_vertices.shape[1], 3),
            )
            first_joints = output.joints[0].detach().cpu().float().numpy()
        cache[start:end] = chunk_vertices
        assert normal_cache is not None
        normal_cache[start:end] = chunk_normals
        minimum_y = min(minimum_y, float(chunk_vertices[..., 1].min()))
        del output, chunk_vertices, chunk_normals, gpu_vertices, gpu_normals, face_normals
    if cache is None or normal_cache is None or first_joints is None:
        raise RuntimeError("SMPL 网格缓存为空")
    cache.flush()
    normal_cache.flush()
    offset = first_joints[0].copy()
    offset[1] = minimum_y
    normalized_joints = torch.from_numpy((first_joints - offset)[None]).float()
    transform = compute_T_ayfz2ay(normalized_joints, inverse=True)[0].cpu().numpy()
    return cache, normal_cache, offset, transform[:3, :3], transform[:3, 3]


def mesh_scene_parameters(
    cache: np.memmap,
    offset: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    chunk: int = 128,
) -> dict[str, np.ndarray | float]:
    """扫描归一化网格，计算与旧渲染器一致的静态相机和棋盘地面范围。"""

    frames = cache.shape[0]
    roots = np.empty((frames, 3), dtype=np.float32)
    vertex_min = np.full(3, np.inf, dtype=np.float64)
    vertex_max = np.full(3, -np.inf, dtype=np.float64)
    for start in range(0, frames, chunk):
        end = min(start + chunk, frames)
        vertices = transform_vertices(
            np.asarray(cache[start:end]), offset, rotation, translation
        )
        roots[start:end] = vertices.mean(axis=1)
        vertex_min = np.minimum(vertex_min, vertices.reshape(-1, 3).min(axis=0))
        vertex_max = np.maximum(vertex_max, vertices.reshape(-1, 3).max(axis=0))
    targets = roots.copy()
    targets[:, 1] = 0.0
    target_center = targets.mean(axis=0)
    target_radius = float(np.linalg.norm(targets - target_center, axis=-1).max())
    camera_scale = max(target_radius, 1.0) * 6.0
    angle = np.deg2rad(45.0)
    camera_position = target_center + np.array(
        [np.sin(angle), 0.0, np.cos(angle)], dtype=np.float32
    ) * camera_scale
    camera_position[1] = camera_scale * np.tan(np.deg2rad(30.0)) + 1.0
    target_center[1] = 1.0
    root_min = roots.min(axis=0)
    root_max = roots.max(axis=0)
    return {
        "camera_position": camera_position.astype(np.float32),
        "target_center": target_center.astype(np.float32),
        "ground_scale": float((vertex_max - vertex_min)[[0, 2]].max()),
        "ground_cx": float((root_min[0] + root_max[0]) / 2.0),
        "ground_cz": float((root_min[2] + root_max[2]) / 2.0),
    }


def render_smpl_mesh(
    body_model: Any,
    body: dict[str, torch.Tensor],
    output: Path,
    width: int,
    height: int,
    cache_path: Path,
    normal_cache_path: Path,
) -> int:
    """按旧 Open3D 风格流式渲染完整 SMPL-X 表面网格和棋盘地面。"""

    import open3d as o3d

    from gem.utils.cam_utils import create_camera_sensor
    from gem.utils.vis.o3d_render import Settings, create_meshes, get_ground

    cache_path.unlink(missing_ok=True)
    normal_cache_path.unlink(missing_ok=True)
    cache: np.memmap | None = None
    normal_cache: np.memmap | None = None
    renderer: Any | None = None
    encoder: subprocess.Popen[bytes] | None = None
    try:
        cache, normal_cache, offset, rotation, translation = cache_smpl_vertices(
            body_model, body, cache_path, normal_cache_path
        )
        scene = mesh_scene_parameters(cache, offset, rotation, translation)
        faces = np.asarray(body_model.faces, dtype=np.int32)
        material_settings = Settings()
        lit_material = material_settings._materials[Settings.LIT]
        _, _, intrinsics = create_camera_sensor(width, height, fov_deg=32)
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        renderer.scene.set_lighting(
            renderer.scene.LightingProfile.NO_SHADOWS,
            np.array([0.577, -0.577, -0.577]),
        )
        renderer.scene.camera.set_projection(
            intrinsics.cpu().double().numpy(), 0.1, 100.0, float(width), float(height)
        )
        renderer.scene.camera.look_at(
            scene["target_center"],
            scene["camera_position"],
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        ground = get_ground(
            max(float(scene["ground_scale"]), 3.0) * 1.5,
            float(scene["ground_cx"]),
            float(scene["ground_cz"]),
        )
        ground_mesh = create_meshes(ground[0], ground[1], ground[2][..., :3])
        ground_material = o3d.visualization.rendering.MaterialRecord()
        ground_material.shader = Settings.LIT
        renderer.scene.add_geometry("mesh_ground", ground_mesh, ground_material)

        output.parent.mkdir(parents=True, exist_ok=True)
        encoder = subprocess.Popen(
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
        body_mesh = o3d.geometry.TriangleMesh()
        body_mesh.triangles = o3d.utility.Vector3iVector(faces)
        body_mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.repeat(
                np.array([[0.69019608, 0.39215686, 0.95686275]], dtype=np.float64),
                cache.shape[1],
                axis=0,
            )
        )
        assert encoder.stdin is not None
        for frame_number in range(cache.shape[0]):
            vertices = transform_vertices(
                np.asarray(cache[frame_number]), offset, rotation, translation
            )
            normals = np.asarray(normal_cache[frame_number]) @ rotation.T
            body_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            body_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
            if frame_number:
                renderer.scene.remove_geometry("mesh_body")
            renderer.scene.add_geometry("mesh_body", body_mesh, lit_material)
            frame = np.asarray(renderer.render_to_image(), dtype=np.uint8)
            if frame.shape != (height, width, 3):
                raise RuntimeError(f"Open3D 返回异常画面形状：{frame.shape}")
            encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
        encoder.stdin.close()
        assert encoder.stderr is not None
        error = encoder.stderr.read().decode("utf-8", errors="replace").strip()
        returncode = encoder.wait()
        if returncode != 0:
            raise RuntimeError(f"SMPL mesh ffmpeg 编码失败（{returncode}）：{error}")
        return int(cache.shape[0])
    except BaseException:
        if encoder is not None:
            if encoder.stdin is not None and not encoder.stdin.closed:
                encoder.stdin.close()
            if encoder.poll() is None:
                encoder.terminate()
                encoder.wait()
        output.unlink(missing_ok=True)
        raise
    finally:
        # Open3D 0.18 的 OffscreenRenderer 没有 close()；释放最后一个 Python
        # 引用即可销毁 Filament/EGL 资源，同时兼容未来可能增加 close() 的版本。
        if renderer is not None and hasattr(renderer, "close"):
            renderer.close()
        renderer = None
        if cache is not None:
            del cache
        if normal_cache is not None:
            del normal_cache
        cache_path.unlink(missing_ok=True)
        normal_cache_path.unlink(missing_ok=True)


def mux_audio(video: Path, audio: Path, output: Path, duration_sec: float) -> None:
    """为无声视频复用 H.264 并编码 AAC；不足一帧的音频尾部以静音补齐。"""

    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-af",
            "apad",
            "-t",
            f"{duration_sec:.9f}",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"音视频封装失败：{result.stderr.strip()}")


def run_generation(item: dict[str, Any], args: argparse.Namespace, artifact_dir: Path) -> dict[str, Any]:
    """调用完整 PyTorch DDIM 长序列入口；已有完整报告时直接复用。"""

    report_path = artifact_dir / "demo_report.json"
    body_path = artifact_dir / "pred_body_params_global.pt"
    if report_path.is_file() and body_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("seed") == item["seed"] and report.get("checkpoint_global_step") == 300000:
            return report
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    log_path = args.output_root / "logs" / f"{item['web_id']}_generation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "scripts/demo_music_only.py"),
        "--audio",
        item["audio_path"],
        "--ckpt",
        str(args.ckpt),
        "--exp",
        args.exp,
        "--output-dir",
        str(artifact_dir),
        "--max-frames",
        str(args.max_frames),
        "--chunk-frames",
        str(args.chunk_frames),
        "--chunk-overlap-frames",
        str(args.chunk_overlap_frames),
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
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = args.device
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0 or not report_path.is_file() or not body_path.is_file():
        raise RuntimeError(f"{item['web_id']} 生成失败（code={process.returncode}），见 {log_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def quoted_media(dataset: str, filename: str) -> str:
    return f"videos/{quote(dataset)}/{quote(filename)}"


def write_page(
    completed: list[dict[str, Any]],
    output_root: Path,
    checkpoint: Path,
    onnx: Path,
    args: argparse.Namespace,
) -> None:
    """生成可筛选、同长度可同步播放的自包含验收网页。"""

    cards: list[str] = []
    for item in completed:
        original_name = f"{item['web_id']}_original.mp4"
        generated_name = f"{item['web_id']}_generated.mp4"
        original_url = quoted_media(item["dataset"], original_name)
        generated_url = quoted_media(item["dataset"], generated_name)
        original_seconds = item["original_frames"] / 30.0
        generated_seconds = item["generated_frames"] / 30.0
        same_length = abs(original_seconds - generated_seconds) <= 0.15
        sanity = item.get("motion_sanity") or {}
        sanity_pass = bool(sanity.get("physical_sanity_pass"))
        badge = "物理粗检通过" if sanity_pass else "需人工复核"
        badge_class = "pass" if sanity_pass else "warn"
        sync_note = "可同步播放" if same_length else "原始片段与完整音乐时长不同，独立播放"
        cards.append(
            f'''<article class="card pair" data-dataset="{html.escape(item['dataset'])}" data-sync="{str(same_length).lower()}">
  <header><div><span class="dataset">{html.escape(item['dataset_label'])}</span><h2>{html.escape(item['sample_id'])}</h2></div><span class="badge {badge_class}">{badge}</span></header>
  <p class="meta">原始 {item['original_frames']:,} 帧 / {original_seconds:.2f}s · 生成 {item['generated_frames']:,} 帧 / {generated_seconds:.2f}s · seed {item['seed']} · {html.escape(sync_note)}</p>
  <p class="contract">质量来源：{html.escape(item['quality_gate'])}；原始与生成均使用 SMPL-X neutral 完整人体表面网格，由 Open3D 渲染紫色人体和棋盘地面；MuJoCo 未参与人体渲染。</p>
  <div class="videos"><section><h3>原始动作（有声）</h3><video class="original" preload="metadata" controls src="{original_url}"></video></section><section><h3>模型生成完整音乐动作（默认静音）</h3><video class="generated" preload="metadata" controls muted src="{generated_url}"></video></section></div>
  <div class="actions"><button data-action="play">{'同步播放' if same_length else '同时从头播放'}</button><button data-action="pause">暂停</button><button data-action="restart">回到开头</button><span class="status">等待播放</span></div>
</article>'''
        )
    options = "".join(
        f'<option value="{key}">{html.escape(label)}</option>' for key, label in DATASET_LABELS.items()
    )
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GENMO 四库完整音乐动作验收</title><style>
:root{{--bg:#0a0d13;--panel:#141923;--line:#2a3342;--text:#eef2f8;--muted:#9da8b8;--blue:#73a9ff;--green:#51d59a;--orange:#ffb86b}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,"Noto Sans SC",sans-serif}}main{{max-width:1500px;margin:auto;padding:26px}}.hero{{background:linear-gradient(135deg,#172544,#141923);border:1px solid var(--line);border-radius:18px;padding:24px}}h1{{margin:0 0 8px}}.hero p,.meta,.contract,.status{{color:var(--muted)}}.toolbar{{position:sticky;top:0;z-index:4;padding:12px 0;background:#0a0d13e8;backdrop-filter:blur(8px)}}button,select{{background:#242c3a;color:var(--text);border:1px solid #3b475c;border-radius:8px;padding:8px 13px;cursor:pointer}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px;margin:17px 0}}header{{display:flex;justify-content:space-between;align-items:center;gap:12px}}h2{{margin:2px 0;font-size:19px}}.dataset{{font-size:12px;color:var(--blue)}}.badge{{padding:4px 9px;border-radius:99px;font-size:12px}}.pass{{background:#143c30;color:var(--green)}}.warn{{background:#4a321b;color:var(--orange)}}.videos{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.videos h3{{font-size:14px;margin:0 0 7px}}video{{display:block;width:100%;aspect-ratio:16/9;background:#030509;border-radius:10px}}.actions{{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-top:12px}}@media(max-width:820px){{main{{padding:12px}}.videos{{grid-template-columns:1fr}}}}
</style></head><body><main><section class="hero"><h1>GENMO 四个高质量 SMPLX 数据集：原始 vs 完整音乐生成</h1><p>共 {len(completed)}/40 条。每库固定 10 首；生成端覆盖完整音频，30 FPS、DDIM {args.ddim_steps}、CFG {args.cfg_scale}。所有原始动作均来自服务器 1 人工评分 1 的正式四库。AIST++ 等原始动作可能只是完整歌曲中的真实短片，因此网页保留真实原始长度，不循环、不拉伸。</p><p>Checkpoint：{html.escape(checkpoint.name)}（服务器 1 main，step 300000）<br>ONNX：{html.escape(onnx.name)}（固定 120 帧单个 CFG 去噪步；完整长序列生成仍由仓库 PyTorch DDIM/后处理完成）</p></section><div class="toolbar"><select id="filter"><option value="all">全部数据集</option>{options}</select> <button id="pauseAll">全部暂停</button></div>{''.join(cards)}</main><script>
const cards=[...document.querySelectorAll('.pair')];const controls=cards.map(card=>{{const a=card.querySelector('.original'),b=card.querySelector('.generated'),s=card.querySelector('.status');b.muted=true;let timer=null;const stop=msg=>{{a.pause();b.pause();if(timer)clearInterval(timer);timer=null;s.textContent=msg}};card.querySelector('[data-action=pause]').onclick=()=>stop('已暂停');card.querySelector('[data-action=restart]').onclick=()=>{{stop('已回到开头');a.currentTime=0;b.currentTime=0}};card.querySelector('[data-action=play]').onclick=async()=>{{stop('正在启动');a.currentTime=0;b.currentTime=0;const r=await Promise.allSettled([a.play(),b.play()]);if(r.some(x=>x.status==='rejected')){{stop('浏览器阻止播放，请重试');return}}timer=setInterval(()=>{{if(card.dataset.sync==='true'&&Math.abs(a.currentTime-b.currentTime)>.08)b.currentTime=a.currentTime;s.textContent=card.dataset.sync==='true'?`同步误差 ${{Math.abs(a.currentTime-b.currentTime).toFixed(3)}}s`:'独立时长同时播放';if(a.paused&&b.paused)stop('播放结束')}},250)}};return()=>stop('已暂停')}});document.getElementById('pauseAll').onclick=()=>controls.forEach(x=>x());document.getElementById('filter').onchange=e=>cards.forEach(c=>c.hidden=e.target.value!=='all'&&c.dataset.dataset!==e.target.value);
</script></body></html>'''
    temporary = output_root / "index.html.tmp"
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(output_root / "index.html")


def write_summary(
    selection: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    """写入机器可读进度、数据集时长和部署文件指纹。"""

    by_dataset: dict[str, dict[str, Any]] = {}
    for key, label in DATASET_LABELS.items():
        selected = [value for value in selection if value["dataset"] == key]
        done = [value for value in completed if value["dataset"] == key]
        by_dataset[key] = {
            "label": label,
            "selected_count": len(selected),
            "completed_count": len(done),
            "audio_duration_sec": sum(value["audio_duration_sec"] for value in selected),
            "original_duration_sec": sum(value.get("original_frames", 0) / 30 for value in done),
            "generated_duration_sec": sum(value.get("generated_frames", 0) / 30 for value in done),
        }
    payload = {
        "contract_version": "genmo.smpl_hq4_full_comparison.v1",
        "checkpoint": {
            "path": str(args.ckpt),
            "size_bytes": args.ckpt.stat().st_size,
            "sha256": args.checkpoint_sha256,
            "global_step": 300000,
        },
        "onnx": {
            "path": str(args.onnx),
            "size_bytes": args.onnx.stat().st_size,
            "sha256": args.onnx_sha256,
            "boundary": "120 帧单个 CFG 去噪步；DDIM 调度、长序列拼接与 SMPL 解码在 ONNX 外",
        },
        "generation": {
            "ddim_steps": args.ddim_steps,
            "cfg_scale": args.cfg_scale,
            "chunk_frames": args.chunk_frames,
            "chunk_overlap_frames": args.chunk_overlap_frames,
            "base_seed": args.seed,
            "complete_audio": True,
            "render_style": "smplx_neutral_full_mesh_open3d_streaming_v1",
        },
        "selected_count": len(selection),
        "completed_count": len(completed),
        "datasets": by_dataset,
        "items": completed,
    }
    atomic_json(args.output_root / "summary.json", payload)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--fourset-selection",
        type=Path,
        default=Path("outputs/smpl_hq4_s300000_full_20260824/selection.json"),
    )
    value.add_argument(
        "--source-root",
        type=Path,
        default=Path("outputs/smpl_hq4_s300000_full_20260824/sources"),
    )
    value.add_argument(
        "--ckpt",
        type=Path,
        default=Path("outputs/gem_smpl_music_only_4set_manual_q1_finetune_300k_v1/version_0/checkpoints/s300000.ckpt"),
    )
    value.add_argument(
        "--onnx",
        type=Path,
        default=Path("outputs/onnx/smpl_music/s300000_hq4_finetune/music_only_denoiser.onnx"),
    )
    value.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/smpl_hq4_s300000_full_20260824"),
    )
    value.add_argument("--exp", default="gem_smpl_music_only_4set_curated")
    value.add_argument("--device", default="0")
    value.add_argument("--ddim-steps", type=int, default=50)
    value.add_argument("--cfg-scale", type=float, default=2.5)
    value.add_argument("--chunk-frames", type=int, default=600)
    value.add_argument("--chunk-overlap-frames", type=int, default=120)
    value.add_argument("--max-frames", type=int, default=30000)
    value.add_argument("--seed", type=int, default=20260824)
    value.add_argument("--width", type=int, default=640)
    value.add_argument("--height", type=int, default=360)
    value.add_argument("--limit", type=int)
    value.add_argument(
        "--only-web-id",
        help="只处理一个 selection 中的 web_id，用于定点重渲染或故障恢复。",
    )
    value.add_argument("--shard-count", type=int, default=1)
    value.add_argument("--shard-index", type=int, default=0)
    value.add_argument(
        "--worker-no-manifest",
        action="store_true",
        help="分片 worker 只写独立 mesh-ready sidecar，不争写全局网页清单。",
    )
    value.add_argument("--prepare-only", action="store_true")
    value.add_argument(
        "--rerender-mesh",
        action="store_true",
        help="复用已有生成参数，确保原始/生成视频均渲染为完整 SMPL-X 网格。",
    )
    return value


def validate_args(args: argparse.Namespace) -> None:
    """解析绝对路径并检查会影响正式生成结果的参数。"""

    args.fourset_selection = args.fourset_selection.expanduser().resolve()
    args.source_root = args.source_root.expanduser().resolve()
    args.ckpt = args.ckpt.expanduser().resolve()
    args.onnx = args.onnx.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    for path in (args.fourset_selection, args.ckpt, args.onnx):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.ddim_steps < 2 or args.chunk_frames <= 0:
        raise ValueError("DDIM 至少 2 步且 chunk_frames 必须为正")
    if not 0 <= args.chunk_overlap_frames < args.chunk_frames:
        raise ValueError("chunk overlap 必须位于 [0, chunk_frames)")
    if args.max_frames < args.chunk_frames or args.width <= 0 or args.height <= 0:
        raise ValueError("max_frames/画面尺寸不合法")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index 必须位于 [0, shard-count)")
    if args.only_web_id is not None and args.shard_count != 1:
        raise ValueError("--only-web-id 不能与多分片参数同时使用")
    args.output_root.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    validate_args(args)
    selection = build_selection(
        args.fourset_selection,
        args.source_root / "human",
        args.seed,
    )
    if not args.worker_no_manifest:
        atomic_json(
            args.output_root / "selection.json",
            {
                "contract_version": "genmo.smpl_hq4_selection.v1",
                "policy": "服务器1 main 模型的四个人工评分1 SMPLX 数据集各10条",
                "items": selection,
            },
        )
    args.checkpoint_sha256 = sha256_file(args.ckpt)
    args.onnx_sha256 = sha256_file(args.onnx)
    completed_path = args.output_root / "completed_items.json"
    previous = (
        json.loads(completed_path.read_text(encoding="utf-8"))
        if completed_path.is_file()
        else []
    )
    previous_by_id = {value["web_id"]: value for value in previous}
    completed = [
        {
            **{
                key: child
                for key, child in previous_by_id[value["web_id"]].items()
                if key != "original_skeleton"
            },
            **value,
        }
        for value in selection
        if value["web_id"] in previous_by_id
    ]
    for value in completed:
        sidecar = (
            args.output_root
            / "videos"
            / value["dataset"]
            / f"{value['web_id']}_mesh_ready.json"
        )
        if sidecar.is_file():
            sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
            if sidecar_payload.get("web_id") != value["web_id"]:
                raise ValueError(f"mesh-ready sidecar 身份不匹配：{sidecar}")
            value.update(sidecar_payload)
    if not args.worker_no_manifest:
        atomic_json(completed_path, completed)
        write_summary(selection, completed, args)
        write_page(completed, args.output_root, args.ckpt, args.onnx, args)
    if args.prepare_only:
        print(f"已冻结 {len(selection)} 条四库清单：{args.output_root / 'selection.json'}")
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("完整 GEM 生成和 SMPL 渲染需要 CUDA")
    from gem.utils.smplx_utils import make_smplx

    body_model = make_smplx("supermotion").cuda().eval()
    if args.only_web_id is not None:
        requested = [value for value in selection if value["web_id"] == args.only_web_id]
        if not requested:
            raise ValueError(f"selection 中不存在 --only-web-id={args.only_web_id}")
    else:
        requested = selection if args.limit is None else selection[: args.limit]
        requested = requested[args.shard_index :: args.shard_count]
    completed_by_id = {value["web_id"]: value for value in completed}
    for position, item in enumerate(requested, start=1):
        started = time.monotonic()
        existing = completed_by_id.get(item["web_id"])
        mesh_ready = (
            existing is not None
            and existing.get("render_style")
            == "smplx_neutral_full_mesh_open3d_streaming_v1"
        )
        if existing is not None and (not args.rerender_mesh or mesh_ready):
            print(f"[{position:02d}/{len(requested):02d}] 复用 {item['web_id']}", flush=True)
            continue
        action = "网格重渲染" if args.rerender_mesh else "生成"
        print(f"[{position:02d}/{len(requested):02d}] {action} {item['web_id']}", flush=True)
        artifact_dir = args.output_root / "artifacts" / item["dataset"] / item["web_id"]
        report = run_generation(item, args, artifact_dir)
        generated_body = load_generated_body(artifact_dir / "pred_body_params_global.pt")
        generated_frames = body_frames(generated_body)
        video_dir = args.output_root / "videos" / item["dataset"]
        video_dir.mkdir(parents=True, exist_ok=True)
        work_dir = args.output_root / "_work" / item["web_id"]
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        original_output = video_dir / f"{item['web_id']}_original.mp4"
        generated_output = video_dir / f"{item['web_id']}_generated.mp4"
        original_candidate = work_dir / "original_with_audio.mp4"
        generated_candidate = work_dir / "generated_with_audio.mp4"
        try:
            original_body = load_human_original(Path(item["original_motion_path"]))
            original_frames = render_smpl_mesh(
                body_model,
                original_body,
                work_dir / "original_silent.mp4",
                args.width,
                args.height,
                work_dir / "original_vertices.f32",
                work_dir / "original_normals.f32",
            )
            mux_audio(
                work_dir / "original_silent.mp4",
                Path(item["audio_path"]),
                original_candidate,
                original_frames / 30.0,
            )
            render_smpl_mesh(
                body_model,
                generated_body,
                work_dir / "generated_silent.mp4",
                args.width,
                args.height,
                work_dir / "generated_vertices.f32",
                work_dir / "generated_normals.f32",
            )
            mux_audio(
                work_dir / "generated_silent.mp4",
                Path(item["audio_path"]),
                generated_candidate,
                generated_frames / 30.0,
            )
            original_probe = probe_final_video(original_candidate, original_frames)
            generated_probe = probe_final_video(generated_candidate, generated_frames)
            original_candidate.replace(original_output)
            generated_candidate.replace(generated_output)
        finally:
            if work_dir.exists():
                shutil.rmtree(work_dir)
        finished = {**completed_by_id.get(item["web_id"], {}), **item}
        finished.update(
            {
                "original_frames": original_frames,
                "generated_frames": generated_frames,
                "original_video": str(original_output.relative_to(args.output_root)),
                "generated_video": str(generated_output.relative_to(args.output_root)),
                "original_media": original_probe,
                "generated_media": generated_probe,
                "motion_sanity": report.get("motion_sanity"),
                "chunking": report.get("chunking"),
                "render_style": "smplx_neutral_full_mesh_open3d_streaming_v1",
                "rendering_seconds": time.monotonic() - started,
            }
        )
        mesh_sidecar = video_dir / f"{item['web_id']}_mesh_ready.json"
        atomic_json(
            mesh_sidecar,
            {
                "web_id": item["web_id"],
                "original_frames": original_frames,
                "generated_frames": generated_frames,
                "original_media": original_probe,
                "generated_media": generated_probe,
                "render_style": "smplx_neutral_full_mesh_open3d_streaming_v1",
                "rendering_seconds": finished["rendering_seconds"],
            },
        )
        completed_by_id[item["web_id"]] = finished
        completed = [completed_by_id[value["web_id"]] for value in selection if value["web_id"] in completed_by_id]
        if not args.worker_no_manifest:
            atomic_json(completed_path, completed)
            write_summary(selection, completed, args)
            write_page(completed, args.output_root, args.ckpt, args.onnx, args)
        print(
            f"[{position:02d}/{len(requested):02d}] 完成 {item['web_id']}："
            f"原始 {original_frames} 帧，生成 {generated_frames} 帧，"
            f"耗时 {time.monotonic() - started:.1f}s",
            flush=True,
        )
    work_root = args.output_root / "_work"
    if work_root.is_dir() and not any(work_root.iterdir()):
        work_root.rmdir()
    missing_requested = [
        value["web_id"] for value in requested if value["web_id"] not in completed_by_id
    ]
    if missing_requested:
        raise RuntimeError(f"本次请求仍缺少结果：{missing_requested}")
    if args.worker_no_manifest:
        print(f"渲染分片 {args.shard_index}/{args.shard_count} 已完成")
    else:
        print(f"四库网页已完成：{args.output_root / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
