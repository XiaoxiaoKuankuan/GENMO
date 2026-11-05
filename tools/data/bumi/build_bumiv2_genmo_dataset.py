"""整理 BUMI V2 的 30 Hz 音乐舞蹈输入包，供后续 bumiv2_genmo 接入。

输入为已有 UMR V2 qpos、人体导出配对索引、自建 50 Hz V2 CSV 和原始 WAV。
UMR 按权威 sample_id 复用 EDGE35 特征；AIST 必须使用逐动作 aligned_wav，
不能以同曲完整音乐代替。自建动作按表头映射关节，平移/关节线性插值、旋转
最短弧 SLERP，50→30 Hz；使用 BUMI4340 同名原音乐重新提取 EDGE35 特征。
自建动作与音乐相差超过一个 30 Hz 帧即剔除。UMR 已有配对只允许最多两帧
的音频端点裁齐；所有输出都取共同时间轴，不外推动作、不补音乐或特征。

输出包含 qpos[T,28] 安全 NPZ、float32 音乐 Tensor[T,35]、精确时长 WAV、
逐样本 manifest、剔除清单及全量验证报告。根四元数为 wxyz，关节按显式名称
统一为腰、左臂、右臂、左腿、右腿；保留原始世界坐标、根高度及关节角。
不复用 BUMI4340 的接触标签、动力学筛选结果或训练统计量，不修改源数据，
不启动训练，也不宣称现有 BUMI4340 reader 已支持 V2。原始 split 仅供追溯，
训练前仍须按 music_group_id 划分；报告列出源 split 的同曲交叉情况。

全流程仅使用 CPU。只有显式 --trust-umr-metadata 才读取可信 UMR 的 object
关节名称；复用已有 NumPy 1/2 兼容 reader，发布 NPZ 无需 pickle。先写独立
.building 目录并全量回读、校验哈希与帧数，再原子改名发布，拒绝覆盖旧数据。
示例（服务器1、GENMO仓库根目录）：
  python -m tools.data.bumi.build_bumiv2_genmo_dataset \
    --output /data0/user/liwei/datasets/bumiv2_genmo_music30_20261105 \
    --trust-umr-metadata
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import wave
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import numpy as np
import torch

from gem.robots.bumi.motion_utils import NumpyCompatibleUnpickler, sha256_file
from gem.utils.music_features import EDGE_BASELINE_FEATURE_NAMES, extract_edge_baseline35
from scripts.csv_to_npz_bumi_v2 import SUPERMIMIC_JOINT_NAMES, load_motion, resample_motion

FPS = 30
JOINT_NAMES = (
    ("waist_yaw_joint",)
    + tuple(
        f"{side}_{part}_joint"
        for side in ("l", "r")
        for part in ("arm_pitch", "arm_roll", "arm_yaw", "elbow_pitch")
    )
    + tuple(
        f"{side}_{part}_joint"
        for side in ("l", "r")
        for part in ("leg_pitch", "leg_roll", "leg_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")
    )
)
DIRECTORIES = {
    "aistpp": "AIST++",
    "aioz_gdance": "AIOZ-GDANCE",
    "finedance": "FineDance",
    "mine": "Mine",
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


@cache
def source_hash(path):
    return sha256_file(path)


def umr_sample_id(path):
    return path.stem.removesuffix("_bumi_v2").removeprefix("aioz_gdance__")


def load_umr(path, trust_metadata):
    """一次性检查外部 UMR 输入；仅显式关节名用于重排，位姿数值不修正。"""
    with zipfile.ZipFile(path) as archive, archive.open("robot_joint_names.npy") as handle:
        version = np.lib.format.read_magic(handle)
        readers = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
        }
        _, _, dtype = readers[version](handle)
        if dtype.hasobject and not trust_metadata:
            raise ValueError("object 关节名称需要 --trust-umr-metadata")
        names = NumpyCompatibleUnpickler(handle).load().tolist() if dtype.hasobject else None
    with np.load(path, allow_pickle=False) as z:
        names = names if names is not None else z["robot_joint_names"].tolist()
        qpos = z["qpos"].astype(np.float32)
        if (
            str(z["robot_name"].item()) != "bumi_v2"
            or float(z["fps"].item()) != FPS
            or not np.array_equal(z["frame_ids"], np.arange(len(qpos)))
        ):
            raise ValueError(f"UMR 机器人、帧率或时间轴错误: {path}")
        metadata = {
            k: str(z[k].item())
            for k in ("robot_xml", "source_data", "source_sequence_key", "source_format")
        }
    if len(names) != 21 or set(names) != set(JOINT_NAMES):
        raise ValueError(f"V2 关节名称不完整: {path}")
    validate_qpos(qpos)
    return qpos[:, list(range(7)) + [7 + names.index(n) for n in JOINT_NAMES]], metadata


def validate_qpos(qpos):
    if qpos.ndim != 2 or qpos.shape[1] != 28 or len(qpos) < 2 or not np.isfinite(qpos).all():
        raise ValueError("qpos 必须为有限的 [T>=2,28]")
    if np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1)) > 1e-5:
        raise ValueError("根四元数未归一化")


def audio_info(path):
    with wave.open(str(path), "rb") as wav:
        return {
            "samples": wav.getnframes(),
            "sample_rate": wav.getframerate(),
            "channels": wav.getnchannels(),
            "sample_width": wav.getsampwidth(),
        }


def aligned_frames(motion_frames, info):
    """保留音频可覆盖的完整 30 Hz 帧；整数计算避免端点浮点误差。"""
    return min(motion_frames, info["samples"] * FPS // info["sample_rate"])


def duration_matches(source_frames, source_fps, info):
    return (
        abs(source_frames / source_fps - info["samples"] / info["sample_rate"])
        <= 1 / FPS + 1 / info["sample_rate"]
    )


def crop_audio(source, destination, frames):
    """保留原采样率/声道/PCM 位宽；裁尾到 T/30，不补零、不拉伸。"""
    info = audio_info(source)
    count = round(frames * info["sample_rate"] / FPS)
    if count > info["samples"]:
        raise ValueError(f"音频不足: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if count == info["samples"]:
        shutil.copy2(source, destination)
    else:
        with wave.open(str(source), "rb") as src, wave.open(str(destination), "wb") as dst:
            dst.setparams(src.getparams())
            remaining = count
            while remaining:
                chunk = min(remaining, 65536)
                dst.writeframesraw(src.readframes(chunk))
                remaining -= chunk
    return info


def mine_audio(root, song):
    aliases = {
        "APT啦啦操": "APT啦啦操_30fps",
        "APT啦啦操_short": "APT啦啦操_30fps",
        "shakeit": "shake-it-nutuan4",
    }
    stem = aliases.get(song, song)
    for folder in ("dance_3_音频", "dance_2_音频"):
        path = root / folder / (stem + ".wav")
        if path.is_file():
            return path
    raise FileNotFoundError(f"未找到已确认的同曲音乐: {song}")


def save_pair(stage, dataset, sid, qpos, features, audio, provenance, audio_cache):
    directory = Path(DIRECTORIES[dataset])
    n = len(qpos)
    validate_qpos(qpos)
    features = torch.as_tensor(features).detach().cpu().float().contiguous()
    if features.shape != (n, 35) or not torch.isfinite(features).all():
        raise ValueError(f"音乐特征应为有限的 [{n},35]: {sid}")
    key = (str(audio), n)
    if key not in audio_cache:
        audio_rel = directory / "audio" / f"{audio.stem}__{n}f.wav"
        crop_audio(audio, stage / audio_rel, n)
        audio_cache[key] = (str(audio_rel), sha256_file(stage / audio_rel))
    audio_rel, audio_hash = audio_cache[key]
    motion_rel = directory / "motions" / f"{sid}.npz"
    feature_rel = directory / "musicfeat_v2" / f"{sid}_musicfeat_fps30.pt"
    for rel in (motion_rel, feature_rel):
        (stage / rel).parent.mkdir(parents=True, exist_ok=True)
        if (stage / rel).exists():
            raise FileExistsError(stage / rel)
    np.savez_compressed(
        stage / motion_rel,
        qpos=qpos.astype(np.float32),
        fps=np.asarray([FPS], np.float32),
        frame_ids=np.arange(n, dtype=np.int32),
        robot_name=np.asarray("bumi_v2"),
        robot_joint_names=np.asarray(JOINT_NAMES),
        quaternion_order=np.asarray("wxyz"),
    )
    torch.save(features, stage / feature_rel)
    return {
        "sample_id": sid,
        "dataset": dataset,
        "robot_name": "bumi_v2",
        "fps": FPS,
        "num_frames": n,
        "duration_sec": n / FPS,
        "audio_start_sec": 0,
        "motion_path": str(motion_rel),
        "music_feature_path": str(feature_rel),
        "audio_path": audio_rel,
        "motion_sha256": sha256_file(stage / motion_rel),
        "music_feature_sha256": sha256_file(stage / feature_rel),
        "audio_sha256": audio_hash,
        **provenance,
    }


def validate_release(root, rows):
    """全量回读全部模态并核对 SHA；共享音频只校验一次。"""
    audio_checked = {}
    for row in rows:
        n = row["num_frames"]
        with np.load(root / row["motion_path"], allow_pickle=False) as z:
            for key in z.files:
                z[key]  # 所有字段必须无需 pickle。
            validate_qpos(z["qpos"])
            assert z["qpos"].shape == (n, 28)
            assert tuple(z["robot_joint_names"].tolist()) == JOINT_NAMES
            assert float(z["fps"].item()) == FPS and str(z["robot_name"].item()) == "bumi_v2"
        feature = torch.load(
            root / row["music_feature_path"], map_location="cpu", weights_only=True
        )
        assert (
            feature.shape == (n, 35)
            and feature.dtype == torch.float32
            and torch.isfinite(feature).all()
        )
        for kind in ("motion", "music_feature"):
            assert sha256_file(root / row[kind + "_path"]) == row[kind + "_sha256"]
        ap = row["audio_path"]
        if ap not in audio_checked:
            info = audio_info(root / ap)
            assert sha256_file(root / ap) == row["audio_sha256"]
            audio_checked[ap] = info
        info = audio_checked[ap]
        assert abs(info["samples"] / info["sample_rate"] - n / FPS) <= 1 / info["sample_rate"]
    return {
        "validated_pairs": len(rows),
        "unique_audio_files": len(audio_checked),
        "all_finite": True,
        "all_sha256_verified": True,
        "all_npz_without_pickle": True,
        "all_modalities_same_duration": True,
    }


def build(args):
    output = args.output.resolve()
    stage = output.with_name(output.name + ".building")
    if output.exists() or stage.exists():
        raise FileExistsError(f"拒绝覆盖: {output} 或 {stage}")
    index = {r["sample_id"]: r for r in map(json.loads, args.source_index.open())}
    paths = sorted(args.umr_root.glob("*.npz"))
    csvs = sorted(args.mine_csv_root.glob("*.csv"))
    if not paths or not csvs:
        raise ValueError("UMR 与自建源目录均须非空")
    stage.mkdir(parents=True)
    (stage / "provenance").mkdir()
    shutil.copy2(args.source_index, stage / "provenance/source_pairing_index.jsonl")
    rows, excluded, audio_cache = [], [], {}
    for i, path in enumerate(paths, 1):
        sid = umr_sample_id(path)
        src = index[sid]
        ds = src["dataset"]
        qpos, meta = load_umr(path, args.trust_umr_metadata)
        feature_path = Path(src["source_root"]) / src["music_feature_path"]
        feature = torch.load(feature_path, map_location="cpu", weights_only=True)
        if len(qpos) != src["num_frames"] or feature.shape != (len(qpos), 35):
            raise ValueError(f"UMR 权威配对的帧数不一致: {sid}")
        audio = {
            "aistpp": args.aist_audio_root / f"{sid}.wav",
            "aioz_gdance": args.aioz_audio_root / f"{src['group_id']}.wav",
            "finedance": args.finedance_audio_root / f"{sid}.wav",
        }[ds]
        info = audio_info(audio)
        n = aligned_frames(len(qpos), info)
        if len(qpos) - n > 2:
            raise ValueError(f"UMR 配对音乐不足且超过两帧端点差，需核对: {sid}")
        provenance = {
            "source_motion": str(path),
            "source_motion_sha256": source_hash(path),
            "source_audio": str(audio),
            "source_audio_sha256": source_hash(audio),
            "source_music_feature": str(feature_path),
            "source_music_feature_sha256": source_hash(feature_path),
            "source_num_frames": len(qpos),
            "source_fps": FPS,
            "trimmed_motion_frames": len(qpos) - n,
            "source_audio_duration_sec": info["samples"] / info["sample_rate"],
            "source_split": src["split"],
            "music_group_id": f"{ds}:{src['music_id']}",
            "pairing_basis": "source_index_and_aligned_audio",
            "umr_metadata": meta,
        }
        rows.append(
            save_pair(stage, ds, sid, qpos[:n], feature[:n], audio, provenance, audio_cache)
        )
        if i % 500 == 0:
            print(f"UMR {i}/{len(paths)}", flush=True)
    for i, path in enumerate(csvs, 1):
        song = re.sub(r"^bumi2_\d+_", "", path.stem).removesuffix("_50fps")
        motion = load_motion(path)
        audio = mine_audio(args.mine_audio_root, song)
        info = audio_info(audio)
        provenance = {
            "source_motion": str(path.resolve()),
            "source_motion_sha256": source_hash(path),
            "source_audio": str(audio),
            "source_audio_sha256": source_hash(audio),
            "source_num_frames": len(motion),
            "source_fps": 50,
            "source_motion_duration_sec": len(motion) / 50,
            "source_audio_duration_sec": info["samples"] / info["sample_rate"],
            "song_name": song,
            "source_split": "unassigned",
            "music_group_id": f"mine:{song}",
            "pairing_basis": "same_song_bumi4340_audio_and_duration_check",
        }
        if not duration_matches(len(motion), 50, info):
            excluded.append(
                {"sample_id": path.stem, "reason": "audio_motion_duration_mismatch", **provenance}
            )
            print(
                f"剔除 {song}: 动作 {len(motion) / 50:.5f}s，音乐 {info['samples'] / info['sample_rate']:.5f}s",
                flush=True,
            )
            continue
        qpos = resample_motion(motion, 50, FPS)
        qpos = qpos[:, list(range(7)) + [7 + SUPERMIMIC_JOINT_NAMES.index(j) for j in JOINT_NAMES]]
        n = aligned_frames(len(qpos), info)
        feature, feature_meta = extract_edge_baseline35(audio, duration_sec=n / FPS)
        if len(feature) < n:
            raise ValueError(f"自建音乐特征不足，禁止补帧: {song}")
        provenance.update(
            {
                "resampled_num_frames": len(qpos),
                "trimmed_motion_frames": len(qpos) - n,
                "music_extraction": feature_meta,
            }
        )
        rows.append(
            save_pair(
                stage,
                "mine",
                path.stem.removesuffix("_50fps"),
                qpos[:n],
                feature[:n],
                audio,
                provenance,
                audio_cache,
            )
        )
        print(f"自建 {i}/{len(csvs)} {song}: {n} 帧", flush=True)
    write_jsonl(stage / "manifest.jsonl", rows)
    write_jsonl(stage / "excluded.jsonl", excluded)
    for ds, folder in DIRECTORIES.items():
        write_jsonl(stage / folder / "manifest.jsonl", [r for r in rows if r["dataset"] == ds])
    groups = defaultdict(set)
    for row in rows:
        groups[row["music_group_id"]].add(row["source_split"])
    overlap = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    validation = validate_release(stage, rows)
    validation["source_split_music_overlap"] = overlap
    write_json(stage / "validation_report.json", validation)
    counts = dict(Counter(r["dataset"] for r in rows))
    total_frames = sum(r["num_frames"] for r in rows)
    summary = {
        "schema": "bumiv2_genmo_music30.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robot_name": "bumi_v2",
        "fps": FPS,
        "qpos_dim": 28,
        "quaternion_order": "wxyz",
        "joint_names": JOINT_NAMES,
        "music_feature_names": EDGE_BASELINE_FEATURE_NAMES,
        "counts": counts,
        "total_pairs": len(rows),
        "total_frames": total_frames,
        "total_hours": total_frames / FPS / 3600,
        "excluded_mine": len(excluded),
        "trimmed_pairs": sum(r["trimmed_motion_frames"] > 0 for r in rows),
        "trimmed_frames": sum(r["trimmed_motion_frames"] for r in rows),
        "source_index": str(args.source_index),
        "source_index_sha256": source_hash(args.source_index),
        "source_split_policy": "provenance_only_regroup_by_music_before_training",
        "physics_quality_screened": False,
        "contact_labels_provided": False,
        "existing_bumi4340_loader_compatible": False,
    }
    write_json(stage / "dataset_info.json", summary)
    (stage / "README.zh-CN.txt").write_text(
        "BUMI V2 GENMO 音乐舞蹈输入包\n\n"
        f"共 {len(rows)} 对，30 Hz，{total_frames} 帧，{total_frames / FPS / 3600:.4f} 小时。\n"
        f"子集数量：{json.dumps(counts, ensure_ascii=False)}；自建剔除 {len(excluded)} 条。\n"
        "manifest.jsonl 的路径均相对本目录；每条提供动作、音乐特征、WAV、来源和 SHA256。\n"
        "动作 NPZ 用 np.load(allow_pickle=False) 读取；qpos=[世界位置xyz,根四元数wxyz,21关节角]。\n"
        "音乐 PT 用 torch.load(weights_only=True) 读取，为 float32[T,35]；WAV 时长=T/30。\n"
        "UMR 保留原始位姿，只按音频共同时间轴裁尾最多两帧。自建50→30Hz，不外推、不补帧。\n"
        "AIST 使用逐动作 aligned_wav。自建同名配对、APT 文件别名及排除理由可查 manifest/excluded。\n"
        "source_split 仅保留来源；其同曲交叉见 validation_report.json，正式训练应按音乐组重新划分。\n"
        "本包未进行 V2 动力学筛选，未附接触标签，也没有移用 BUMI4340 的 PASS 标记或归一化统计。\n"
        "现有 BUMI4340 loader/Adapter 不因此自动兼容；后续接入须使用本包具名的 V2 机器人契约。\n",
        encoding="utf-8",
    )
    stage.rename(output)
    print(
        json.dumps(
            {"output": str(output), **summary, "validation": validation}, ensure_ascii=False
        ),
        flush=True,
    )


def main():
    base = Path("/data0/user/liwei/datasets")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-umr-metadata", action="store_true")
    defaults = {
        "umr-root": base / "music7286_umr_bumi_v2/raw_umr",
        "source-index": Path(
            "/data0/user/liwei/dataset_packages/music_smplx_4set_7286_30hz_zup_v1/index/manifest.jsonl"
        ),
        "mine-csv-root": Path("/home/user/liwei/noetix_supermimic/data/bumi2_dance"),
        "mine-audio-root": base / "bumi_music_umr70_mine_sources_v1/bumi_music_mine_raw_v1",
        "aist-audio-root": base
        / "music7286_umr_bumi_v2/imports/aist_aligned_wav_for_genmo_20261105",
        "aioz-audio-root": base / "music_dance_raw/AIOZ-GDANCE/extracted/musics",
        "finedance-audio-root": base / "music_dance_raw/FineDance/raw/finedance/music_wav",
    }
    for name, value in defaults.items():
        parser.add_argument("--" + name, type=Path, default=value)
    torch.set_num_threads(1)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
