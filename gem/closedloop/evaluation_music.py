"""第8步四库音乐的固定选择、来源校验和只读特征加载。

选择只依赖已有 manifest：按重划分关联组 ID 排序，每组取最长样本，同长度按 sample_id
排序。在线模型只读取音乐特征，不打开动作文件或监督标签。音频保留给同步视频，两个
文件均按 manifest SHA 验证；本模块不修改数据划分，也不把旧预训练见过的 val 声称为未见。
训练集专项显式读取 train 清单，按固定种子散列排列关联组，组内仍取最长条目，
并按音频内容 SHA 全局去重，避免把同一音乐的不同舞者计为不同歌曲。选样不依赖
模型输出或文件是否已下载。无视频评估可只要求音乐特征；缺少的音频单独报告，
不伪造已验证音频，也不影响基于 EDGE35 节拍特征的指标计算。
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path

import numpy as np


def music_control_steps(feature_frames, maximum_seconds):
    """把30Hz音乐长度映射到50Hz可执行步数，协调器和报告共用同一边界。

    音乐帧数用整数有理数计算，避免492帧的16.4秒在浮点乘50后成为
    819.999...并少执行一步。显式秒上限只容忍1e-9步的浮点舍入误差，
    不把不足一个控制周期的音乐尾部补齐或循环。
    """
    if isinstance(feature_frames, bool) or not isinstance(feature_frames, Integral) or feature_frames < 0:
        raise ValueError("Music feature frame count must be a nonnegative integer")
    if isinstance(maximum_seconds, bool) or not math.isfinite(float(maximum_seconds)) or maximum_seconds <= 0:
        raise ValueError("Maximum music seconds must be finite and positive")
    return min(int(feature_frames) * 50 // 30, math.floor(float(maximum_seconds) * 50 + 1e-9))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_val_music(data_root, datasets, groups_per_dataset=2):
    root = Path(data_root).resolve()
    selected = []
    for dataset in datasets:
        count = groups_per_dataset[dataset] if isinstance(groups_per_dataset, Mapping) else groups_per_dataset
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"Invalid group count for {dataset}: {count}")
        manifest = root / dataset / "manifests/val.jsonl"
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        groups = {}
        for row in rows:
            if row["split"] != "val" or float(row["fps"]) != 30:
                raise ValueError(f"Invalid val/fps record: {row['sample_id']}")
            group_id = row["resplit_provenance"]["group_id"]
            groups.setdefault(group_id, []).append(row)
        if len(groups) < count:
            raise ValueError(f"{dataset}: insufficient independent val groups")
        for group_id in sorted(groups)[:count]:
            row = sorted(groups[group_id], key=lambda v: (-int(v["num_frames"]), v["sample_id"]))[0]
            selected.append({"dataset": dataset, "group_id": group_id, "row": row,
                             "manifest_sha256": sha256_file(manifest),
                             "pretraining_unseen_claim": False})
    return selected


def music_path(data_root, sample, key):
    root = (Path(data_root) / sample["dataset"]).resolve()
    path = (root / sample["row"][key]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Music path escapes dataset root")
    return path


def select_train_music(data_root, datasets, groups_per_dataset, *, seed=42):
    """从完整 train manifest 确定性抽取独立音乐；不足时明确失败，不混入 val/test。"""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Music selection seed must be an integer")
    root = Path(data_root).resolve()
    selected, seen_audio = [], set()
    for dataset in datasets:
        count = groups_per_dataset[dataset] if isinstance(groups_per_dataset, Mapping) else groups_per_dataset
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"Invalid group count for {dataset}: {count}")
        manifest = root / dataset / "manifests/train.jsonl"
        groups = {}
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] != "train" or float(row["fps"]) != 30:
                raise ValueError(f"Invalid train/fps record: {row['sample_id']}")
            groups.setdefault(row["resplit_provenance"]["group_id"], []).append(row)
        ranked = sorted(groups, key=lambda group: (
            hashlib.sha256(f"{seed}/{dataset}/{group}".encode("utf-8")).hexdigest(), group))
        chosen = []
        for group in ranked:
            row = sorted(groups[group], key=lambda item: (-int(item["num_frames"]), item["sample_id"]))[0]
            audio_sha = row.get("source_audio_sha256")
            if not isinstance(audio_sha, str) or len(audio_sha) != 64:
                raise ValueError(f"Missing source audio SHA for unique-music selection: {row['sample_id']}")
            if audio_sha in seen_audio:
                continue
            chosen.append({"dataset": dataset, "group_id": group, "row": row,
                           "manifest_sha256": sha256_file(manifest), "selection_seed": seed,
                           "selection_policy": "seeded_group_longest_unique_audio_sha256",
                           "pretraining_unseen_claim": False})
            seen_audio.add(audio_sha)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f"{dataset}: requested {count} independent train songs, found {len(chosen)}")
        selected.extend(chosen)
    return selected


def check_music_files(data_root, selected, *, require_audio=True):
    missing, invalid, verified, optional_missing = [], [], [], []
    for sample in selected:
        for key, sha_key in (("music_feature_path", "source_music_feature_sha256"),
                             ("audio_path", "source_audio_sha256")):
            path = music_path(data_root, sample, key)
            if not path.is_file():
                (optional_missing if key == "audio_path" and not require_audio else missing).append(str(path))
            elif sha256_file(path) != sample["row"][sha_key]:
                invalid.append(str(path))
            else:
                verified.append(str(path))
    return {"missing": sorted(set(missing)), "invalid_sha256": sorted(set(invalid)),
            "verified": sorted(set(verified)), "optional_audio_missing": sorted(set(optional_missing)),
            "audio_required": bool(require_audio)}


def load_music_features(data_root, sample):
    import torch

    path = music_path(data_root, sample, "music_feature_path")
    if sha256_file(path) != sample["row"]["source_music_feature_sha256"]:
        raise ValueError(f"Music feature SHA mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, torch.Tensor):
        raise ValueError("Existing genmo.bumi_music.v1 requires a raw EDGE35 tensor")
    values = payload.detach().cpu().float().numpy()
    if values.ndim != 2 or values.shape[1] != 35 or not np.isfinite(values).all():
        raise ValueError(f"Music features must be finite [T,35]: {path}")
    frames = int(sample["row"]["num_frames"])
    if len(values) != frames:
        raise ValueError(f"Music feature/manifest length mismatch: {path}")
    return values.copy()
