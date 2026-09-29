"""第8步四库 val 音乐的固定选择、来源校验和只读特征加载。

选择只依赖已有 manifest：按重划分关联组 ID 排序，每组取最长样本，同长度按 sample_id
排序。在线模型只读取音乐特征，不打开动作文件或监督标签。音频保留给同步视频，两个
文件均按 manifest SHA 验证；本模块不修改数据划分，也不把旧预训练见过的 val 声称为未见。
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


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


def check_music_files(data_root, selected):
    missing, invalid, verified = [], [], []
    for sample in selected:
        for key, sha_key in (("music_feature_path", "source_music_feature_sha256"),
                             ("audio_path", "source_audio_sha256")):
            path = music_path(data_root, sample, key)
            if not path.is_file():
                missing.append(str(path))
            elif sha256_file(path) != sample["row"][sha_key]:
                invalid.append(str(path))
            else:
                verified.append(str(path))
    return {"missing": sorted(set(missing)), "invalid_sha256": sorted(set(invalid)),
            "verified": sorted(set(verified))}


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
