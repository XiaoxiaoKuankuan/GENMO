"""BUMI 音乐数据集发布器共享的中性文件、索引与配对工具。

本模块把四个权威人体音乐库的固定名称、JSONL 读写、命令行 ``DATASET=PATH``
解析、受根目录约束的相对路径解析、人体 split 索引、音乐配对字段、SHA256 校验、
硬链接优先物化和 EDGE35 张量校验集中在一个 producer 无关的边界。当前 UMR qpos、
robot_retargeter 以及 transfer filelist 直接复用这些规则，不必再为了发布辅助函数导入
某个已经退役或特定来源的数据构建入口。

这里不决定动作是否通过质量门禁，也不改变 split、音乐配对或落地语义。所有函数
保留原有异常类型、路径逃逸检查、摘要格式和 hardlink→copy 回退行为；调用方仍
负责 staging/原子发布以及最终数据契约的完整校验。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch

from gem.datasets.music_dance.music_dance_bumi import safe_torch_load
from gem.robots.bumi.motion_utils import sha256_file

DATASET_SPECS: dict[str, dict[str, str]] = {
    "aistpp": {"output": "AIST++", "contract_name": "aistpp_bumi"},
    "aioz_gdance": {"output": "AIOZ-GDANCE", "contract_name": "aioz_gdance_bumi"},
    "finedance": {"output": "FineDance", "contract_name": "finedance_bumi"},
    "compas3d": {"output": "CoMPAS3D", "contract_name": "compas3d_bumi"},
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_AIOZ_DANCER = re.compile(r"_dancer_\d+$")
_COMPAS_ROLE = re.compile(r"_(leader|follower)$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取非空 JSONL 行，并要求每行都是 object。"""

    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    """以稳定 UTF-8、排序键和结尾换行写 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """以稳定键序写 JSONL，并在需要时创建父目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def parse_dataset_mapping(value: str) -> tuple[str, Path]:
    """解析 argparse 使用的 ``DATASET=/absolute/path`` 参数。"""

    if "=" not in value:
        raise argparse.ArgumentTypeError("expected DATASET=/absolute/path")
    name, raw_path = value.split("=", 1)
    if name not in DATASET_SPECS:
        raise argparse.ArgumentTypeError(
            f"unknown dataset {name!r}; expected one of {sorted(DATASET_SPECS)}"
        )
    return name, Path(raw_path).expanduser().resolve()


def require_dataset_mapping(values: list[tuple[str, Path]], option: str) -> dict[str, Path]:
    """验证四库路径映射完整、唯一且都指向目录。"""

    result: dict[str, Path] = {}
    for name, path in values:
        if name in result:
            raise ValueError(f"duplicate {option} entry for {name}")
        result[name] = path
    missing = set(DATASET_SPECS) - set(result)
    if missing:
        raise ValueError(f"{option} is missing {sorted(missing)}")
    for name, path in result.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{option} {name}: {path}")
    return result


def resolve_relative_file(root: Path, value: Any, label: str) -> Path:
    """在给定根目录内解析并验证一个非空相对文件路径。"""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{label} must be relative: {value}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes {root}: {value}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label}: {resolved}")
    return resolved


def human_jsonl_index(root: Path) -> dict[str, dict[str, Any]]:
    """从 train/val/test manifest 建立非 AIST 人体样本索引。"""

    result: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        for row in read_jsonl(root / "manifests" / f"{split}.jsonl"):
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in result:
                raise ValueError(f"duplicate/empty human sample_id={sample_id!r} under {root}")
            if row.get("split") != split:
                raise ValueError(f"{sample_id}: source split mismatch in {root}")
            result[sample_id] = dict(row)
    return result


def _sequence_frames(value: Mapping[str, Any], sequence_id: str) -> int:
    for key in (
        "bbox_xyxy",
        "smpl_pose_global",
        "smpl_pose",
        "smpl_trans_global",
        "smpl_trans",
    ):
        candidate = value.get(key)
        shape = getattr(candidate, "shape", None)
        if shape is not None and len(shape) >= 1 and int(shape[0]) > 0:
            return int(shape[0])
    raise ValueError(f"{sequence_id}: cannot determine AIST++ annotation length")


def aist_index(root: Path) -> dict[str, dict[str, Any]]:
    """从 AIST++ annotation 与三个 split 文件建立权威样本索引。"""

    annotation_path = root / "annot_aist_30fps.pt"
    annotation = safe_torch_load(annotation_path)
    if not isinstance(annotation, dict):
        raise ValueError(f"AIST++ annotation must be a dict: {annotation_path}")
    memberships: dict[str, str] = {}
    for split in ("train", "val", "test"):
        values = safe_torch_load(root / f"{split}.pt")
        for sequence_id_raw in values:
            sequence_id = str(sequence_id_raw)
            if sequence_id in memberships:
                raise ValueError(f"AIST++ sequence appears in two splits: {sequence_id}")
            memberships[sequence_id] = split
    result: dict[str, dict[str, Any]] = {}
    for sequence_id, split in memberships.items():
        value = annotation.get(sequence_id)
        if not isinstance(value, Mapping):
            raise ValueError(f"AIST++ split references missing annotation: {sequence_id}")
        token_fields = sequence_id.split("_")
        if len(token_fields) < 5 or not token_fields[4].startswith("m"):
            raise ValueError(f"{sequence_id}: cannot parse AIST++ music token")
        result[sequence_id] = {
            "sample_id": sequence_id,
            "sequence_id": sequence_id,
            "split": split,
            "num_frames": _sequence_frames(value, sequence_id),
            "fps": 30,
            "music_feature_path": f"musicfeat_v2/{sequence_id}_musicfeat_fps30.pt",
            "music_token": token_fields[4],
        }
    return result


def load_human_indices(roots: Mapping[str, Path]) -> dict[str, dict[str, dict[str, Any]]]:
    """按固定四库名称加载所有权威人体索引。"""

    return {
        "aistpp": aist_index(roots["aistpp"]),
        "aioz_gdance": human_jsonl_index(roots["aioz_gdance"]),
        "finedance": human_jsonl_index(roots["finedance"]),
        "compas3d": human_jsonl_index(roots["compas3d"]),
    }


def _normalise_song_name(value: Any) -> str:
    result = re.sub(r"[^a-z0-9]+", "", str(value).casefold())
    if not result:
        raise ValueError(f"invalid FineDance song_name={value!r}")
    return result


def pairing_fields(dataset: str, sample_id: str, human: Mapping[str, Any]) -> dict[str, str]:
    """根据人体权威元数据返回动作、音乐组和音频键的固定配对。"""

    if dataset == "aistpp":
        sequence_id = sample_id
        music_group_id = str(human["music_token"])
        audio_key = music_group_id
    elif dataset == "aioz_gdance":
        sequence_id = str(human.get("group_id") or _AIOZ_DANCER.sub("", sample_id))
        expected = _AIOZ_DANCER.sub("", sample_id)
        if sequence_id != expected:
            raise ValueError(
                f"{sample_id}: AIOZ group_id={sequence_id!r} does not match {expected!r}"
            )
        music_group_id = sequence_id
        audio_key = sequence_id
    elif dataset == "finedance":
        sequence_id = sample_id
        music_group_id = _normalise_song_name(human.get("song_name"))
        audio_key = sample_id
    elif dataset == "compas3d":
        sequence_id = str(human.get("sequence_id") or _COMPAS_ROLE.sub("", sample_id))
        expected = _COMPAS_ROLE.sub("", sample_id)
        if sequence_id != expected:
            raise ValueError(
                f"{sample_id}: CoMPAS3D sequence_id={sequence_id!r} does not match {expected!r}"
            )
        music_group_id = str(human.get("song_id", ""))
        if not music_group_id:
            raise ValueError(f"{sample_id}: missing CoMPAS3D song_id")
        audio_key = sequence_id
    else:  # pragma: no cover - guarded by DATASET_SPECS
        raise ValueError(dataset)
    return {
        "sequence_id": sequence_id,
        "music_group_id": music_group_id,
        "audio_key": audio_key,
    }


def materialize_file(source: Path, destination: Path) -> str:
    """同内容目标复用；否则优先硬链接，跨设备时回退到 copy2。"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise ValueError(f"destination collision with different content: {destination}")
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def sample_basename(selected_row: Mapping[str, Any]) -> tuple[str, str]:
    """从选择记录中拆出固定四库名称和单层 sample basename。"""

    dataset = str(selected_row.get("dataset", ""))
    if dataset not in DATASET_SPECS:
        raise ValueError(f"unknown selected dataset={dataset!r}")
    selected_id = str(selected_row.get("sample_id", ""))
    prefix = f"{dataset}/"
    if not selected_id.startswith(prefix):
        raise ValueError(f"selected sample_id must start with {prefix!r}: {selected_id!r}")
    sample_id = selected_id[len(prefix) :]
    if not sample_id or Path(sample_id).name != sample_id:
        raise ValueError(f"sample_id must be one basename: {sample_id!r}")
    return dataset, sample_id


def validate_sha256(value: Any, label: str) -> str:
    """验证小写 64 位 SHA256 文本并原样返回。"""

    digest = str(value)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} is not a SHA256 digest: {value!r}")
    return digest


def load_music_tensor(path: Path, sample_id: str) -> torch.Tensor:
    """读取并验证原始 ``[T,35]`` finite EDGE35 张量。"""

    value = safe_torch_load(path)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{sample_id}: EDGE35 must be a raw Tensor: {path}")
    value = value.detach().cpu().float()
    if value.ndim != 2 or value.shape[1] != 35 or value.shape[0] <= 0:
        raise ValueError(f"{sample_id}: EDGE35 must have shape [T,35], got {value.shape}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{sample_id}: EDGE35 contains NaN/Inf: {path}")
    return value
