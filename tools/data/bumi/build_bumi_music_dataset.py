"""当前 BUMI 音乐数据生产共用的配对、文件与清单校验工具。

原模块中的旧 GMR pickle/482138 整库转换与命令行入口已退役；保留原导入路径，
避免改变 UMR、robot_retargeter、CSV 配套流程及对比评估的依赖。这里只保留既有
函数实现：来源 train/val/test 索引、音乐分组、EDGE35 有效性、路径边界、摘要和
原子发布前的文件物化。不会自行运行转换，不替换正式数据、统计量或机器人资产。
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
from gem.robots.bumi.legacy_motion import sha256_file

DATASET_SPECS: dict[str, dict[str, str]] = {
    "aistpp": {"output": "AIST++", "contract_name": "aistpp_bumi"},
    "aioz_gdance": {"output": "AIOZ-GDANCE", "contract_name": "aioz_gdance_bumi"},
    "finedance": {"output": "FineDance", "contract_name": "finedance_bumi"},
    "compas3d": {"output": "CoMPAS3D", "contract_name": "compas3d_bumi"},
}


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


_AIOZ_DANCER = re.compile(r"_dancer_\d+$")


_COMPAS_ROLE = re.compile(r"_(leader|follower)$")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected DATASET=/absolute/path")
    name, raw_path = value.split("=", 1)
    if name not in DATASET_SPECS:
        raise argparse.ArgumentTypeError(
            f"unknown dataset {name!r}; expected one of {sorted(DATASET_SPECS)}"
        )
    path = Path(raw_path).expanduser().resolve()
    return name, path


def _mapping(values: list[tuple[str, Path]], option: str) -> dict[str, Path]:
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


def _relative_file(root: Path, value: Any, label: str) -> Path:
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


def _human_jsonl_index(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        for row in _read_jsonl(root / "manifests" / f"{split}.jsonl"):
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


def _aist_index(root: Path) -> dict[str, dict[str, Any]]:
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
            "music_feature_path": (f"musicfeat_v2/{sequence_id}_musicfeat_fps30.pt"),
            "music_token": token_fields[4],
        }
    return result


def load_human_indices(roots: Mapping[str, Path]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        "aistpp": _aist_index(roots["aistpp"]),
        "aioz_gdance": _human_jsonl_index(roots["aioz_gdance"]),
        "finedance": _human_jsonl_index(roots["finedance"]),
        "compas3d": _human_jsonl_index(roots["compas3d"]),
    }


def _normalise_song_name(value: Any) -> str:
    result = re.sub(r"[^a-z0-9]+", "", str(value).casefold())
    if not result:
        raise ValueError(f"invalid FineDance song_name={value!r}")
    return result


def pairing_fields(dataset: str, sample_id: str, human: Mapping[str, Any]) -> dict[str, str]:
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


def _materialize(source: Path, destination: Path) -> str:
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


def _sample_basename(selected_row: Mapping[str, Any]) -> tuple[str, str]:
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


def _check_digest(value: Any, label: str) -> str:
    digest = str(value)
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{label} is not a SHA256 digest: {value!r}")
    return digest


def _music_tensor(path: Path, sample_id: str) -> torch.Tensor:
    value = safe_torch_load(path)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{sample_id}: EDGE35 must be a raw Tensor: {path}")
    value = value.detach().cpu().float()
    if value.ndim != 2 or value.shape[1] != 35 or value.shape[0] <= 0:
        raise ValueError(f"{sample_id}: EDGE35 must have shape [T,35], got {value.shape}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{sample_id}: EDGE35 contains NaN/Inf: {path}")
    return value
