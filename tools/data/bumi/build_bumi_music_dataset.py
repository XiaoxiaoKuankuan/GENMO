#!/usr/bin/env python3
"""Build the four formal ``genmo.bumi_music.v1`` datasets atomically.

The quality-selected legacy pickle tree is immutable input.  This converter
joins every accepted BUMI motion to the authoritative human-dataset split and
EDGE35 artifact, converts xyzw/root + legacy joints to MuJoCo-native qpos28,
and materializes auditable per-dataset roots.  It does not alter GMR 已验收的 Root Z；对于
``gmr_foot_sole_ground_zero_v1``，会在最终 qpos 上用同一 BUMI3 足底 FK、地面高度 0 和
进入/退出滞回阈值重新生成整段左右脚接触标签。这样接触监督与正确坐标下的足底几何绑定，
不会沿用旧错误坐标数据的标签。A versioned ``selection_info.json`` declares the ground contract.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.music_dance.music_dance_bumi import (  # noqa: E402
    BUMI_MUSIC_CONTRACT_VERSION,
    safe_torch_load,
)
from gem.robots.bumi.kinematics import BumiKinematics  # noqa: E402
from gem.robots.bumi.contacts import (  # noqa: E402
    BUMI_CONTACT_CONTRACT_VERSION,
    derive_bumi_foot_contact,
)
from gem.robots.bumi.legacy_motion import (  # noqa: E402
    LEGACY_BUMI_MOTION_CONTRACT_VERSION,
    enforce_root_tilt_gate,
    load_legacy_bumi_motion,
    root_tilt_statistics,
    sha256_file,
)

EXPECTED_SOURCE_MJCF_SHA256 = "482138b437dbdabd6171fa8d44b55db5d7125a228c95b69ce3d1159cafe8537c"
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


def _selection_info(selected_root: Path) -> tuple[dict[str, Any], str | None]:
    """读取可选选择契约；旧 selected root 继续使用历史默认语义。"""

    path = selected_root / "meta" / "selection_info.json"
    if not path.exists():
        return {
            "contract_version": None,
            "ground_semantics": "legacy_body_origin_min_zero",
            "root_z_adjusted": False,
            "root_z_adjustment_method": "none",
        }, None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"selection info must be an object: {path}")
    ground_semantics = value.get("ground_semantics")
    if ground_semantics not in {
        "legacy_body_origin_min_zero",
        "gmr_foot_sole_ground_zero_v1",
    }:
        raise ValueError(f"unsupported selected ground_semantics={ground_semantics!r}")
    if not isinstance(value.get("root_z_adjusted"), bool):
        raise ValueError("selection info root_z_adjusted must be a boolean")
    method = value.get("root_z_adjustment_method")
    if not isinstance(method, str) or not method:
        raise ValueError("selection info root_z_adjustment_method must be non-empty")
    return dict(value), sha256_file(path)


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


def convert_datasets(
    *,
    selected_root: Path,
    human_roots: Mapping[str, Path],
    audio_roots: Mapping[str, Path],
    source_mjcf: Path,
    ik_config: Path,
    quality_config: Path,
    kinematics_path: Path,
    output_root: Path,
    expected_total: int | None = 6610,
    expected_splits: Mapping[str, int] | None = None,
    expected_dataset_counts: Mapping[str, int] | None = None,
    expected_unique_music_features: int | None = 2548,
) -> dict[str, Any]:
    """Perform a complete staged conversion and atomically publish ``output_root``."""

    selected_root = selected_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    for label, path in (
        ("selected root", selected_root),
        ("source MJCF", source_mjcf),
        ("IK config", ik_config),
        ("quality config", quality_config),
        ("kinematics", kinematics_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label}: {path}")
    mjcf_sha = sha256_file(source_mjcf)
    if mjcf_sha != EXPECTED_SOURCE_MJCF_SHA256:
        raise ValueError(
            f"source MJCF mismatch: expected={EXPECTED_SOURCE_MJCF_SHA256}, actual={mjcf_sha}"
        )
    ik_sha = sha256_file(ik_config)
    quality_sha = sha256_file(quality_config)
    kinematics = BumiKinematics(kinematics_path)
    if kinematics.source_mjcf_sha256 != mjcf_sha:
        raise ValueError("kinematics was not exported from the selected 482138 MJCF")

    selected_manifest = selected_root / "manifests" / "selected.jsonl"
    selected_rows = _read_jsonl(selected_manifest)
    selection_info, selection_info_sha = _selection_info(selected_root)
    ground_semantics = str(selection_info["ground_semantics"])
    root_z_adjusted = bool(selection_info["root_z_adjusted"])
    root_z_adjustment_method = str(selection_info["root_z_adjustment_method"])
    if expected_total is not None and len(selected_rows) != int(expected_total):
        raise ValueError(
            f"selected count mismatch: expected={expected_total}, actual={len(selected_rows)}"
        )
    human_indices = load_human_indices(human_roots)
    seen: set[tuple[str, str]] = set()
    split_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        dataset: {split: [] for split in ("train", "val", "test")} for dataset in DATASET_SPECS
    }
    counters: Counter[str] = Counter()
    hash_cache: dict[Path, str] = {}
    music_frame_cache: dict[Path, int] = {}
    materialized_destinations: dict[Path, str] = {}
    materialization: Counter[str] = Counter()
    root_tilt_by_dataset: dict[str, list[np.ndarray]] = {
        name: [] for name in DATASET_SPECS
    }

    def digest(path: Path) -> str:
        if path not in hash_cache:
            hash_cache[path] = sha256_file(path)
        return hash_cache[path]

    def materialize_once(source: Path, destination: Path, source_sha: str) -> str:
        previous = materialized_destinations.get(destination)
        if previous is not None:
            if previous != source_sha:
                raise ValueError(f"destination collision with different source SHA: {destination}")
            return "existing"
        mode = _materialize(source, destination)
        materialized_destinations[destination] = source_sha
        return mode

    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        raise FileExistsError(
            f"formal output already exists; refusing to merge partial data: {output_root}"
        )
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    try:
        for selected in selected_rows:
            dataset, sample_id = _sample_basename(selected)
            key = (dataset, sample_id)
            if key in seen:
                raise ValueError(f"duplicate selected sample: {dataset}/{sample_id}")
            seen.add(key)
            if (
                selected.get("quality_accepted") is not True
                or selected.get("quality_status") != "PASS"
            ):
                raise ValueError(f"{dataset}/{sample_id}: only PASS quality records are accepted")
            if (
                _check_digest(selected.get("quality_config_sha256"), "quality config")
                != quality_sha
            ):
                raise ValueError(f"{dataset}/{sample_id}: quality config SHA mismatch")
            if _check_digest(selected.get("source_mjcf_sha256"), "source MJCF") != mjcf_sha:
                raise ValueError(f"{dataset}/{sample_id}: source MJCF SHA mismatch")
            legacy_path = _relative_file(
                selected_root,
                selected.get("motion_path"),
                f"{dataset}/{sample_id} selected motion_path",
            )
            source_motion_sha = digest(legacy_path)
            if source_motion_sha != _check_digest(selected.get("source_sha256"), "source motion"):
                raise ValueError(f"{dataset}/{sample_id}: source motion SHA mismatch")
            human = human_indices[dataset].get(sample_id)
            if human is None:
                raise ValueError(f"{dataset}/{sample_id}: no exact human manifest match")
            if float(human.get("fps", 30)) != 30.0:
                raise ValueError(f"{dataset}/{sample_id}: human source is not 30 FPS")
            split = str(human.get("split", ""))
            if split not in {"train", "val", "test"}:
                raise ValueError(f"{dataset}/{sample_id}: invalid split={split!r}")
            pair = pairing_fields(dataset, sample_id, human)
            feature_source = _relative_file(
                human_roots[dataset],
                human.get("music_feature_path"),
                f"{dataset}/{sample_id} music_feature_path",
            )
            audio_source = audio_roots[dataset] / f"{pair['audio_key']}.wav"
            if not audio_source.is_file():
                raise FileNotFoundError(f"{dataset}/{sample_id}: audio missing: {audio_source}")

            motion = load_legacy_bumi_motion(legacy_path, expected_fps=30)
            direct_tilt = motion.root_tilt_degrees()
            root_orientation_audit = root_tilt_statistics(direct_tilt)
            root_tilt_by_dataset[dataset].append(direct_tilt)
            qpos = torch.from_numpy(motion.qpos_wxyz(kinematics.joint_order)).float()
            contact_targets = derive_bumi_foot_contact(
                qpos,
                kinematics,
                valid_mask=torch.ones(qpos.shape[0], dtype=torch.bool),
                fps=30,
                ground_height=0.0,
            )
            if feature_source not in music_frame_cache:
                music_frame_cache[feature_source] = int(
                    _music_tensor(feature_source, sample_id).shape[0]
                )
            expected_frames = int(human.get("num_frames", -1))
            lengths = {
                "qpos": int(qpos.shape[0]),
                "edge35": music_frame_cache[feature_source],
                "human_manifest": expected_frames,
            }
            if len(set(lengths.values())) != 1:
                raise ValueError(f"{dataset}/{sample_id}: frame alignment mismatch {lengths}")

            dataset_root = staging / DATASET_SPECS[dataset]["output"]
            motion_relative = Path("motions") / f"{sample_id}.pt"
            feature_relative = Path("musicfeat_v2") / feature_source.name
            audio_relative = Path("audio") / f"{pair['audio_key']}.wav"
            feature_sha = digest(feature_source)
            audio_sha = digest(audio_source)
            motion_payload = {
                "contract_version": BUMI_MUSIC_CONTRACT_VERSION,
                "source_motion_contract_version": LEGACY_BUMI_MOTION_CONTRACT_VERSION,
                "qpos": qpos.contiguous(),
                "fps": 30,
                "robot_name": "bumi",
                "joint_names": list(kinematics.joint_order),
                "quaternion_convention": "wxyz",
                "qpos_order": "mujoco_native",
                "source_dataset": dataset,
                "source_sample_id": sample_id,
                "source_motion_sha256": source_motion_sha,
                "source_mjcf_sha256": mjcf_sha,
                "retarget_config_sha256": ik_sha,
                "quality_config_sha256": quality_sha,
                "quality_accepted": True,
                "quality_record": dict(selected),
                "root_z_adjusted": root_z_adjusted,
                "root_z_adjustment_method": root_z_adjustment_method,
                "ground_semantics": ground_semantics,
                "selection_contract_version": selection_info.get("contract_version"),
                "selection_info_sha256": selection_info_sha,
                "root_orientation_audit": root_orientation_audit,
                "foot_contact": contact_targets.contact.contiguous(),
                "foot_contact_contract_version": BUMI_CONTACT_CONTRACT_VERSION,
                "foot_contact_source": "derived_from_final_zup_gmr_qpos_fk_ground_zero",
                "foot_contact_ground_height_m": float(contact_targets.ground_height),
            }
            motion_destination = dataset_root / motion_relative
            motion_destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(motion_payload, motion_destination)
            materialization[
                materialize_once(feature_source, dataset_root / feature_relative, feature_sha)
            ] += 1
            materialization[
                materialize_once(audio_source, dataset_root / audio_relative, audio_sha)
            ] += 1
            row = {
                "sample_id": sample_id,
                "sequence_id": pair["sequence_id"],
                "music_group_id": pair["music_group_id"],
                "audio_key": pair["audio_key"],
                "dataset": DATASET_SPECS[dataset]["contract_name"],
                "motion_path": motion_relative.as_posix(),
                "music_feature_path": feature_relative.as_posix(),
                "audio_path": audio_relative.as_posix(),
                "fps": 30,
                "num_frames": expected_frames,
                "split": split,
                "quality_accepted": True,
                "source_motion_sha256": source_motion_sha,
                "source_music_feature_sha256": feature_sha,
                "source_audio_sha256": audio_sha,
            }
            for field in (
                "person_id",
                "dance_style",
                "music_genre",
                "coarse_style",
                "fine_style",
                "song_name",
                "pair_id",
                "role",
                "song_id",
                "take_id",
                "group_id",
            ):
                if field in human:
                    row[field] = human[field]
            split_rows[dataset][split].append(row)
            counters[f"{dataset}:{split}"] += 1
            counters[f"{dataset}:total"] += 1

        root_orientation_by_dataset = {
            dataset: enforce_root_tilt_gate(
                np.concatenate(values),
                context=f"dataset={dataset}",
            )
            for dataset, values in root_tilt_by_dataset.items()
        }
        actual_splits = Counter()
        for dataset, by_split in split_rows.items():
            dataset_root = staging / DATASET_SPECS[dataset]["output"]
            for split, rows in by_split.items():
                rows.sort(key=lambda item: str(item["sample_id"]))
                _write_jsonl(dataset_root / "manifests" / f"{split}.jsonl", rows)
                actual_splits[split] += len(rows)
            info = {
                "contract_version": BUMI_MUSIC_CONTRACT_VERSION,
                "robot_name": "bumi",
                "dataset_name": DATASET_SPECS[dataset]["contract_name"],
                "source_dataset": dataset,
                "qpos_dim": 28,
                "joint_dim": 21,
                "joint_names": list(kinematics.joint_order),
                "quaternion_convention": "wxyz",
                "qpos_order": "mujoco_native",
                "fps": 30,
                "quality_filter_applied": True,
                "mjcf_sha256": mjcf_sha,
                "source_mjcf_sha256": mjcf_sha,
                "kinematics_sha256": kinematics.kinematics_sha256,
                "retarget_config_sha256": ik_sha,
                "quality_config_sha256": quality_sha,
                "ground_semantics": ground_semantics,
                "root_z_adjusted": root_z_adjusted,
                "root_z_adjustment_method": root_z_adjustment_method,
                "selection_contract_version": selection_info.get("contract_version"),
                "selection_info_sha256": selection_info_sha,
                "root_orientation_gate": {
                    "scope": "per_dataset_all_frames",
                    "max_dataset_median_deg": 45.0,
                    "max_dataset_p95_deg": 75.0,
                    "max_dataset_over_45deg_fraction": 0.5,
                    "statistics": root_orientation_by_dataset[dataset],
                    "all_sequences_recomputed_and_dataset_passed": True,
                },
                "split_counts": {name: len(rows) for name, rows in by_split.items()},
            }
            _write_json(dataset_root / "meta" / "dataset_info.json", info)

        if expected_splits is not None and dict(actual_splits) != dict(expected_splits):
            raise ValueError(
                f"formal split counts mismatch: expected={dict(expected_splits)}, "
                f"actual={dict(actual_splits)}"
            )
        report = {
            "status": "passed",
            "contract_version": BUMI_MUSIC_CONTRACT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_files_modified": False,
            "root_z_adjusted": root_z_adjusted,
            "root_z_adjustment_method": root_z_adjustment_method,
            "ground_semantics": ground_semantics,
            "selection_contract_version": selection_info.get("contract_version"),
            "selection_info_sha256": selection_info_sha,
            "source_mjcf_sha256": mjcf_sha,
            "retarget_config_sha256": ik_sha,
            "quality_config_sha256": quality_sha,
            "kinematics_sha256": kinematics.kinematics_sha256,
            "root_orientation_gate": {
                "scope": "per_dataset_all_frames",
                "max_dataset_median_deg": 45.0,
                "max_dataset_p95_deg": 75.0,
                "max_dataset_over_45deg_fraction": 0.5,
                "per_dataset_statistics": root_orientation_by_dataset,
                "all_datasets_recomputed_and_passed": True,
            },
            "selected_manifest_sha256": sha256_file(selected_manifest),
            "total_sequences": len(seen),
            "split_counts": dict(sorted(actual_splits.items())),
            "dataset_counts": {
                dataset: {
                    "total": counters[f"{dataset}:total"],
                    **{split: counters[f"{dataset}:{split}"] for split in ("train", "val", "test")},
                }
                for dataset in DATASET_SPECS
            },
            "unique_music_features": sum(
                len({row["music_feature_path"] for rows in values.values() for row in rows})
                for values in split_rows.values()
            ),
            "materialization": dict(materialization),
        }
        if expected_dataset_counts is not None:
            actual_dataset_counts = {
                dataset: int(report["dataset_counts"][dataset]["total"])
                for dataset in DATASET_SPECS
            }
            if actual_dataset_counts != dict(expected_dataset_counts):
                raise ValueError(
                    "formal dataset counts mismatch: "
                    f"expected={dict(expected_dataset_counts)}, "
                    f"actual={actual_dataset_counts}"
                )
        if expected_unique_music_features is not None and report["unique_music_features"] != int(
            expected_unique_music_features
        ):
            raise ValueError(
                "unique EDGE35 count mismatch: "
                f"expected={expected_unique_music_features}, "
                f"actual={report['unique_music_features']}"
            )
        for dataset in DATASET_SPECS:
            _write_json(
                staging / DATASET_SPECS[dataset]["output"] / "reports" / "conversion_report.json",
                {**report, "dataset": dataset, "dataset_counts": report["dataset_counts"][dataset]},
            )
        _write_json(staging / "conversion_report.json", report)
        os.replace(staging, output_root)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-root", required=True, type=Path)
    parser.add_argument("--human-root", action="append", required=True, type=_parse_mapping)
    parser.add_argument("--audio-root", action="append", required=True, type=_parse_mapping)
    parser.add_argument("--source-mjcf", required=True, type=Path)
    parser.add_argument("--ik-config", required=True, type=Path)
    parser.add_argument(
        "--quality-config",
        type=Path,
        default=REPO_ROOT / "configs" / "bumi" / "quality_filter_v1.yaml",
    )
    parser.add_argument("--kinematics", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-total", type=int, default=6610)
    parser.add_argument(
        "--expected-splits",
        default="train=5537,val=547,test=526",
        help="comma-separated SPLIT=COUNT; pass an empty string to disable",
    )
    parser.add_argument(
        "--expected-dataset-counts",
        default="aistpp=824,aioz_gdance=5608,finedance=111,compas3d=67",
    )
    parser.add_argument("--expected-unique-music-features", type=int, default=2548)
    args = parser.parse_args()
    expected_splits = None
    if args.expected_splits:
        expected_splits = {
            name: int(count)
            for name, count in (item.split("=", 1) for item in args.expected_splits.split(","))
        }
    expected_dataset_counts = None
    if args.expected_dataset_counts:
        expected_dataset_counts = {
            name: int(count)
            for name, count in (
                item.split("=", 1) for item in args.expected_dataset_counts.split(",")
            )
        }
    report = convert_datasets(
        selected_root=args.selected_root,
        human_roots=_mapping(args.human_root, "--human-root"),
        audio_roots=_mapping(args.audio_root, "--audio-root"),
        source_mjcf=args.source_mjcf.expanduser().resolve(),
        ik_config=args.ik_config.expanduser().resolve(),
        quality_config=args.quality_config.expanduser().resolve(),
        kinematics_path=args.kinematics.expanduser().resolve(),
        output_root=args.output_root,
        expected_total=args.expected_total,
        expected_splits=expected_splits,
        expected_dataset_counts=expected_dataset_counts,
        expected_unique_music_features=args.expected_unique_music_features,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
