#!/usr/bin/env python3
"""把 GMR 人工评分 1 的 BUMI3 发布包整理成 GENMO 可消费的受审计选择根。

这个工具位于 GMR 重定向与 ``genmo.bumi_music.v1`` 正式数据构建之间，解决两类不能
靠目录复制表达的契约问题。第一，人工评分表的 ``score=1`` 是用户指定的收录门，GMR
发布包中的 ``safety_overall`` 是关节位置/速度/加速度、上半身帧差和 Root Z 等硬安全
门；二者必须同时通过，但末端 FK 漂移的 fidelity 诊断只保留为可追溯字段，不能偷偷
改变人工收录集合。第二，GMR 的 Root Z 已按真实足底网格和接触状态做有界 QP 修正，
这里禁止再次平移到旧 body-origin 地面，否则会破坏已经验收的足底穿透与加速度结论。

工具会严格核对人工选择索引、GMR release audit、PKL SHA256 清单、每条 legacy pickle
的字段/关节顺序/帧数及嵌入式质量报告；所有集合、哈希和硬安全门一致后，才在 staging
目录中用硬链接（跨文件系统时复制）物化 ``motions/``，生成 ``selected.jsonl`` 和
``selection_info.json``，并快照 MJCF、IK 配置、质量规则及上游审计文件。最终目录通过
同文件系统原子替换发布，失败时不会留下可被下游误用的半成品。
"""

from __future__ import annotations

import argparse
import json
import math
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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.robots.bumi.legacy_motion import (  # noqa: E402
    LEGACY_BUMI_JOINT_ORDER,
    LEGACY_BUMI_MOTION_CONTRACT_VERSION,
    enforce_root_tilt_gate,
    load_legacy_bumi_motion,
    root_tilt_statistics,
    sha256_file,
)

SELECTION_CONTRACT_VERSION = "genmo.bumi_gmr_manual_q1_selection.v1"
EXPECTED_RELEASE_SCHEMA = "gmr_bumi3_manual_q1_release_audit_v1"
EXPECTED_SOURCE_MJCF_SHA256 = "482138b437dbdabd6171fa8d44b55db5d7125a228c95b69ce3d1159cafe8537c"
EXPECTED_DATASETS = ("aistpp", "aioz_gdance", "finedance", "compas3d")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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
                raise ValueError(f"{path}:{line_number}: expected JSON object")
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


def _digest(value: Any, label: str) -> str:
    result = str(value)
    if _SHA256.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest, got {value!r}")
    return result


def _safe_relative(value: str, *, suffix: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"manifest path must stay relative: {value!r}")
    if len(relative.parts) != 2 or relative.parts[0] not in EXPECTED_DATASETS:
        raise ValueError(f"manifest path must be DATASET/BASENAME{suffix}: {value!r}")
    if relative.suffix != suffix or relative.name in {"", suffix}:
        raise ValueError(f"manifest path must end in {suffix}: {value!r}")
    return relative


def _read_sha_manifest(
    path: Path, *, suffix: str, strip_prefix: str | None = None
) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_number}: invalid sha256 manifest row")
        digest = _digest(parts[0], f"{path}:{line_number}")
        raw_relative = Path(parts[1].lstrip("*"))
        if strip_prefix is not None:
            if not raw_relative.parts or raw_relative.parts[0] != strip_prefix:
                raise ValueError(f"{path}:{line_number}: expected {strip_prefix!r} path prefix")
            raw_relative = Path(*raw_relative.parts[1:])
        relative = _safe_relative(raw_relative.as_posix(), suffix=suffix).as_posix()
        if relative in result:
            raise ValueError(f"{path}:{line_number}: duplicate path {relative}")
        result[relative] = digest
    if not result:
        raise ValueError(f"empty sha256 manifest: {path}")
    return result


def _load_quality_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"quality config must be a mapping: {path}")
    if value.get("contract_version") != "genmo.bumi_gmr_manual_q1_quality_gate.v1":
        raise ValueError(f"unexpected GMR manual-q1 quality config contract: {path}")
    return value


def _expected_counts(config: Mapping[str, Any], key: str) -> dict[str, int]:
    value = config.get("selection", {}).get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"quality config selection.{key} must be a mapping")
    return {str(name): int(count) for name, count in value.items()}


def _materialize(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _verify_release_contract(
    release: Mapping[str, Any],
    *,
    expected_total: int,
    expected_frames: int,
    expected_dataset_counts: Mapping[str, int],
    pkl_manifest_sha256: str,
    expected_schema: str = EXPECTED_RELEASE_SCHEMA,
) -> None:
    if release.get("schema") != expected_schema:
        raise ValueError(f"unexpected GMR release schema={release.get('schema')!r}")
    selection = release.get("selection")
    dataset_contract = release.get("dataset_contract")
    acceptance = release.get("acceptance_counts")
    if not all(isinstance(value, Mapping) for value in (selection, dataset_contract, acceptance)):
        raise ValueError(
            "GMR release audit is missing selection/dataset_contract/acceptance_counts"
        )
    checks = {
        "selected_unique_clips": (selection.get("selected_unique_clips"), expected_total),
        "selected_total_frames": (selection.get("selected_total_frames"), expected_frames),
        "output_pkl_count": (dataset_contract.get("output_pkl_count"), expected_total),
        "loaded_total_frames": (dataset_contract.get("loaded_total_frames"), expected_frames),
        "safety_overall": (acceptance.get("safety_overall"), expected_total),
        "source_sha256_verified": (
            selection.get("source_sha256_verified"),
            expected_total,
        ),
    }
    for label, (actual, expected) in checks.items():
        if int(actual) != int(expected):
            raise ValueError(f"release {label} mismatch: expected={expected}, actual={actual}")
    if float(dataset_contract.get("fps")) != 30.0:
        raise ValueError("GMR release dataset is not 30 FPS")
    if tuple(map(str, dataset_contract.get("dof_names", ()))) != LEGACY_BUMI_JOINT_ORDER:
        raise ValueError("GMR release joint order does not match GENMO legacy adapter")
    if _digest(dataset_contract.get("pkl_manifest_sha256"), "release pkl manifest") != (
        pkl_manifest_sha256
    ):
        raise ValueError("GMR release pkl manifest SHA mismatch")
    per_dataset = release.get("per_dataset")
    if not isinstance(per_dataset, Mapping):
        raise ValueError("GMR release per_dataset is missing")
    actual_counts = {
        name: int(per_dataset.get(name, {}).get("clips", -1)) for name in EXPECTED_DATASETS
    }
    if actual_counts != dict(expected_dataset_counts):
        raise ValueError(
            f"GMR release dataset counts mismatch: expected={dict(expected_dataset_counts)}, "
            f"actual={actual_counts}"
        )


def prepare_selected_root(
    *,
    motion_root: Path,
    selection_index: Path,
    selection_sha256_manifest: Path,
    release_audit: Path,
    pkl_sha256_manifest: Path,
    source_mjcf: Path,
    retarget_config: Path,
    quality_config: Path,
    output_root: Path,
    expected_total: int = 3162,
) -> dict[str, Any]:
    """完整验证 GMR 发布包，并原子生成 GENMO selected root。"""

    motion_root = motion_root.expanduser().resolve(strict=True)
    selection_index = selection_index.expanduser().resolve(strict=True)
    selection_sha256_manifest = selection_sha256_manifest.expanduser().resolve(strict=True)
    release_audit = release_audit.expanduser().resolve(strict=True)
    pkl_sha256_manifest = pkl_sha256_manifest.expanduser().resolve(strict=True)
    source_mjcf = source_mjcf.expanduser().resolve(strict=True)
    retarget_config = retarget_config.expanduser().resolve(strict=True)
    quality_config = quality_config.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite selected root: {output_root}")

    config = _load_quality_config(quality_config)
    if not motion_root.is_dir():
        raise NotADirectoryError(motion_root)
    source_sha = sha256_file(source_mjcf)
    configured_source_sha = _digest(
        config.get("source", {}).get("mjcf_sha256"), "quality config source MJCF"
    )
    if source_sha != EXPECTED_SOURCE_MJCF_SHA256 or source_sha != configured_source_sha:
        raise ValueError(
            "source MJCF SHA mismatch: "
            f"expected={EXPECTED_SOURCE_MJCF_SHA256}, config={configured_source_sha}, actual={source_sha}"
        )
    if tuple(map(str, config.get("source", {}).get("joint_order", ()))) != (
        LEGACY_BUMI_JOINT_ORDER
    ):
        raise ValueError("quality config joint_order does not match the legacy BUMI adapter")
    required_flags = tuple(map(str, config.get("gmr_acceptance", {}).get("required_flags", ())))
    if not required_flags or "safety_overall" not in required_flags:
        raise ValueError("quality config must require explicit GMR safety_overall")
    root_policy = config.get("root_z")
    if not isinstance(root_policy, Mapping):
        raise ValueError("quality config root_z policy is missing")
    ground_semantics = str(root_policy.get("ground_semantics", ""))
    root_method = str(root_policy.get("required_method", ""))
    max_penetration = float(root_policy.get("max_sole_penetration_m"))
    if (
        ground_semantics != "gmr_foot_sole_ground_zero_v1"
        or root_method != "foot_contact_bounded_qp"
    ):
        raise ValueError("quality config must bind the GMR foot-sole bounded-QP Root Z contract")
    if not math.isfinite(max_penetration) or max_penetration <= 0.0:
        raise ValueError("quality config max_sole_penetration_m must be finite and positive")

    expected_dataset_counts = _expected_counts(config, "expected_dataset_counts")
    expected_split_counts = _expected_counts(config, "expected_split_counts")
    if int(config.get("selection", {}).get("expected_total", -1)) != int(expected_total):
        raise ValueError("quality config selection.expected_total differs from --expected-total")
    if set(expected_dataset_counts) != set(EXPECTED_DATASETS):
        raise ValueError(
            "quality config expected_dataset_counts does not cover exactly four corpora"
        )
    if sum(expected_dataset_counts.values()) != int(expected_total):
        raise ValueError("quality config dataset counts do not sum to --expected-total")

    index_rows = _read_jsonl(selection_index)
    selection_hashes = _read_sha_manifest(
        selection_sha256_manifest, suffix=".npz", strip_prefix="motions"
    )
    if len(index_rows) != int(expected_total):
        raise ValueError(
            f"manual selection count mismatch: expected={expected_total}, actual={len(index_rows)}"
        )
    index: dict[str, dict[str, Any]] = {}
    index_dataset_counts: Counter[str] = Counter()
    index_split_counts: Counter[str] = Counter()
    expected_frames = 0
    for row in index_rows:
        dataset = str(row.get("dataset", ""))
        sample_id = str(row.get("sample_id", ""))
        key = f"{dataset}/{sample_id}"
        if dataset not in EXPECTED_DATASETS or not sample_id or Path(sample_id).name != sample_id:
            raise ValueError(f"invalid manual selection identity: {key!r}")
        if key in index:
            raise ValueError(f"duplicate manual selection identity: {key}")
        if int(row.get("score", -1)) != 1:
            raise ValueError(f"{key}: manual score gate did not pass")
        actual_npz_sha = _digest(row.get("actual_sha256"), f"{key} manual source NPZ")
        if selection_hashes.get(f"{key}.npz") != actual_npz_sha:
            raise ValueError(f"{key}: manual package NPZ SHA manifest mismatch")
        frames = int(row.get("num_frames", 0))
        split = str(row.get("split", ""))
        if frames <= 0 or split not in {"train", "val", "test"}:
            raise ValueError(f"{key}: invalid num_frames/split")
        index[key] = dict(row)
        expected_frames += frames
        index_dataset_counts[dataset] += 1
        index_split_counts[split] += 1
    actual_dataset_counts = {name: index_dataset_counts[name] for name in expected_dataset_counts}
    if actual_dataset_counts != expected_dataset_counts:
        raise ValueError(
            f"manual dataset counts mismatch: expected={expected_dataset_counts}, "
            f"actual={actual_dataset_counts}"
        )
    actual_split_counts = {name: index_split_counts[name] for name in expected_split_counts}
    if actual_split_counts != expected_split_counts:
        raise ValueError(
            f"manual split counts mismatch: expected={expected_split_counts}, "
            f"actual={actual_split_counts}"
        )
    expected_npz_paths = {f"{key}.npz" for key in index}
    if set(selection_hashes) != expected_npz_paths:
        raise ValueError("manual package NPZ SHA manifest set differs from selection index")

    pkl_manifest_sha = sha256_file(pkl_sha256_manifest)
    pkl_hashes = _read_sha_manifest(pkl_sha256_manifest, suffix=".pkl")
    expected_pkl_paths = {f"{key}.pkl" for key in index}
    if set(pkl_hashes) != expected_pkl_paths:
        raise ValueError(
            "GMR PKL manifest set differs from manual selection: "
            f"missing={sorted(expected_pkl_paths - set(pkl_hashes))[:10]}, "
            f"extra={sorted(set(pkl_hashes) - expected_pkl_paths)[:10]}"
        )

    release = _read_json(release_audit)
    _verify_release_contract(
        release,
        expected_total=expected_total,
        expected_frames=expected_frames,
        expected_dataset_counts=expected_dataset_counts,
        pkl_manifest_sha256=pkl_manifest_sha,
        expected_schema=str(
            config.get("source", {}).get("release_audit_schema", EXPECTED_RELEASE_SCHEMA)
        ),
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    selected_rows: list[dict[str, Any]] = []
    materialization: Counter[str] = Counter()
    fidelity_counts: Counter[str] = Counter()
    root_tilt_by_dataset: dict[str, list[np.ndarray]] = {
        name: [] for name in EXPECTED_DATASETS
    }
    try:
        for relative_text in sorted(pkl_hashes):
            relative = _safe_relative(relative_text, suffix=".pkl")
            key = relative.with_suffix("").as_posix()
            selected = index[key]
            source = (motion_root / relative).resolve()
            try:
                source.relative_to(motion_root)
            except ValueError as exc:
                raise ValueError(f"GMR motion escapes motion root: {source}") from exc
            if not source.is_file():
                raise FileNotFoundError(source)
            actual_sha = sha256_file(source)
            if actual_sha != pkl_hashes[relative_text]:
                raise ValueError(f"{key}: GMR PKL SHA mismatch")
            motion = load_legacy_bumi_motion(source, expected_fps=30)
            if motion.declared_dof_names != LEGACY_BUMI_JOINT_ORDER:
                raise ValueError(f"{key}: production GMR PKL must explicitly declare dof_names")
            if motion.num_frames != int(selected["num_frames"]):
                raise ValueError(
                    f"{key}: frame mismatch index={selected['num_frames']}, pkl={motion.num_frames}"
                )
            direct_tilt = motion.root_tilt_degrees()
            direct_root_orientation = root_tilt_statistics(direct_tilt)
            root_tilt_by_dataset[relative.parts[0]].append(direct_tilt)
            quality = motion.quality
            if not isinstance(quality, Mapping):
                raise ValueError(f"{key}: missing embedded GMR quality report")
            if quality.get("pipeline_version") != config.get("source", {}).get("pipeline_version"):
                raise ValueError(f"{key}: unexpected GMR pipeline version")
            if float(quality.get("aligned_fps")) != 30.0:
                raise ValueError(f"{key}: embedded quality is not aligned to 30 FPS")
            acceptance = quality.get("acceptance")
            if not isinstance(acceptance, Mapping):
                raise ValueError(f"{key}: embedded acceptance is missing")
            failed = [flag for flag in required_flags if acceptance.get(flag) is not True]
            if failed:
                raise ValueError(f"{key}: required GMR safety gates failed: {failed}")
            if quality.get("joint_limit_contract", {}).get("pass") is not True:
                raise ValueError(f"{key}: XML/URDF joint limit contract failed")
            if quality.get("trajectory", {}).get("constraint_pass") is not True:
                raise ValueError(f"{key}: bounded trajectory optimizer contract failed")
            root_height = quality.get("root_height")
            root_audit = quality.get("final_root_audit")
            if not isinstance(root_height, Mapping) or not isinstance(root_audit, Mapping):
                raise ValueError(f"{key}: Root Z diagnostics are missing")
            if root_height.get("method") != root_method or root_audit.get("finite") is not True:
                raise ValueError(f"{key}: Root Z method/audit gate failed")
            penetration = float(root_audit.get("max_sole_penetration"))
            if not math.isfinite(penetration) or penetration > max_penetration + 1.0e-9:
                raise ValueError(f"{key}: sole penetration {penetration} exceeds {max_penetration}")

            destination_relative = Path("motions") / relative
            materialization[_materialize(source, staging / destination_relative)] += 1
            fidelity = bool(acceptance.get("fidelity_overall"))
            fidelity_counts[str(fidelity).lower()] += 1
            selected_rows.append(
                {
                    "report_contract_version": SELECTION_CONTRACT_VERSION,
                    "source_motion_contract_version": LEGACY_BUMI_MOTION_CONTRACT_VERSION,
                    "dataset": relative.parts[0],
                    "sample_id": key,
                    "source_relative_path": relative.as_posix(),
                    "motion_path": destination_relative.as_posix(),
                    "source_sha256": actual_sha,
                    "source_mjcf_sha256": source_sha,
                    "quality_config_sha256": sha256_file(quality_config),
                    "status": "PASS",
                    "quality_status": "PASS",
                    "quality_accepted": True,
                    "reason_codes": [],
                    "manual_rating_score": 1,
                    "manual_source_npz_sha256": selected.get("actual_sha256"),
                    "split": selected["split"],
                    "num_frames": motion.num_frames,
                    "gmr_safety_overall": True,
                    "gmr_fidelity_overall": fidelity,
                    "gmr_root_z_method": root_method,
                    "gmr_max_sole_penetration_m": penetration,
                    "gmr_root_orientation": direct_root_orientation,
                }
            )

        root_orientation_by_dataset = {
            dataset: enforce_root_tilt_gate(
                np.concatenate(values),
                context=f"dataset={dataset}",
            )
            for dataset, values in root_tilt_by_dataset.items()
        }
        selected_manifest = staging / "manifests" / "selected.jsonl"
        _write_jsonl(selected_manifest, selected_rows)
        snapshots = {
            "source_mjcf.snapshot.xml": source_mjcf,
            "retarget_config.snapshot.json": retarget_config,
            "quality_config.snapshot.yaml": quality_config,
            "upstream_release_audit.json": release_audit,
            "upstream_selection_index.jsonl": selection_index,
            "upstream_selection_sha256.txt": selection_sha256_manifest,
            "upstream_pkl_sha256.txt": pkl_sha256_manifest,
        }
        for name, source in snapshots.items():
            destination = staging / "meta" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        info = {
            "contract_version": SELECTION_CONTRACT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_rule": "manual_rating_score_equals_1_and_gmr_safety_overall",
            "fidelity_is_diagnostic_only": True,
            "source_motion_contract_version": LEGACY_BUMI_MOTION_CONTRACT_VERSION,
            "source_pipeline_version": config["source"]["pipeline_version"],
            "ground_semantics": ground_semantics,
            "root_z_adjusted": True,
            "root_z_adjustment_method": root_method,
            "second_root_z_adjustment_applied": False,
            "max_sole_penetration_m": max_penetration,
            "root_orientation_gate": {
                "scope": "per_dataset_all_frames",
                "max_dataset_median_deg": 45.0,
                "max_dataset_p95_deg": 75.0,
                "max_dataset_over_45deg_fraction": 0.5,
                "per_dataset_statistics": root_orientation_by_dataset,
                "all_datasets_recomputed_and_passed": True,
            },
            "total_sequences": len(selected_rows),
            "total_frames": expected_frames,
            "total_hours_at_30fps": expected_frames / 30.0 / 3600.0,
            "dataset_counts": actual_dataset_counts,
            "split_counts": actual_split_counts,
            "fidelity_counts": dict(fidelity_counts),
            "materialization": dict(materialization),
            "source_mjcf_sha256": source_sha,
            "retarget_config_sha256": sha256_file(retarget_config),
            "quality_config_sha256": sha256_file(quality_config),
            "release_audit_sha256": sha256_file(release_audit),
            "selection_index_sha256": sha256_file(selection_index),
            "selection_sha256_manifest_sha256": sha256_file(selection_sha256_manifest),
            "upstream_pkl_manifest_sha256": pkl_manifest_sha,
            "selected_manifest_sha256": sha256_file(selected_manifest),
            "source_files_modified": False,
        }
        _write_json(staging / "meta" / "selection_info.json", info)
        os.replace(staging, output_root)
        return info
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", required=True, type=Path)
    parser.add_argument("--selection-index", required=True, type=Path)
    parser.add_argument("--selection-sha256-manifest", required=True, type=Path)
    parser.add_argument("--release-audit", required=True, type=Path)
    parser.add_argument("--pkl-sha256-manifest", required=True, type=Path)
    parser.add_argument("--source-mjcf", required=True, type=Path)
    parser.add_argument("--retarget-config", required=True, type=Path)
    parser.add_argument(
        "--quality-config",
        type=Path,
        default=REPO_ROOT / "configs" / "bumi" / "quality_filter_gmr_manual_q1_v3.yaml",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-total", type=int, default=3162)
    args = parser.parse_args()
    report = prepare_selected_root(
        motion_root=args.motion_root,
        selection_index=args.selection_index,
        selection_sha256_manifest=args.selection_sha256_manifest,
        release_audit=args.release_audit,
        pkl_sha256_manifest=args.pkl_sha256_manifest,
        source_mjcf=args.source_mjcf,
        retarget_config=args.retarget_config,
        quality_config=args.quality_config,
        output_root=args.output_root,
        expected_total=args.expected_total,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
