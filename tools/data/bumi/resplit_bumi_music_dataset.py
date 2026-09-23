#!/usr/bin/env python3
"""对已通过质量筛选的 BUMI 四库动作重新做 90/5/5 分组划分与独立发布。

本工具合并旧 train/val/test，只处理既有 PASS 动作，不重新重定向、改变 qpos、接触、
音乐特征、时间轴或地面语义。相同音乐标识、歌曲标题、源文件哈希、同一 AIOZ 原视频
及其多舞者片段通过并查集归为不可拆分的组；Mine 中明确的编号/版本后缀也保守归组。
每库分别以动作条数的 5%（四舍五入）作为 val/test 目标，以固定 seed 的子集和算法
选择完整组，余下为 train。比例在完整组约束下尽量接近目标，不靠重复或丢弃动作凑数。
跨库关联组统一保留在 train，并在报告中列出；划分只使用标识和条数，不读取模型分数。

发布采用新的同盘目录、payload 硬链接、独立 manifest/meta 和原子 rename。旧清单与
统计量完整归档到 provenance，不覆盖原数据。复用现有严格 Dataset validator 验证全部
动作，并调用既有 qpos30 统计工具只遍历新 train；不重新定义 batch 或动作表示。报告
记录原始归属、新归属、分组规则、文件指纹、统计量指纹及旧训练数据进入新留出集的数量。
旧 checkpoint 已见过的数据不能因重新命名 split 而成为严格未见样本，工具不会启动训练。
--plan-only 只读取清单并输出计划；正常发布在任一步失败时清理本次精确 staging 目录。
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VERSION = "genmo.bumi_music_grouped_resplit.v1"
SPLITS = ("train", "val", "test")
DATASETS = {
    "AIST++": "aistpp_bumi",
    "AIOZ-GDANCE": "aioz_gdance_bumi",
    "FineDance": "finedance_bumi",
    "Mine": "mine_bumi",
}
PATH_FIELDS = ("motion_path", "music_feature_path", "audio_path")
HASH_FIELDS = (
    "source_motion_sha256",
    "source_audio_sha256",
    "source_music_feature_sha256",
    "original_source_audio_sha256",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def read_source(root: Path) -> tuple[list[dict], dict, dict]:
    """严格读取四库和原清单指纹；拒绝重复、混入非 PASS 或归属错误。"""
    records, infos, fingerprints = [], {}, {}
    for name, dataset in DATASETS.items():
        info_path = root / name / "meta/dataset_info.json"
        info = json.loads(info_path.read_text())
        if info.get("quality_filter_applied") is not True:
            raise ValueError(f"{name}: input must be quality filtered")
        if info.get("quality_acceptance_policy") != "PASS_ONLY":
            raise ValueError(f"{name}: input must use PASS_ONLY")
        infos[name] = info
        fingerprints[str(info_path.relative_to(root))] = digest(info_path)
        seen = set()
        for split in SPLITS:
            path = root / name / "manifests" / f"{split}.jsonl"
            fingerprints[str(path.relative_to(root))] = digest(path)
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(rows) != info["split_counts"][split]:
                raise ValueError(f"{name}/{split}: metadata count mismatch")
            for row in rows:
                sid = row["sample_id"]
                if not isinstance(sid, str) or not sid or sid in seen:
                    raise ValueError(f"{name}: duplicate/invalid sample ID {sid!r}")
                seen.add(sid)
                if row["dataset"] != dataset or row["split"] != split:
                    raise ValueError(f"{name}/{sid}: dataset/split mismatch")
                if row.get("quality_accepted") is not True:
                    raise ValueError(f"{name}/{sid}: only PASS rows can be resplit")
                if row.get("resplit_provenance"):
                    raise ValueError("Input must be the original release, not an existing resplit")
                if int(row["fps"]) != 30 or int(row["num_frames"]) <= 0:
                    raise ValueError(f"{name}/{sid}: invalid time contract")
                for field in HASH_FIELDS[:3]:
                    if not re.fullmatch(r"[a-f0-9]{64}", row.get(field, "")):
                        raise ValueError(f"{name}/{sid}: invalid {field}")
                for field in ("sequence_id", "music_group_id", "audio_key"):
                    if not isinstance(row.get(field), str) or not row[field]:
                        raise ValueError(f"{name}/{sid}: missing {field}")
                records.append({"source": name, "row": row})
    records.sort(key=lambda item: (item["source"], item["row"]["sample_id"]))
    for path in sorted(root.glob("stats/*.json")):
        fingerprints[str(path.relative_to(root))] = digest(path)
    return records, infos, fingerprints


def title_key(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())


def grouping_keys(record: dict) -> list[tuple[str, str]]:
    """显式保守分组，不以不同裁剪的音频 SHA 不同推断它们是不同歌曲。"""
    name, row = record["source"], record["row"]
    keys = []
    for field in HASH_FIELDS:
        if row.get(field):
            # 原 WAV 与发布 WAV 的同一哈希也应跨字段建立关联。
            kind = "audio_sha256" if "audio" in field else field
            keys.append((kind, row[field]))
    for field in ("sequence_id", "music_group_id", "group_id", "audio_key"):
        if row.get(field):
            keys.append((f"{name}:{field}", row[field]))
    if row.get("song_name"):
        # FineDance 元数据中的纯数字歌名（例如 123）可能保存为 JSON number。
        title = str(row["song_name"])
        if name == "Mine":
            title = re.sub(r"-(?:complete|nutuan\d+)$", "", title, flags=re.I)
            title = re.sub(r"(?<=\D)\d+$", "", title)
        normalized = title_key(title)
        if normalized:
            keys.append(("song_title", normalized))
    if name == "AIOZ-GDANCE":
        # 上游名称为 11 位视频 ID + clip_index + start + end。
        match = re.fullmatch(r"([A-Za-z0-9_-]{11})_\d+_\d+_\d+", row["sequence_id"])
        if not match:
            raise ValueError(f"unrecognized AIOZ source video ID: {row['sequence_id']}")
        keys.append(("aioz_source_video", match[1]))
    return keys


def build_groups(records: list[dict]) -> list[dict]:
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners = {}
    for index, record in enumerate(records):
        for key in grouping_keys(record):
            if key in owners:
                parent[find(index)] = find(owners[key])
            else:
                owners[key] = index
    buckets = collections.defaultdict(list)
    for index, record in enumerate(records):
        buckets[find(index)].append(record)
    groups = []
    for rows in buckets.values():
        rows.sort(key=lambda r: (r["source"], r["row"]["sample_id"]))
        identity = json.dumps([(r["source"], r["row"]["sample_id"]) for r in rows])
        groups.append(
            {
                "id": hashlib.sha256(identity.encode()).hexdigest(),
                "records": rows,
                "sources": sorted({r["source"] for r in rows}),
            }
        )
    return sorted(groups, key=lambda g: g["id"])


def choose_holdout(groups: list[dict], target: int, rng: random.Random, reserve: int) -> set[str]:
    """固定随机顺序的子集和；条数最接近目标，至少给其他集合保留组。"""
    ordered = list(groups)
    rng.shuffle(ordered)
    if len(ordered) <= reserve:
        raise ValueError("Not enough independent groups for nonempty train/val/test")
    limit = min(
        sum(len(g["records"]) for g in ordered), target + max(len(g["records"]) for g in ordered)
    )
    reachable = {0: ()}
    for group in ordered:
        size = len(group["records"])
        for count, selected in list(reachable.items()):
            new = count + size
            if new > limit or len(selected) + 1 > len(ordered) - reserve:
                continue
            candidate = (*selected, group["id"])
            # 同样条数使用 seed 顺序首次到达的组合，不刻意偏向少数大视频。
            if new not in reachable:
                reachable[new] = candidate
    candidates = [count for count in reachable if count > 0]
    if not candidates:
        raise ValueError("No feasible nonempty grouped holdout")
    closest = min(candidates, key=lambda count: (abs(count - target), count))
    return set(reachable[closest])


def make_plan(records: list[dict], seed: int) -> tuple[list[dict], dict, dict]:
    groups = build_groups(records)
    assignment = {g["id"]: "train" for g in groups}
    counts = collections.Counter(r["source"] for r in records)
    targets = {}
    for name in DATASETS:
        holdout = max(1, math.floor(counts[name] * 0.05 + 0.5))
        targets[name] = {"train": counts[name] - 2 * holdout, "val": holdout, "test": holdout}
        candidates = [g for g in groups if g["sources"] == [name]]
        rng = random.Random(f"{seed}:{name}")
        for split, reserve in (("val", 2), ("test", 1)):
            selected = choose_holdout(candidates, holdout, rng, reserve)
            for group_id in selected:
                assignment[group_id] = split
            candidates = [g for g in candidates if g["id"] not in selected]
    per_source = {}
    for name in DATASETS:
        result = {}
        for split in SPLITS:
            selected = [
                r
                for g in groups
                if assignment[g["id"]] == split
                for r in g["records"]
                if r["source"] == name
            ]
            result[split] = {
                "sequences": len(selected),
                "frames": sum(r["row"]["num_frames"] for r in selected),
                "groups": sum(
                    assignment[g["id"]] == split and name in g["sources"] for g in groups
                ),
                "previous_train_sequences": sum(r["row"]["split"] == "train" for r in selected),
            }
        per_source[name] = result
    # 独立检查所有分组依据均只出现在一个 split。
    key_splits = collections.defaultdict(set)
    for group in groups:
        for record in group["records"]:
            for key in grouping_keys(record):
                key_splits[key].add(assignment[group["id"]])
    if any(len(splits) != 1 for splits in key_splits.values()):
        raise AssertionError("group leakage in split plan")
    report = {
        "contract_version": VERSION,
        "seed": seed,
        "requested_ratios": {"train": 0.9, "val": 0.05, "test": 0.05},
        "ratio_unit": "motion_sequences_per_source_before_windowing",
        "rounding": "val_and_test_each_round_half_up_5_percent_train_remainder",
        "allocation": "seeded_subset_sum_nearest_sequence_count_whole_groups",
        "grouping": [
            "source_hashes_global",
            "sequence_music_group_audio_key_within_source",
            "normalized_song_title_global",
            "aioz_original_video",
            "mine_version_suffix_family",
        ],
        "cross_source_policy": "keep_connected_cross_source_groups_in_train",
        "cross_source_groups": [
            {"id": g["id"], "sources": g["sources"], "sequences": len(g["records"])}
            for g in groups
            if len(g["sources"]) > 1
        ],
        "targets": targets,
        "datasets": per_source,
        "total_sequences": len(records),
        "total_groups": len(groups),
        "counts": {s: sum(per_source[n][s]["sequences"] for n in DATASETS) for s in SPLITS},
        "group_key_cross_split_overlaps": 0,
        "coverage": "every_source_record_exactly_once",
        "unverified_identity_scope": "unknown_song_aliases_or_audio_similarity_not_acoustically_audited",
        "old_checkpoint_holdout_status": "new_holdouts_are_not_unseen_if_old_checkpoint_used_original_train",
    }
    return groups, assignment, report


def materialize(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def populate(root: Path, staging: Path, records, infos, fingerprints, groups, assignment) -> dict:
    """在 staging 中物化独立清单；文件保持原字节，源归属另外记录。"""
    rows_by_split = {(name, split): [] for name in DATASETS for split in SPLITS}
    membership = {}
    for group in groups:
        for record in group["records"]:
            membership[(record["source"], record["row"]["sample_id"])] = (
                group["id"],
                assignment[group["id"]],
            )
    linked, copied, artifacts = 0, 0, {}
    for record in records:
        name, original = record["source"], record["row"]
        group_id, split = membership[(name, original["sample_id"])]
        row = dict(original)
        row["split"] = split
        row["resplit_provenance"] = {
            "contract_version": VERSION,
            "original_split": original["split"],
            "group_id": group_id,
        }
        rows_by_split[(name, split)].append(row)
        for field in PATH_FIELDS:
            relative = Path(original[field])
            source_base = (root / name).resolve()
            source = (source_base / relative).resolve()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not source.is_relative_to(source_base)
                or not source.is_file()
            ):
                raise ValueError(f"invalid artifact path: {name}/{relative}")
            destination = staging / name / relative
            if not destination.exists():
                materialize(source, destination)
                same_inode = os.path.samefile(source, destination)
                linked += int(same_inode)
                copied += int(not same_inode)
                sha = digest(source)
                if not same_inode and digest(destination) != sha:
                    raise ValueError(f"copy checksum mismatch: {destination}")
                artifacts[str(destination.relative_to(staging))] = sha
    for name in DATASETS:
        info = dict(infos[name])
        info["split_counts"] = {split: len(rows_by_split[name, split]) for split in SPLITS}
        info["resplit_provenance"] = {
            "contract_version": VERSION,
            "source_root": str(root),
            "original_split_counts": infos[name]["split_counts"],
        }
        write_json(staging / name / "meta/dataset_info.json", info)
        for split in SPLITS:
            path = staging / name / "manifests" / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in rows_by_split[name, split]
                )
            )
    for relative in fingerprints:
        destination = staging / "provenance/original_release" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    for source in [*root.glob("*.json"), *root.glob("stats/*.json")]:
        destination = staging / "provenance/original_release" / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write_json(staging / "provenance/payload_sha256.json", artifacts)
    return {"hardlinked_files": linked, "copied_files": copied, "payload_files": len(artifacts)}


def validate_and_compute_stats(staging: Path, kinematics: Path) -> dict:
    """使用现有 Dataset/validator 和 train-only 统计工具，不改变旧检查规则。"""
    environment = dict(
        os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    validation = {}
    for name, dataset in DATASETS.items():
        print(f"validate all splits: {name}", file=sys.stderr, flush=True)
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools/data/bumi/validate_bumi_music_dataset.py"),
                "--root",
                str(staging / name),
                "--dataset-name",
                dataset,
                "--kinematics",
                str(kinematics),
                "--joint-limit-tolerance",
                "0.0001",
            ],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        validation[name] = json.loads(result.stdout)
        # 报告中的相对位置不绑定稍后被原子重命名的 staging 名。
        validation[name]["root"] = name
    stats = staging / "stats/qpos30_train_stats.json"
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/data/bumi/compute_bumi_30d_stats.py"),
        "--kinematics",
        str(kinematics),
        "--output",
        str(stats),
        "--joint-limit-tolerance",
        "0.0001",
    ]
    for name, dataset in DATASETS.items():
        command.extend(["--dataset", f"{dataset}={staging / name}"])
    print("compute qpos30 stats from NEW train only", file=sys.stderr, flush=True)
    subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
    values = json.loads(stats.read_text())
    for name, dataset in DATASETS.items():
        fingerprint = values["dataset_fingerprints"][dataset]
        if fingerprint["train_manifest_sha256"] != digest(staging / name / "manifests/train.jsonl"):
            raise AssertionError("stats manifest fingerprint mismatch")
    return {
        "strict_dataset_validation": validation,
        "qpos30_stats_sha256": digest(stats),
        "qpos30_stats_scope": "new_train_only_existing_qpos30_v3_tool",
    }


def publish(source_root: Path, output_root: Path, kinematics: Path, seed: int) -> dict:
    root, output = source_root.resolve(), output_root.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing release: {output}")
    if output.resolve().is_relative_to(root):
        raise ValueError("Output must be separate from the original release")
    records, infos, fingerprints = read_source(root)
    groups, assignment, report = make_plan(records, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        report["materialization"] = populate(
            root, staging, records, infos, fingerprints, groups, assignment
        )
        report.update(validate_and_compute_stats(staging, kinematics))
        for relative, sha in fingerprints.items():
            if digest(root / relative) != sha:
                raise ValueError(f"Source changed during resplit: {relative}")
        report["source_root"] = str(root)
        report["output_root"] = str(output)
        report["source_fingerprints"] = fingerprints
        report["source_manifests_unchanged"] = True
        report["output_fingerprints"] = {
            str(p.relative_to(staging)): digest(p)
            for p in sorted(staging.glob("*/manifests/*.jsonl"))
        }
        report["status"] = "published_and_strictly_validated"
        report["ground_semantics_preserved"] = {
            name: info["ground_semantics"] for name, info in infos.items()
        }
        report["training_started"] = False
        write_json(staging / "split_report.json", report)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--kinematics",
        type=Path,
        default=REPO_ROOT / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.plan_only:
        records, _infos, _fingerprints = read_source(args.source_root.resolve())
        _groups, _assignment, report = make_plan(records, args.seed)
    else:
        if args.output_root is None:
            parser.error("--output-root is required for publication")
        report = publish(args.source_root, args.output_root, args.kinematics, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
