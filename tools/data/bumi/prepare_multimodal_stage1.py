#!/usr/bin/env python3
"""为音乐/文本联合 Stage1 发布跨库 split 白名单与仅训练集归一化统计。

读取显式 YAML 中三个 split 的正式 30 Hz GENMO release，以来源母 ID、镜像归一化 ID、
原始动作哈希及音乐音频/分组建立连通组。跨 split 冲突采用 test > val > train，只删除
较低优先级的可采样索引，不修改、重标或复制已有 motion/text/music payload。来源提供的
别名可通过 source_identity_aliases 显式合并；缺少身份依据的条目报错而非猜测。

统计只遍历最终 train 白名单，复用 Stage1 完整序列窗口和逐坐标有效 mask，排除 padding
及没有右邻真实帧的 XY delta。文本按真实 caption 区间产生确定窗口，不加载 T5；地面
始终沿用原完整序列 provenance。工具只向一个原先不存在的输出目录写 split_allowlist、
审计报告、qpos30 stats；不修改源 release、不访问服务器、不启动训练或占用 GPU。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.closedloop.stage1_multimodal_dataset import (  # noqa: E402
    SPLIT_ALLOWLIST_SCHEMA,
    _AllowedSamples,
    _sampling_weights,
    make_multimodal_source,
    multimodal_data_identity,
)
from gem.datasets.pure_motion.bumi_text import caption_intervals  # noqa: E402
from gem.datasets.text_source_contract import mirror_base_id  # noqa: E402
from gem.robots.bumi.endecoder import STATS_CONTRACT_VERSION  # noqa: E402
from gem.robots.bumi.feature_codec import (  # noqa: E402
    BUMI_ANCHOR_MODE,
    BUMI_FEATURE_SLICES,
    BUMI_REPRESENTATION_CONTRACT_VERSION,
)


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _canonical(value, aliases):
    value = mirror_base_id(str(value).strip())
    seen = set()
    while value in aliases:
        if value in seen:
            raise ValueError("source_identity_aliases contains a cycle")
        seen.add(value)
        value = mirror_base_id(str(aliases[value]))
    return value


def _record_identities(source, index, aliases, unverified_lineage=None):
    row = source.rows[index]
    identities = set()
    if source.modality == "text":
        _, _, catalog_row = source.release.index[index]
        # manifest 新版索引提供母来源；其他原始哈希可能仅位于 record provenance。
        record = source.release.read_record(index)
        provenance = record["provenance"]
        if (
            provenance.get("cross_dataset_lineage_verified") is not True
            and unverified_lineage is not None
        ):
            unverified_lineage.append(
                {
                    "split": source.split,
                    "source": source.source_name,
                    "record_key": row["sample_id"],
                    "published_verified_flag": provenance.get("cross_dataset_lineage_verified"),
                }
            )
        canonical = catalog_row.get("canonical_source_id") or provenance.get("canonical_source_id")
        if not canonical:
            raise ValueError(f"text record lacks canonical_source_id: {row['sample_id']}")
        identities.add("canonical:" + _canonical(canonical, aliases))
        for key in ("source_sha256", "source_motion_sha256", "original_motion_sha256"):
            digest = provenance.get(key) or record.get(key)
            if digest:
                identities.add("motion_sha256:" + str(digest))
        # 发布时保留的原始人动作与机器人 qpos 哈希可补充跨库精确重复证据。
        human_sha = record.get("quality_provenance", {}).get("human_sha256")
        if human_sha:
            identities.add("human_payload_sha256:" + str(human_sha))
        if record.get("source_qpos_sha256"):
            identities.add("robot_payload_sha256:" + str(record["source_qpos_sha256"]))
    else:
        for key in ("canonical_source_id", "source_canonical_id"):
            if row.get(key):
                identities.add("canonical:" + _canonical(row[key], aliases))
        if row.get("source_motion_sha256"):
            identities.add("motion_sha256:" + str(row["source_motion_sha256"]))
        for key in ("source_audio_sha256", "original_source_audio_sha256"):
            if row.get(key):
                identities.add("audio_sha256:" + str(row[key]))
        for key in ("original_source_motion_sha256", "original_motion_sha256"):
            if row.get(key):
                identities.add("motion_sha256:" + str(row[key]))
        resplit_group = row.get("resplit_provenance", {}).get("group_id")
        if resplit_group:
            identities.add("published_resplit_group:" + str(resplit_group))
        for key in ("music_group_id", "sequence_id"):
            if row.get(key):
                identities.add(
                    f"music_group:{source.dataset_name}:{key}:" + _canonical(row[key], aliases)
                )
    if not identities:
        raise ValueError(
            f"record lacks auditable source identity: {source.source_name}/{row['sample_id']}"
        )
    return sorted(identities)


def audit_split_overlap(data_config):
    """返回可复核的 allowlist；相同来源绝不从 train 迁移生成伪造的 val/test 记录。"""
    data = dict(_plain(data_config))
    if set(data["datasets"]) != {"train", "val", "test"}:
        raise ValueError("split audit requires explicit train, val and test configurations")
    identity_data = {**data, "split_allowlist": None}
    identity = multimodal_data_identity(identity_data)
    aliases = data.get("source_identity_aliases", {})
    records, parents, owner = [], [], {}
    source_boundaries = {}
    unverified_lineage = []

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for split in ("test", "val", "train"):
        for name, entry in data["datasets"][split].items():
            source = make_multimodal_source(entry, source_name=name, split=split, evaluation=True)
            report_path = source.root / "build_report.json"
            if report_path.is_file():
                release_report = json.loads(report_path.read_text())
                # 原 release 标注的未知关系必须原样披露，canonical 存在不等于跨库血缘已证实。
                source_boundaries[f"{split}/{name}"] = {
                    key: value
                    for key, value in release_report.items()
                    if "unverified" in key or "lineage" in key
                }
            for row_index, key in enumerate(source.record_keys):
                tokens = _record_identities(source, row_index, aliases, unverified_lineage)
                index = len(records)
                parents.append(index)
                records.append(
                    {"split": split, "source": name, "record_key": key, "identities": tokens}
                )
                for token in tokens:
                    if token in owner:
                        parents[find(index)] = find(owner[token])
                    owner[token] = index
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[find(i)].append(record)
    priorities = {"train": 0, "val": 1, "test": 2}
    entries = {
        f"{split}/{name}": [] for split, sources in data["datasets"].items() for name in sources
    }
    excluded, conflicts = [], []
    for group in groups.values():
        chosen = max((record["split"] for record in group), key=priorities.__getitem__)
        group_id = hashlib.sha256(
            "\n".join(sorted({i for r in group for i in r["identities"]})).encode()
        ).hexdigest()
        if len({record["split"] for record in group}) > 1:
            conflicts.append({"group_id": group_id, "retained_split": chosen, "records": group})
        for record in group:
            if record["split"] == chosen:
                entries[f"{record['split']}/{record['source']}"].append(record["record_key"])
            else:
                excluded.append({**record, "retained_split": chosen, "group_id": group_id})
    payload = {
        "schema": SPLIT_ALLOWLIST_SCHEMA,
        "policy": "test_gt_val_gt_train_exclusion_only",
        "manifests": identity["manifests"],
        "entries": entries,
        "excluded": excluded,
        "source_identity_aliases": aliases,
    }
    report = {
        "schema": "genmo.bumi_multimodal.split_audit.v1",
        "record_count": len(records),
        "source_group_count": len(groups),
        "conflicting_group_count": len(conflicts),
        "conflicts": conflicts,
        "excluded_count": len(excluded),
        "accepted_counts": {key: len(value) for key, value in entries.items()},
        "identity_boundary": "exact provided source IDs/hashes and declared aliases; cross-dataset lineage remains unverified unless backed by shared hashes or explicit aliases",
        "source_release_boundaries": source_boundaries,
        "unverified_cross_dataset_lineage": unverified_lineage,
    }
    return payload, report


class MaskedMoments:
    """逐坐标加权稳定累计，单窗口各坐标有效帧总权重相同，padding 不参与。"""

    def __init__(self, dimension=30):
        self.count = torch.zeros(dimension, dtype=torch.float64)
        self.mass = torch.zeros_like(self.count)
        self.mean = torch.zeros_like(self.count)
        self.m2 = torch.zeros_like(self.count)

    def update(self, values, mask, *, sample_weight=None):
        values, mask = values.double(), mask.bool()
        if values.shape != mask.shape or not torch.isfinite(values[mask]).all():
            raise ValueError("invalid masked train statistics input")
        count = mask.sum(0).double()
        if sample_weight is None:
            mass = count
            per_frame = torch.ones_like(count)
        else:
            weight = float(sample_weight)
            if not 0 < weight < float("inf"):
                raise ValueError("sample weight must be finite and positive")
            mass = (count > 0).double() * weight
            per_frame = mass / count.clamp_min(1)
        batch_mean = torch.where(mask, values, 0).sum(0) / count.clamp_min(1)
        batch_m2 = torch.where(mask, (values - batch_mean).square(), 0).sum(0) * per_frame
        total = self.mass + mass
        delta = batch_mean - self.mean
        self.mean += delta * mass / total.clamp_min(1e-300)
        self.m2 += batch_m2 + delta.square() * self.mass * mass / total.clamp_min(1e-300)
        self.count += count
        self.mass = total

    def finalize(self):
        if (self.mass <= 0).any():
            raise ValueError("no valid train values in one or more qpos30 coordinates")
        return self.mean.float(), (self.m2 / self.mass).clamp_min(0).sqrt().clamp_min(1e-6).float()


def _window_starts(begin, end, frames):
    last = max(begin, end - frames)
    return sorted(set((*range(begin, last + 1, frames), last)))


def compute_train_statistics(data_config, allowlist):
    """训练白名单仅计算真实动作条件目标统计，T5/val/test 不参与拟合。"""
    data = _plain(data_config)
    current_identity = multimodal_data_identity({**data, "split_allowlist": None})
    if allowlist.get("manifests") != current_identity["manifests"]:
        raise ValueError("train statistics require an allowlist for the current manifests")
    moments = MaskedMoments()
    fingerprints = {}
    reference_kinematics = None
    window_count = 0
    source_datasets = []
    for name, entry in data["datasets"]["train"].items():
        # 保留训练 music duration-aware 索引才能与 sampler 的动作权重一致。
        source = make_multimodal_source(entry, source_name=name, split="train")
        allowed = set(allowlist["entries"][f"train/{name}"])
        source_datasets.append(_AllowedSamples(source, allowed))
    all_weights = _sampling_weights(source_datasets, data)
    source_offset = 0
    for selected in source_datasets:
        source, name = selected.source, selected.source_name
        source_weights = all_weights[source_offset : source_offset + len(selected)]
        source_offset += len(selected)
        record_weights = Counter()
        for local_index, weight in zip(selected.indices, source_weights):
            record_weights[source.idx2meta[local_index]] += weight
        kin = source.kinematics
        if (
            reference_kinematics is not None
            and kin.kinematics_sha256 != reference_kinematics.kinematics_sha256
        ):
            raise ValueError("all sources must share the same BUMI kinematics")
        reference_kinematics = kin
        frames_total, accepted = 0, 0
        for index, record_weight in record_weights.items():
            row = source.rows[index]
            if source.modality == "text":
                record = source.release.read_record(index)
                if source.release.require_temporal_annotations and (
                    record.get("text_annotation_scope") != "temporal"
                    or "caption_intervals" not in record
                ):
                    raise ValueError("BONES train stats require temporal annotations")
                sequence = source.reader.load_aligned_sequence(row, record=record)
                # 每个 record 的 caption 权重和为1；更多描述不会增加该母动作总权重。
                intervals = [tuple(pair) for pair in caption_intervals(record)]
            else:
                sequence = source.reader.load_aligned_sequence(row)
                intervals = [(0, len(sequence["qpos"]))]
            interval_counts = Counter(intervals)
            for (begin, end), frequency in interval_counts.items():
                starts = _window_starts(begin, end, source.motion_frames)
                window_weight = record_weight * frequency / (len(intervals) * len(starts))
                for decision in starts:
                    sample = source.get_sequence_window(sequence, decision, end_frame=end)
                    moments.update(
                        sample["target_qpos30"],
                        sample["target_qpos30_valid"],
                        sample_weight=window_weight,
                    )
                    window_count += 1
            frames_total += len(sequence["qpos"])
            accepted += 1
        fingerprints[name] = {
            **source.data_identity,
            "sequences": accepted,
            "source_frames": frames_total,
            "probability_mass": sum(source_weights),
        }
    mean, std = moments.finalize()
    kin = reference_kinematics
    return {
        "contract_version": STATS_CONTRACT_VERSION,
        "root_height_reference_m": float(source.codec.default_root_height),
        "representation_contract_version": BUMI_REPRESENTATION_CONTRACT_VERSION,
        "robot_name": "bumi",
        "feature_dim": 30,
        "split": "train",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feature_slices": {key: list(value) for key, value in BUMI_FEATURE_SLICES.items()},
        "joint_names": list(kin.joint_order),
        "anchor_mode": BUMI_ANCHOR_MODE,
        "quaternion_convention": "wxyz",
        "kinematics_sha256": kin.kinematics_sha256,
        "dataset_fingerprints": fingerprints,
        "num_windows": window_count,
        "valid_coordinate_counts": moments.count.long().tolist(),
        "stored_std_minimum": 1e-6,
        "training_clip_std_min": 0.01,
        "is_placeholder": False,
        "sampling_scope": "final_train_allowlist_modality_source_parent_motion_caption_window_weighted",
        "coordinate_probability_mass": moments.mass.tolist(),
        "window_coverage": "deterministic nonoverlapping starts plus final legal start; each record caption/window probability is normalized",
        "modality_probabilities": data["modality_probabilities"],
        "source_probabilities": data["source_probabilities"],
        "split_allowlist_content_sha256": hashlib.sha256(
            json.dumps(allowlist, sort_keys=True).encode()
        ).hexdigest(),
    }


def _write_new(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--audit-only", action="store_true", help="仅发布split审计；正式训练仍需train统计"
    )
    args = parser.parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preparation output: {output}")
    data = OmegaConf.to_container(OmegaConf.load(args.data_config), resolve=True)
    allowlist, report = audit_split_overlap(data)
    stats = None if args.audit_only else compute_train_statistics(data, allowlist)
    output.mkdir(parents=True, exist_ok=False)
    _write_new(output / "split_allowlist.json", allowlist)
    _write_new(output / "split_audit.json", report)
    if stats is not None:
        _write_new(output / "qpos30_train_stats.json", stats)
    print(
        json.dumps(
            {
                "output_root": str(output),
                "accepted_counts": report["accepted_counts"],
                "excluded": report["excluded_count"],
                "stats_written": stats is not None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
