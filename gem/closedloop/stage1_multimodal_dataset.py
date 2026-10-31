"""文本与音乐联合 Stage1 的完整序列读取、条件缺失和可恢复分层采样。

统一复用已有 Stage1 的 qpos30/contact2、严格因果 proprio48、teacher-forced prefix 与
完整源序列地面监督。音乐读取正式 30 Hz music release；文本读取完整 30 Hz qpos28
release，先选 caption 的真实区间，再决定目标窗口，历史仍来自区间前的完整动作。
未提供的模态采用有限零值和 false mask，绝不把无音乐文本动作解释为静音舞蹈。

训练按配置先分配音乐/文本概率，再分配库内概率；文本母来源均匀抽样，镜像和多个子片段
共享母来源权重。复用按 epoch、rank、已消费 offset 确定的抽样位置，随机 caption/窗口
不依赖 worker 预取进度。正式读取必须消费准备工具发布的跨库 split allowlist，并核验
对应 manifest 内容。在线冻结 T5 延迟到训练 rank；DataLoader worker 不加载编码器。
本模块不启动训练、修改源数据或创建占位统计量，50 Hz SuperMimic 缓存不是输入格式。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, default_collate

from gem.datasets.music_dance.music_dance_bumi import sha256_file
from gem.datasets.pure_motion.bumi_text import (
    GROUND,
    BumiTextDataset,
    caption_intervals,
    read_embedding,
)
from gem.datasets.text_source_contract import mirror_base_id

from .contracts import STAGE1_CONDITION_KEYS, STAGE1_TARGET_KEYS
from .sampling import DeterministicDrawDataset, ResumableDistributedWeightedSampler
from .stage1_dataset import BumiClosedLoopStage1Dataset

SPLIT_ALLOWLIST_SCHEMA = "genmo.bumi_multimodal.split_allowlist.v1"
MULTIMODAL_DATA_SCHEMA = "genmo.bumi_multimodal.stage1_data.v1"
TEXT_TOKENS = 150
TEXT_DIM = 1024


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _missing_text(sample):
    sample.update(
        text_features=torch.zeros(TEXT_TOKENS, TEXT_DIM),
        text_attention_mask=torch.zeros(TEXT_TOKENS, dtype=torch.bool),
        has_text=torch.tensor(False),
        text_needs_encoding=torch.tensor(False),
        caption="",
    )
    return sample


class MultimodalMusicDataset(BumiClosedLoopStage1Dataset):
    """音乐完整序列窗口补充显式缺失的文本条件，保持原 qpos/history/contact 契约。"""

    modality = "music"

    def __init__(self, *, source_name=None, **kwargs):
        super().__init__(**kwargs)
        self.source_name = source_name or self.dataset_name
        self.record_keys = [str(row["sample_id"]) for row in self.rows]
        self.data_identity = {
            "modality": self.modality,
            "source_name": self.source_name,
            "manifest": str(self.reader.manifest_path),
            "manifest_sha256": sha256_file(self.reader.manifest_path),
            "info_sha256": sha256_file(self.reader.dataset_info_path),
        }

    def get_window(self, row_index, start_frame=None):
        sample = _missing_text(super().get_window(row_index, start_frame))
        sample["meta"].update(modality="music", source_name=self.source_name, has_text=False)
        return sample


class _TextSequenceReader:
    """把经过校验的文本完整 record 暴露成公共 Stage1 序列接口，不预先裁剪。"""

    def __init__(self, release, kinematics):
        self.release = release
        self.root = release.root
        self.manifest_path = self.root / "manifests" / f"{release.split}.json"
        self.dataset_info = {"ground_semantics": GROUND}
        manifest = release.manifest
        if list(kinematics.joint_order) != manifest["joint_names"]:
            raise ValueError("text release joint order disagrees with Stage1 kinematics")
        if kinematics.kinematics_sha256 != manifest["kinematics"]["sha256"]:
            raise ValueError("text release kinematics differs from Stage1 kinematics")
        self.rows = []
        for index, (sid, _rid, item) in enumerate(release.index):
            self.rows.append(
                {
                    "sample_id": f"{item['dataset']}/{item['motion_id']}",
                    "sequence_id": item.get("canonical_source_id", item["motion_id"]),
                    "num_frames": int(item["frames"]),
                    "record_index": index,
                    "source_motion_sha256": manifest["shards"][sid]["sha256"],
                }
            )

    def load_aligned_sequence(self, row, *, record=None):
        index = int(row["record_index"])
        record = self.release.read_record(index) if record is None else record
        sid = self.release.index[index][0]
        path = self.root / self.release.manifest["shards"][sid]["path"]
        count = int(record["frames"])
        qpos = record["qpos"].float().clone()
        row = dict(row)
        row["source_motion_sha256"] = record.get("source_qpos_sha256", row["source_motion_sha256"])
        return {
            "row": row,
            "qpos": qpos,
            "music": torch.zeros(count, 35),
            "source_lengths": {"qpos": count, "music": count, "manifest": count},
            "motion_path": path,
            "music_path": "",
            "foot_contact": record.get("foot_contact"),
            "foot_contact_available": record.get(
                "foot_contact_available", torch.ones(count, 2, dtype=torch.bool)
            )
            if "foot_contact" in record
            else None,
            "foot_contact_source": "validated_text_release_contact",
        }


class BumiTextClosedLoopStage1Dataset(BumiClosedLoopStage1Dataset):
    """文本 caption 的未来窗口使用完整源动作历史，支持预计算或 rank 内冻结 T5。"""

    modality = "text"

    def __init__(
        self,
        root,
        dataset_name,
        kinematics_path,
        split="train",
        source_name=None,
        require_temporal_annotations=None,
        shard_cache_size=2,
        joint_limit_tolerance=1e-4,
        **kwargs,
    ):
        from gem.robots.bumi.kinematics import BumiKinematics

        self.source_name = source_name or dataset_name
        self.release = BumiTextDataset(
            root=root,
            split=split,
            dataset=dataset_name,
            sequence_mode="crop",
            pad_to_frames=120,
            shard_cache_size=shard_cache_size,
            require_temporal_annotations=(dataset_name == "bones_seed")
            if require_temporal_annotations is None
            else bool(require_temporal_annotations),
        )
        reader = _TextSequenceReader(self.release, BumiKinematics(kinematics_path))
        # 文本库按母来源分层权重；不用音乐的时长重复索引放大长动作/镜像权重。
        kwargs.pop("duration_aware_sampling", None)
        super().__init__(
            root=root,
            dataset_name=dataset_name,
            kinematics_path=kinematics_path,
            split=split,
            sequence_reader=reader,
            duration_aware_sampling=False,
            joint_limit_tolerance=joint_limit_tolerance,
            **kwargs,
        )
        self.joint_limit_tolerance = float(joint_limit_tolerance)
        if not math.isfinite(self.joint_limit_tolerance) or self.joint_limit_tolerance < 0:
            raise ValueError("joint_limit_tolerance must be finite and nonnegative")
        self.record_keys = [row["sample_id"] for row in self.rows]
        self.data_identity = {
            **self.release.data_identity,
            "modality": "text",
            "source_name": self.source_name,
            "manifest": str(reader.manifest_path),
            "manifest_sha256": sha256_file(reader.manifest_path),
        }

    def source_group(self, row_index):
        _, _, row = self.release.index[row_index]
        canonical = row.get("canonical_source_id")
        if not canonical:
            canonical = self.release.read_record(row_index)["provenance"].get("canonical_source_id")
        if not canonical:
            raise ValueError("text sampling requires an audited canonical_source_id")
        return mirror_base_id(str(canonical))

    def get_window(self, row_index, start_frame=None):
        row = self.rows[row_index]
        record = self.release.read_record(row_index)
        if self.release.require_temporal_annotations and (
            record.get("text_annotation_scope") != "temporal" or "caption_intervals" not in record
        ):
            raise ValueError("BONES requires temporal caption_intervals")
        qpos = record["qpos"]
        limits = self.kinematics.joint_limits.to(qpos)
        if (
            (qpos[:, 7:] < limits[:, 0] - self.joint_limit_tolerance)
            | (qpos[:, 7:] > limits[:, 1] + self.joint_limit_tolerance)
        ).any():
            raise ValueError("text qpos joint limits exceed configured tolerance")
        caption_index = (
            int(np.random.randint(len(record["captions"]))) if self.random_decision else 0
        )
        begin, end = caption_intervals(record)[caption_index]
        count = min(self.motion_frames, end - begin)
        if start_frame is None:
            choices = end - begin - count
            decision = begin + (
                int(np.random.randint(choices + 1)) if self.random_decision else choices // 2
            )
        else:
            decision = int(start_frame)
        if not begin <= decision < end:
            raise ValueError("text decision frame must be inside selected caption interval")
        sequence = self.reader.load_aligned_sequence(row, record=record)
        sample = _missing_text(self.get_sequence_window(sequence, decision, end_frame=end))
        sample["music_features"].zero_()
        sample["music_valid"].zero_()
        sample["has_text"] = torch.tensor(True)
        sample["caption"] = record["captions"][caption_index]
        if self.release.text_feature_mode == "online_t5":
            sample["text_needs_encoding"] = torch.tensor(True)
        else:
            feature, mask = read_embedding(
                record["embeddings"][caption_index],
                sample["caption"],
                self.release.cache,
                self.root,
                expected_frames=record["frames"],
            )
            feature[~mask] = 0
            sample["text_features"], sample["text_attention_mask"] = feature, mask
        sample["meta"].update(
            modality="text",
            source_name=self.source_name,
            has_text=True,
            text_feature_mode=self.release.text_feature_mode,
            caption_index=caption_index,
            caption_id=record["caption_ids"][caption_index],
            annotation_interval_frames=[begin, end],
            canonical_source_id=self.source_group(row_index),
            provenance=record["provenance"],
        )
        return sample


class _AllowedSamples(Dataset):
    """只改变可采样索引，不重新发布或重标底层 record 的 split。"""

    def __init__(self, source, allowed_keys):
        self.source = source
        self.source_name, self.modality = source.source_name, source.modality
        self.dataset_name = source.dataset_name
        self.allowed_rows = [i for i, key in enumerate(source.record_keys) if key in allowed_keys]
        allowed = set(self.allowed_rows)
        self.indices = [i for i, row in enumerate(source.idx2meta[: len(source)]) if row in allowed]
        self.data_identity = {**source.data_identity, "accepted_records": len(self.allowed_rows)}
        if not self.indices:
            raise ValueError(
                f"split allowlist leaves empty source {self.source_name}/{source.split}"
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.source[self.indices[index]]

    def relative_weights(self):
        if self.modality == "music":
            return [1.0 / len(self)] * len(self)
        groups = [self.source.source_group(self.source.idx2meta[i]) for i in self.indices]
        counts = Counter(groups)
        return [1.0 / (len(counts) * counts[group]) for group in groups]


def collate_multimodal_stage1_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """缺失模态维度一致；在线文本暂未编码，由训练 rank 在模型校验前补入。"""
    if not samples:
        raise ValueError("cannot collate an empty multimodal batch")
    keys = tuple(
        dict.fromkeys(
            (
                *STAGE1_CONDITION_KEYS,
                *STAGE1_TARGET_KEYS,
                "prefix_frames",
                "text_features",
                "text_attention_mask",
                "has_text",
                "text_needs_encoding",
            )
        )
    )
    batch = {key: default_collate([sample[key] for sample in samples]) for key in keys}
    batch["captions"] = [sample["caption"] for sample in samples]
    batch["meta"] = [sample["meta"] for sample in samples]
    batch["B"] = len(samples)
    return batch


def make_multimodal_source(entry, *, source_name, split, evaluation=False):
    """实例化受限的两个 30 Hz reader，拒绝把任意 Hydra target 当作数据输入。"""
    options = dict(_plain(entry))
    modality = options.pop("modality")
    target = options.pop("_target_", None)
    allowed_targets = {
        "music": {
            None,
            "gem.closedloop.stage1_dataset.BumiClosedLoopStage1Dataset",
            "gem.closedloop.stage1_multimodal_dataset.MultimodalMusicDataset",
        },
        "text": {None, "gem.closedloop.stage1_multimodal_dataset.BumiTextClosedLoopStage1Dataset"},
    }
    if modality not in allowed_targets or target not in allowed_targets[modality]:
        raise ValueError("multimodal source must use the declared music/text 30 Hz reader")
    if "split" in options and options["split"] != split:
        raise ValueError("source split differs from containing split")
    options.update(source_name=source_name, split=split)
    if evaluation:
        options.update(random_decision=False, duration_aware_sampling=False)
    cls = MultimodalMusicDataset if modality == "music" else BumiTextClosedLoopStage1Dataset
    return cls(**options)


def multimodal_data_identity(data_config):
    """按实际 manifest 内容记录 release 身份；不依赖 Git hash 或旧音乐 JSONL 假设。"""
    data = _plain(data_config)
    manifests = {}
    for split, entries in data["datasets"].items():
        for name, entry in entries.items():
            root = Path(entry["root"]).expanduser().resolve()
            suffix = "jsonl" if entry["modality"] == "music" else "json"
            path = root / "manifests" / f"{split}.{suffix}"
            manifests[f"{split}/{name}"] = {"path": str(path), "sha256": sha256_file(path)}
    result = {
        "schema": MULTIMODAL_DATA_SCHEMA,
        "manifests": manifests,
        "sample_contract": data.get("sample_contract", {}),
        "modality_probabilities": data.get("modality_probabilities", {"music": 0.5, "text": 0.5}),
        "source_probabilities": data.get("source_probabilities", {}),
        "source_options": data["datasets"],
    }
    path = data.get("split_allowlist")
    if path:
        path = Path(path).expanduser().resolve(strict=True)
        result["split_allowlist"] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def _load_allowlist(data):
    path = data.get("split_allowlist")
    if not path:
        raise ValueError("multimodal training requires a prepared split_allowlist")
    payload = json.loads(Path(path).expanduser().read_text())
    if payload.get("schema") != SPLIT_ALLOWLIST_SCHEMA:
        raise ValueError("unsupported multimodal split allowlist schema")
    identity = multimodal_data_identity(data)
    if payload.get("manifests") != identity["manifests"]:
        raise ValueError("split allowlist does not match current source manifests")
    if payload.get("source_identity_aliases", {}) != data.get("source_identity_aliases", {}):
        raise ValueError("split allowlist was prepared with different source identity aliases")
    return payload


def validate_multimodal_stats_identity(data_config, stats_path):
    """联合统计必须绑定最终白名单和实际采样概率，拒绝误用单模态或旧来源统计。"""
    data = _plain(data_config)
    allowlist = _load_allowlist(data)
    stats = json.loads(Path(stats_path).expanduser().read_text())
    expected = hashlib.sha256(json.dumps(allowlist, sort_keys=True).encode()).hexdigest()
    if stats.get("split_allowlist_content_sha256") != expected or stats.get("split") != "train":
        raise ValueError("multimodal stats do not match the final train split allowlist")
    for key in ("modality_probabilities", "source_probabilities"):
        if stats.get(key) != data.get(key):
            raise ValueError(f"multimodal stats {key} differs from training sampler")
    return stats


def _sources(data, split, *, evaluation=False):
    allowlist = _load_allowlist(data)
    result = []
    for name, entry in data["datasets"][split].items():
        source = make_multimodal_source(entry, source_name=name, split=split, evaluation=evaluation)
        key = f"{split}/{name}"
        if key not in allowlist["entries"]:
            raise ValueError(f"split allowlist missing source {key}")
        allowed = set(allowlist["entries"][key])
        if len(allowed) != len(allowlist["entries"][key]) or not allowed.issubset(
            source.record_keys
        ):
            raise ValueError(f"split allowlist has duplicate/unknown records: {key}")
        result.append(_AllowedSamples(source, allowed))
    return result


def _sampling_weights(sources, data):
    modal = data.get("modality_probabilities", {"music": 0.5, "text": 0.5})
    if set(modal) != {"music", "text"} or any(
        not math.isfinite(float(v)) or float(v) <= 0 for v in modal.values()
    ):
        raise ValueError("both modality probabilities must be finite and positive")
    if not math.isclose(sum(map(float, modal.values())), 1.0, abs_tol=1e-8):
        raise ValueError("modality probabilities must sum to one")
    result = []
    for modality in modal:
        entries = [source for source in sources if source.modality == modality]
        weights = data["source_probabilities"][modality]
        if set(weights) != {source.source_name for source in entries}:
            raise ValueError(f"source probabilities must exactly cover {modality} sources")
        values = list(map(float, weights.values()))
        if any(not math.isfinite(v) or v <= 0 for v in values) or not math.isclose(
            sum(values), 1.0, abs_tol=1e-8
        ):
            raise ValueError("within-modality source probabilities must be positive and sum to one")
    for source in sources:
        probability = float(modal[source.modality]) * float(
            data["source_probabilities"][source.modality][source.source_name]
        )
        result.extend(probability * value for value in source.relative_weights())
    return result


def build_multimodal_stage1_loader(
    data_config,
    *,
    split,
    batch_size,
    num_workers=0,
    seed=42,
    samples_per_epoch=None,
    rank=0,
    world_size=1,
    resumable=False,
    pin_memory=False,
):
    """与原 Stage1 工厂同形返回 DataLoader；所有训练抽样均支持精确 offset 恢复。"""
    data = _plain(data_config)
    sources = _sources(data, split, evaluation=split != "train")
    merged = ConcatDataset(sources)
    sampler = None
    if split == "train":
        sampler = ResumableDistributedWeightedSampler(
            _sampling_weights(sources, data),
            int(samples_per_epoch or len(merged)),
            seed,
            rank=rank,
            world_size=world_size,
            emit_draw_keys=True,
        )
        merged = DeterministicDrawDataset(merged, seed=seed)
    elif world_size != 1 or rank != 0:
        raise ValueError("validation uses fixed per-source loaders on the monitor rank")
    merged.data_identity = multimodal_data_identity(data)
    return DataLoader(
        merged,
        sampler=sampler,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        collate_fn=collate_multimodal_stage1_samples,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(int(seed)),
        drop_last=False,
    )


def build_multimodal_monitor_loaders(data, options):
    """按来源固定抽取整个验证集合的等距记录，避免混合大库覆盖小库结果。"""
    data, options = _plain(data), _plain(options)
    sources = _sources(data, options["split"], evaluation=True)
    loaders = {}
    for source in sources:
        count = min(len(source), int(options["samples_per_source"]))
        if count < 1:
            raise ValueError("monitor requires at least one sample per source")
        indices = np.linspace(0, len(source) - 1, count, dtype=int).tolist()
        loaders[source.source_name] = DataLoader(
            Subset(source, indices),
            batch_size=1,
            num_workers=0,
            collate_fn=collate_multimodal_stage1_samples,
            generator=torch.Generator().manual_seed(int(options["seed"])),
        )
    return loaders
