"""音乐/文本 Stage1 数据、来源隔离、真实区间与恢复采样的 CPU 回归。

所有 release 和分片在 pytest 临时目录构造，仅使用仓库内真实 BUMI 运动学。测试检查
同一 source 的镜像跨 split 排除、文本边界外保留历史而不监督未来、逐足缺失接触回退、
两级采样权重、不同 worker 与 epoch/offset 的完全恢复，以及最终 train-only 统计。
这类小样本测试不等价于正式数据全量审核、真实 T5 或机器人执行质量验收。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gem.closedloop.stage1_multimodal_dataset import (
    BumiTextClosedLoopStage1Dataset,
    MultimodalMusicDataset,
    _sampling_weights,
    _sources,
    build_multimodal_monitor_loaders,
    build_multimodal_stage1_loader,
    collate_multimodal_stage1_samples,
)
from gem.datasets.music_dance.music_dance_bumi import sha256_file
from gem.datasets.pure_motion.bumi_text import GROUND, MJCF_SHA256, SCHEMA, caption_hash
from gem.robots.bumi.endecoder import BumiEndecoder
from gem.robots.bumi.kinematics import BumiKinematics
from tests.closedloop.test_stage1_dataset import (
    KINEMATICS_PATH,
    _make_qpos,
    _write_dataset,
)
from tools.data.bumi.prepare_multimodal_stage1 import (
    MaskedMoments,
    audit_split_overlap,
    compute_train_statistics,
)


def _write_text_release(
    root,
    *,
    split="train",
    canonical=None,
    dataset="humanml3d",
    interval=(30, 50),
    mode="online_t5",
    motion_id=None,
):
    root = Path(root)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "shards").mkdir(exist_ok=True)
    kin = BumiKinematics(KINEMATICS_PATH)
    frames = 100
    motion_id = motion_id or f"motion-{split}"
    canonical = canonical or f"human:{split}"
    captions = ["a robot waves its arms", "a robot makes a small step"]
    record = {
        "qpos": _make_qpos(frames, kin),
        "frames": frames,
        "fps": 30,
        "dataset": dataset,
        "motion_id": motion_id,
        "split": split,
        "captions": captions,
        "caption_ids": ["caption-0", "caption-1"],
        "caption_intervals": [list(interval), list(interval)],
        "ground_semantics": GROUND,
        "ground_alignment": {"reference": "fixture-floor", "applied": False, "offset_z": 0.0},
        "provenance": {
            "source_id": motion_id,
            "canonical_source_id": canonical,
            "retargeter": "UMR",
            "retarget_version": "fixture",
            "interval_seconds": [0.0, frames / 30],
        },
        "text_feature_mode": mode,
        "embeddings": [],
        "foot_contact": torch.ones(frames, 2),
        "foot_contact_available": torch.ones(frames, 2, dtype=torch.bool),
        "text_annotation_scope": "temporal",
    }
    record["foot_contact_available"][:, 1] = False
    if mode == "precomputed":
        features = torch.ones(2, 150, 1024)
        mask = torch.zeros(2, 150, dtype=torch.bool)
        mask[:, :3] = True
        path = root / f"shards/{split}_embedding.pt"
        torch.save(
            {
                "max_text_len": 150,
                "encoder": "t5-3b",
                "motion_id": motion_id,
                "captions": captions,
                "embeddings": features,
                "attention_mask": mask,
            },
            path,
        )
        record["embeddings"] = [
            {
                "format": "bumi_text_t5_v1",
                "path": str(path),
                "sha256": sha256_file(path),
                "motion_id": motion_id,
                "text_index": i,
                "caption_sha256": caption_hash(caption),
            }
            for i, caption in enumerate(captions)
        ]
    path = root / f"shards/{split}_00000.pt"
    torch.save([record], path)
    manifest = {
        "schema": SCHEMA,
        "split": split,
        "fps": 30,
        "qpos_dim": 28,
        "quaternion_convention": "wxyz",
        "coordinate_system": "z_up",
        "source_mjcf_sha256": MJCF_SHA256,
        "joint_names": list(kin.joint_order),
        "kinematics": {"path": str(KINEMATICS_PATH), "sha256": kin.kinematics_sha256},
        "text_feature_mode": mode,
        "shards": [
            {
                "shard_id": 0,
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
                "record_count": 1,
                "records": [
                    {
                        "record_index": 0,
                        "frames": frames,
                        "dataset": dataset,
                        "motion_id": motion_id,
                        "canonical_source_id": canonical,
                    }
                ],
            }
        ],
    }
    (root / f"manifests/{split}.json").write_text(json.dumps(manifest))
    return root


def _make_data(root, *, text_mode="online_t5"):
    datasets = {}
    for split in ("train", "val", "test"):
        music_root = _write_dataset(root / f"music-{split}", length=100, split=split)
        # fixture 默认相同hash；为无冲突基准显式配置真实独立来源哈希与音频。
        manifest_path = music_root / f"manifests/{split}.jsonl"
        row = json.loads(manifest_path.read_text())
        row.update(
            source_motion_sha256={"train": "1", "val": "2", "test": "3"}[split] * 64,
            source_audio_sha256={"train": "4", "val": "5", "test": "6"}[split] * 64,
            music_group_id=f"music-{split}",
            sequence_id=f"sequence-{split}",
        )
        manifest_path.write_text(json.dumps(row) + "\n")
        payload_path = music_root / "motions/sample.pt"
        payload = torch.load(payload_path, weights_only=False)
        payload["source_motion_sha256"] = row["source_motion_sha256"]
        torch.save(payload, payload_path)
        text_root = _write_text_release(root / "text", split=split, mode=text_mode)
        common = dict(
            kinematics_path=str(KINEMATICS_PATH),
            split=split,
            motion_frames=60,
            history_steps=50,
            prefix_min_frames=6,
            prefix_max_frames=18,
            prefix_zero_probability=0.15,
            joint_limit_tolerance=0.0001,
        )
        datasets[split] = {
            "music_a": dict(
                **common, modality="music", root=str(music_root), dataset_name="test_bumi"
            ),
            "text_a": dict(
                **common, modality="text", root=str(text_root), dataset_name="humanml3d"
            ),
        }
    data = {
        "datasets": datasets,
        "modality_probabilities": {"music": 0.5, "text": 0.5},
        "source_probabilities": {"music": {"music_a": 1.0}, "text": {"text_a": 1.0}},
        "sample_contract": {"future_frames": 60, "history_steps": 50},
        "split_allowlist": str(root / "split_allowlist.json"),
    }
    allowlist, _ = audit_split_overlap(data)
    Path(data["split_allowlist"]).write_text(json.dumps(allowlist))
    return data


def _text(root, **kwargs):
    return BumiTextClosedLoopStage1Dataset(
        root=root,
        dataset_name="humanml3d",
        kinematics_path=KINEMATICS_PATH,
        motion_frames=60,
        random_decision=False,
        **kwargs,
    )


def test_text_interval_retains_full_history_and_masks_missing_music(tmp_path):
    source = _text(_write_text_release(tmp_path))
    sample = source[0]
    assert sample["meta"]["decision_frame"] == 30
    assert sample["future_valid"].sum() == 20
    assert sample["proprio_history_valid"].sum() > 40
    assert not sample["music_features"].any() and not sample["music_valid"].any()
    assert sample["has_text"] and sample["text_needs_encoding"]
    assert not sample["text_features"].any() and not sample["text_attention_mask"].any()
    assert sample["meta"]["ground_supervision"]["ground_height_world_m"] == 0.0
    assert sample["target_contact"][:20, 0].eq(1).all()
    assert sample["target_contact_valid"][:20].all()
    assert not sample["target_qpos30_valid"][20:].any()
    # caption 右边界后的真实邻帧可提供最后一帧delta，但不成为文本监督帧。
    assert sample["target_qpos30_valid"][19, :2].all()
    with pytest.raises(ValueError, match="caption interval"):
        source.get_window(0, start_frame=50)


def test_text_history_not_changed_by_future_mutation(tmp_path):
    source = _text(_write_text_release(tmp_path))
    before = source[0]
    record = source.release.read_record(0)
    record["qpos"][31:, 0] += 0.03
    after = source[0]
    torch.testing.assert_close(before["proprio_history"], after["proprio_history"], rtol=0, atol=0)
    assert not torch.equal(before["target_qpos30"], after["target_qpos30"])


def test_precomputed_text_and_music_explicit_missing_modes(tmp_path):
    text = _text(_write_text_release(tmp_path / "text", mode="precomputed"))[0]
    music = MultimodalMusicDataset(
        root=_write_dataset(tmp_path / "music", length=100),
        dataset_name="test_bumi",
        kinematics_path=KINEMATICS_PATH,
        motion_frames=60,
        joint_limit_tolerance=0.0001,
    )[0]
    batch = collate_multimodal_stage1_samples([text, music])
    assert batch["text_features"].shape == (2, 150, 1024)
    assert batch["has_text"].tolist() == [True, False]
    assert not batch["text_needs_encoding"].any()
    assert batch["text_attention_mask"][0].sum() == 3
    assert not batch["text_features"][~batch["text_attention_mask"]].any()
    assert batch["captions"][0] and batch["captions"][1] == ""


def test_bones_requires_temporal_annotation(tmp_path):
    root = _write_text_release(tmp_path, dataset="bones_seed")
    source = BumiTextClosedLoopStage1Dataset(
        root=root, dataset_name="bones_seed", kinematics_path=KINEMATICS_PATH
    )
    record = source.release.read_record(0)
    record["text_annotation_scope"] = "full"
    with pytest.raises(ValueError, match="BONES"):
        source[0]


def test_split_priority_and_mirror_cross_dataset_exclusion(tmp_path):
    data = _make_data(tmp_path)
    for split, canonical in [
        ("train", "common-motion"),
        ("val", "mirror_common-motion"),
        ("test", "common-motion"),
    ]:
        _write_text_release(tmp_path / "text", split=split, canonical=canonical)
    allowlist, report = audit_split_overlap(data)
    assert allowlist["entries"]["train/text_a"] == []
    assert allowlist["entries"]["val/text_a"] == []
    assert len(allowlist["entries"]["test/text_a"]) == 1
    assert report["excluded_count"] == 2 and report["conflicting_group_count"] == 1
    Path(data["split_allowlist"]).write_text(json.dumps(allowlist))
    with pytest.raises(ValueError, match="empty source"):
        build_multimodal_stage1_loader(data, split="train", batch_size=1)


def test_weights_and_exact_resume_across_worker_counts(tmp_path):
    data = _make_data(tmp_path)
    sources = _sources(data, "train")
    weights = _sampling_weights(sources, data)
    assert sum(weights[: len(sources[0])]) == pytest.approx(0.5)
    assert sum(weights[len(sources[0]) :]) == pytest.approx(0.5)
    loader = build_multimodal_stage1_loader(
        data, split="train", batch_size=2, samples_per_epoch=16, seed=13, num_workers=0
    )
    loader.sampler.set_epoch(3, 0)
    full = list(loader)
    resumed = build_multimodal_stage1_loader(
        data, split="train", batch_size=2, samples_per_epoch=16, seed=13, num_workers=2
    )
    resumed.sampler.set_epoch(3, 6)
    tail = list(resumed)
    for expected, actual in zip(full[3:], tail):
        for key in expected:
            if torch.is_tensor(expected[key]):
                torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
        assert expected["captions"] == actual["captions"]
        assert expected["meta"] == actual["meta"]
    monitors = build_multimodal_monitor_loaders(
        data, {"split": "val", "samples_per_source": 2, "seed": 4}
    )
    assert set(monitors) == {"music_a", "text_a"}


def test_manifest_change_invalidates_allowlist(tmp_path):
    data = _make_data(tmp_path)
    path = tmp_path / "text/manifests/train.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="source manifests"):
        build_multimodal_stage1_loader(data, split="train", batch_size=1)


def test_train_stats_masked_counts_and_endecoder_compatibility(tmp_path):
    data = _make_data(tmp_path)
    allowlist = json.loads(Path(data["split_allowlist"]).read_text())
    stats = compute_train_statistics(data, allowlist)
    assert stats["split"] == "train" and stats["feature_dim"] == 30
    assert set(stats["dataset_fingerprints"]) == {"music_a", "text_a"}
    assert all(value["sequences"] == 1 for value in stats["dataset_fingerprints"].values())
    assert stats["valid_coordinate_counts"][0] < stats["valid_coordinate_counts"][2]
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats))
    codec = BumiEndecoder(KINEMATICS_PATH, path)
    assert torch.isfinite(codec.mean).all() and torch.isfinite(codec.std).all()
    accumulator = MaskedMoments(2)
    accumulator.update(
        torch.tensor([[1.0, 500.0], [3.0, 6.0]]), torch.tensor([[True, False], [True, True]])
    )
    mean, std = accumulator.finalize()
    torch.testing.assert_close(mean, torch.tensor([2.0, 6.0]))
    assert std[0] == 1.0 and std[1] == pytest.approx(1e-6)


def test_weighted_stats_are_not_dominated_by_source_length():
    moments = MaskedMoments(1)
    moments.update(torch.zeros(1000, 1), torch.ones(1000, 1, dtype=torch.bool), sample_weight=0.5)
    moments.update(torch.full((1, 1), 10.0), torch.ones(1, 1, dtype=torch.bool), sample_weight=0.5)
    mean, std = moments.finalize()
    torch.testing.assert_close(mean, torch.tensor([5.0]))
    torch.testing.assert_close(std, torch.tensor([5.0]))
    assert moments.mass.item() == 1.0


def test_caption_duplicates_do_not_multiply_motion_stats_mass(tmp_path):
    data = _make_data(tmp_path)
    allowlist = json.loads(Path(data["split_allowlist"]).read_text())
    before = compute_train_statistics(data, allowlist)
    path = tmp_path / "text/shards/train_00000.pt"
    payload = torch.load(path, weights_only=False)
    record = payload[0]
    record["captions"] *= 5
    record["caption_intervals"] *= 5
    record["caption_ids"] = [f"caption-{i}" for i in range(len(record["captions"]))]
    torch.save(payload, path)
    manifest_path = tmp_path / "text/manifests/train.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shards"][0]["sha256"] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest))
    allowlist, _ = audit_split_overlap(data)
    after = compute_train_statistics(data, allowlist)
    assert before["mean"] == after["mean"] and before["std"] == after["std"]
    assert after["dataset_fingerprints"]["text_a"]["probability_mass"] == 0.5


def test_release_unverified_lineage_is_preserved_and_alias_changes_require_audit(tmp_path):
    data = _make_data(tmp_path)
    (tmp_path / "text/build_report.json").write_text(
        json.dumps(
            {
                "unverified_cross_dataset_lineage": [
                    "MotionMillion versus AMASS identities unresolved"
                ]
            }
        )
    )
    allowlist, report = audit_split_overlap(data)
    assert report["source_release_boundaries"]["train/text_a"]["unverified_cross_dataset_lineage"]
    Path(data["split_allowlist"]).write_text(json.dumps(allowlist))
    data["source_identity_aliases"] = {"human:train": "shared-source"}
    with pytest.raises(ValueError, match="source identity aliases"):
        build_multimodal_stage1_loader(data, split="train", batch_size=1)


def test_stats_require_current_allowlist_and_sampling_weights(tmp_path):
    from gem.closedloop.stage1_multimodal_dataset import validate_multimodal_stats_identity

    data = _make_data(tmp_path)
    allowlist = json.loads(Path(data["split_allowlist"]).read_text())
    stats = compute_train_statistics(data, allowlist)
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats))
    validate_multimodal_stats_identity(data, path)
    data["modality_probabilities"] = {"music": 0.7, "text": 0.3}
    with pytest.raises(ValueError, match="training sampler"):
        validate_multimodal_stats_identity(data, path)
