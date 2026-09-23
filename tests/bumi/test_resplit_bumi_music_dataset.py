"""验证 BUMI PASS 四库重新划分的独立性、可重复性和发布数据完整性。

测试使用系统临时目录里的合成数据，不访问服务器、正式训练目录或用户 checkpoint。
覆盖同一原视频的不同群舞片段、不同舞者、跨库相同音频及 Mine 同曲版本的不可拆分
约束；验证固定 seed、90/5/5 整数目标、非 PASS/重复样本拒绝、旧数据原样保留及失败
清理。端到端测试复用现有 Stage1 数据夹具和原严格 Dataset validator，实际计算新 train
的 qpos30 stats，检查这些统计指纹不会指向旧 train 或新 val/test；不构建或训练模型。
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
import torch

from tests.closedloop.test_stage1_dataset import KINEMATICS_PATH, _write_dataset
from tools.data.bumi.resplit_bumi_music_dataset import (
    DATASETS,
    SPLITS,
    build_groups,
    digest,
    make_plan,
    populate,
    publish,
    read_source,
    write_json,
)


def _record(name, index):
    identity = f"{name}:{index}"
    sid = f"v{index:010d}_00_0_120" if name == "AIOZ-GDANCE" else str(index)
    return {
        "source": name,
        "row": {
            "sample_id": sid,
            "sequence_id": sid,
            "music_group_id": sid,
            "audio_key": sid,
            "num_frames": 120,
            "split": "train",
            "dataset": DATASETS[name],
            "quality_accepted": True,
            "fps": 30,
            **{
                field: hashlib.sha256(f"{identity}:{field}".encode()).hexdigest()
                for field in [
                    "source_motion_sha256",
                    "source_audio_sha256",
                    "source_music_feature_sha256",
                ]
            },
        },
    }


def test_grouping_connects_original_videos_titles_and_cross_source_audio():
    records = [
        _record("AIOZ-GDANCE", 1),
        _record("AIOZ-GDANCE", 2),
        _record("Mine", 1),
        _record("Mine", 2),
        _record("FineDance", 1),
    ]
    records[1]["row"]["sequence_id"] = "v0000000001_09_0_120"
    records[2]["row"]["song_name"] = "shake-it"
    records[3]["row"]["song_name"] = "shake-it-complete"
    records[4]["row"]["source_audio_sha256"] = records[3]["row"]["source_audio_sha256"]
    groups = build_groups(records)
    assert sorted(len(g["records"]) for g in groups) == [2, 3]
    assert any(g["sources"] == ["FineDance", "Mine"] for g in groups)


def test_seeded_plan_preserves_all_records_and_matches_targets():
    records = [_record(name, index) for name in DATASETS for index in range(100)]
    groups, assignments, report = make_plan(records, 42)
    _, reverse_assignments, reverse_report = make_plan(list(reversed(records)), 42)
    assert assignments == reverse_assignments
    assert report == reverse_report
    assert report["counts"] == {"train": 360, "val": 20, "test": 20}
    assert sum(len(g["records"]) for g in groups) == len(records)
    assert all(
        {s: v["sequences"] for s, v in d.items()} == {"train": 90, "val": 5, "test": 5}
        for d in report["datasets"].values()
    )
    assert make_plan(records, 43)[1] != assignments


def test_indivisible_groups_report_nearest_ratio_and_reject_too_few_groups():
    records = [_record(name, index) for name in DATASETS for index in range(20)]
    for record in records:
        if record["source"] == "Mine":
            record["row"]["music_group_id"] = str(int(record["row"]["sample_id"]) // 2)
    _, _, report = make_plan(records, 42)
    assert report["targets"]["Mine"]["val"] == 1
    assert report["datasets"]["Mine"]["val"]["sequences"] == 2
    for record in records:
        if record["source"] == "Mine":
            record["row"]["music_group_id"] = "one-song"
    with pytest.raises(ValueError, match="Not enough independent groups"):
        make_plan(records, 42)


def _source_fixture(root):
    for name, dataset in DATASETS.items():
        base = _write_dataset(root / name, length=12)
        motion = torch.load(base / "motions/sample.pt", weights_only=False)
        music = torch.load(base / "musicfeat_v2/sample.pt", weights_only=False)
        info = json.loads((base / "meta/dataset_info.json").read_text())
        info.update(
            dataset_name=dataset,
            quality_acceptance_policy="PASS_ONLY",
            split_counts={"train": 18, "val": 1, "test": 1},
        )
        write_json(base / "meta/dataset_info.json", info)
        rows = {split: [] for split in SPLITS}
        for index in range(20):
            row = _record(name, index)["row"]
            row["split"] = "train" if index < 18 else ("val" if index == 18 else "test")
            row["num_frames"] = 12
            sid = row["sample_id"]
            payload = copy.deepcopy(motion)
            payload["source_motion_sha256"] = row["source_motion_sha256"]
            row.update(
                motion_path=f"motions/{sid}.pt",
                music_feature_path=f"musicfeat_v2/{sid}.pt",
                audio_path=f"audio/{sid}.wav",
            )
            torch.save(payload, base / row["motion_path"])
            # 独立特征哈希，避免夹具的相同零张量把所有歌曲连成一个组。
            unique_music = music.clone()
            unique_music[:, 1] = index + list(DATASETS).index(name) * 100
            torch.save(unique_music, base / row["music_feature_path"])
            (base / row["audio_path"]).write_bytes(f"test audio {name}:{index}".encode())
            row["source_music_feature_sha256"] = digest(base / row["music_feature_path"])
            row["source_audio_sha256"] = digest(base / row["audio_path"])
            rows[row["split"]].append(row)
        for split, values in rows.items():
            (base / "manifests" / f"{split}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in values)
            )
    return root


def test_publication_validates_payloads_and_computes_new_train_only_stats(tmp_path):
    source = _source_fixture(tmp_path / "source")
    records, _, before = read_source(source)
    output = tmp_path / "release"
    report = publish(source, output, KINEMATICS_PATH, 42)
    assert report["counts"] == {"train": 72, "val": 4, "test": 4}
    assert report["materialization"]["hardlinked_files"] == 80 * 3
    assert report["group_key_cross_split_overlaps"] == 0
    assert read_source(source)[2] == before
    original = {(r["source"], r["row"]["sample_id"]): r["row"] for r in records}
    seen = set()
    for name in DATASETS:
        for split in SPLITS:
            path = output / name / "manifests" / f"{split}.jsonl"
            for line in path.read_text().splitlines():
                row = json.loads(line)
                key = (name, row["sample_id"])
                assert key not in seen
                seen.add(key)
                provenance = row.pop("resplit_provenance")
                assert row["split"] == split
                row["split"] = provenance["original_split"]
                assert row == original[key]
    assert seen == original.keys()
    stats = json.loads((output / "stats/qpos30_train_stats.json").read_text())
    assert stats["num_feature_frames"] == 72 * 12
    for name, dataset in DATASETS.items():
        entry = stats["dataset_fingerprints"][dataset]
        assert entry["sequences"] == 18
        assert entry["train_manifest_sha256"] == digest(output / name / "manifests/train.jsonl")
        assert entry["train_manifest_sha256"] != digest(source / name / "manifests/train.jsonl")
    with pytest.raises(FileExistsError):
        publish(source, output, KINEMATICS_PATH, 42)
    assert not list(tmp_path.glob(".release.staging-*"))


def test_rejects_nonpass_duplicate_and_escaping_artifact(tmp_path):
    source = _source_fixture(tmp_path / "source")
    path = source / "Mine/manifests/train.jsonl"
    content = path.read_text()
    rows = [json.loads(line) for line in content.splitlines()]
    rows[0]["quality_accepted"] = False
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="only PASS"):
        read_source(source)
    rows[0]["quality_accepted"] = True
    rows[1]["sample_id"] = rows[0]["sample_id"]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="duplicate"):
        read_source(source)
    path.write_text(content)
    records, infos, fingerprints = read_source(source)
    groups, assignments, _ = make_plan(records, 42)
    records[0]["row"]["motion_path"] = "../escape.pt"
    with pytest.raises(ValueError, match="invalid artifact path"):
        populate(source, tmp_path / "staging", records, infos, fingerprints, groups, assignments)


def test_failed_publication_cleans_only_its_staging(tmp_path, monkeypatch):
    source = _source_fixture(tmp_path / "source")
    marker = tmp_path / "keep.txt"
    marker.write_text("unrelated")

    def fail(*args):
        raise RuntimeError("validator failed")

    monkeypatch.setattr(
        "tools.data.bumi.resplit_bumi_music_dataset.validate_and_compute_stats", fail
    )
    with pytest.raises(RuntimeError, match="validator failed"):
        publish(source, tmp_path / "release", KINEMATICS_PATH, 42)
    assert marker.read_text() == "unrelated"
    assert not (tmp_path / "release").exists()
    assert not list(tmp_path.glob(".release.staging-*"))
