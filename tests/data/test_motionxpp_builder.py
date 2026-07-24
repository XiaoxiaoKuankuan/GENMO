from __future__ import annotations

import json

import torch

from tests.data.motionxpp_fixtures import make_motionxpp_root
from tools.data.motionxpp.build_motionxpp_genmo import (
    build_dataset,
    build_parser,
)
from tools.data.motionxpp.common import (
    build_asset_index,
    deterministic_split,
    parse_motion_asset,
    resample_motion,
    safe_torch_load,
)


def _args(root, output, *extra):
    return build_parser().parse_args(
        [
            "--root",
            str(root),
            "--output-root",
            str(output),
            "--subsets",
            "toy",
            "--source-up-axis",
            "y",
            *extra,
        ]
    )


def test_builder_writes_valid_shards_and_manifest(tmp_path):
    root = make_motionxpp_root(tmp_path / "raw")
    output = tmp_path / "built"
    result = build_dataset(_args(root, output, "--records-per-shard", "1"))
    assert result["summary"]["accepted_records"] == 2
    assert result["summary"]["rejected_records"] == 0
    rows = [row for values in result["manifests"].values() for row in values]
    assert len(rows) == 2
    for row in rows:
        shard = safe_torch_load(output / row["shard_path"])
        record = shard[row["record_key"]]
        assert record["pose"].shape[1] == 66
        assert record["trans"].shape[1] == 3
        assert record["beta"].shape[-1] == 10
        assert record["pose"].dtype == torch.float32
        assert record["text_data"][0]["source"] == "toy"
    assert all(
        (output / "manifests" / f"{split}.jsonl").exists()
        for split in (
            "train",
            "val",
            "test",
        )
    )


def test_dict_motion_and_30_60_fps_resampling(tmp_path):
    root = make_motionxpp_root(
        tmp_path / "raw",
        records=[
            {
                "stem": "dict_clip1",
                "frames": 61,
                "caption": ["First caption.", "Second caption."],
                "dict_format": True,
                "fps": 60,
            }
        ],
    )
    index = build_asset_index(root, "motion", "toy")
    parsed = parse_motion_asset(index.assets["dict_clip1"])
    assert parsed["body_pose"].shape == (61, 63)
    assert parsed["fps"] == 60
    resampled = resample_motion(parsed, 60, 30)
    assert resampled["body_pose"].shape == (31, 63)
    unchanged = resample_motion(parsed, 30, 30)
    assert unchanged["body_pose"].shape == (61, 63)
    result = build_dataset(_args(root, tmp_path / "built"))
    row = next(row for rows in result["manifests"].values() for row in rows)
    assert row["frames"] == 31
    assert row["caption_count"] == 2


def test_nan_rejected_and_duplicate_hash_reported(tmp_path):
    bad_root = make_motionxpp_root(
        tmp_path / "bad",
        records=[
            {"stem": "bad_clip1", "frames": 40, "caption": "Bad.", "nan": True},
            {"stem": "good_clip1", "frames": 40, "caption": "Good."},
        ],
    )
    bad = build_dataset(_args(bad_root, tmp_path / "bad_out"))
    assert bad["summary"]["accepted_records"] == 1
    assert bad["summary"]["rejected_records"] == 1
    assert "NaN or Inf" in bad["rejected"][0]["error"]

    duplicate_root = make_motionxpp_root(
        tmp_path / "duplicate",
        records=[
            {"stem": "same_clip1", "frames": 40, "caption": "One."},
            {"stem": "same_clip2", "frames": 40, "caption": "Two."},
        ],
    )
    duplicate = build_dataset(_args(duplicate_root, tmp_path / "dup_out"))
    assert duplicate["summary"]["accepted_records"] == 1
    assert duplicate["summary"]["duplicate_records"] == 1
    persisted = json.loads((tmp_path / "dup_out/reports/build_summary.json").read_text())
    assert persisted["duplicate_records"] == 1


def test_strict_mode_still_applies_documented_minimum_length_filter(tmp_path):
    root = make_motionxpp_root(
        tmp_path / "short",
        records=[
            {"stem": "short_clip1", "frames": 12, "caption": "Too short."},
            {"stem": "good_clip1", "frames": 40, "caption": "Long enough."},
        ],
    )
    result = build_dataset(_args(root, tmp_path / "out", "--strict"))
    assert result["summary"]["accepted_records"] == 1
    assert result["summary"]["filtered_short_records"] == 1


def test_group_split_does_not_leak_and_resume_skips_shards(tmp_path):
    for seed in (1, 20260724, 999):
        assert deterministic_split("toy", "sequence_clip1", seed) == (
            deterministic_split("toy", "sequence_clip2", seed)
        )
    root = make_motionxpp_root(tmp_path / "raw")
    output = tmp_path / "built"
    args = _args(root, output, "--records-per-shard", "1")
    first = build_dataset(args)
    assert first["summary"]["resumed_shards"] == 0
    first_meta = next((output / "shards").rglob("*.meta.json"))
    first_meta.unlink()
    resumed_args = _args(root, output, "--records-per-shard", "1", "--resume")
    resumed = build_dataset(resumed_args)
    assert resumed["summary"]["resumed_shards"] == 2
    assert first_meta.is_file()


def test_dry_run_does_not_write_motion_shards_or_manifests(tmp_path):
    root = make_motionxpp_root(tmp_path / "raw")
    output = tmp_path / "built"
    result = build_dataset(_args(root, output, "--dry-run"))
    assert result["summary"]["status"] == "dry_run_complete"
    assert not (output / "shards").exists()
    assert not (output / "manifests").exists()
    assert (output / "reports/build_summary.json").is_file()
