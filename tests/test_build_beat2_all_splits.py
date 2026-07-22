"""CPU-only tests for the BEAT2 all_splits index builder."""

from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

import numpy as np
import pytest

import tools.data.beat2.build_all_splits as builder
from tools.data.beat2.build_all_splits import (
    BEAT2BuildError,
    Reports,
    atomic_save_splits,
    build_parser,
    build_splits,
    is_lfs_pointer,
    make_minitrain,
    normalize_gender,
    normalize_split_type,
    parse_split_csv,
    read_audio_info,
    safe_torch_load,
    validate_npz,
    validate_split_artifact,
)

SUBSET = "beat_english_v2.0.0"


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([" id ", " TYPE "])
        writer.writerows(rows)


def _write_npz(
    path: Path,
    *,
    frames: int = 12,
    pose_dim: int = 165,
    trans_frames: int | None = None,
    betas: np.ndarray | None = None,
    gender: object = np.array("neutral"),
    omit: str | None = None,
    finite: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    poses = np.zeros((frames, pose_dim), dtype=np.float32)
    if not finite:
        poses[0, 0] = np.nan
    values = {
        "poses": poses,
        "betas": (
            np.zeros(300, dtype=np.float32) if betas is None else np.asarray(betas)
        ),
        "trans": np.zeros(
            (frames if trans_frames is None else trans_frames, 3), dtype=np.float32
        ),
        "gender": gender,
        "mocap_frame_rate": np.array(30, dtype=np.int32),
        "expressions": np.zeros((frames, 100), dtype=np.float32),
        "model": np.array("smplx2020"),
    }
    if omit is not None:
        values.pop(omit)
    np.savez(path, **values)


def _write_wav(path: Path, *, seconds: float = 1.0, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros(max(1, round(seconds * rate)), dtype=np.int16)
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(samples.tobytes())


def _create_subset(
    root: Path,
    rows: list[tuple[str, str]],
    *,
    subset: str = SUBSET,
    missing_npz: set[str] | None = None,
    missing_wav: set[str] | None = None,
    frames: int = 12,
    audio_seconds: float = 1.0,
) -> Path:
    subset_path = root / subset
    _write_csv(subset_path / "train_test_split.csv", rows)
    missing_npz = missing_npz or set()
    missing_wav = missing_wav or set()
    for video_id, _ in rows:
        if video_id and video_id not in missing_npz:
            _write_npz(subset_path / "smplxflame_30" / f"{video_id}.npz", frames=frames)
        if video_id and video_id not in missing_wav:
            _write_wav(subset_path / "wave16k" / f"{video_id}.wav", seconds=audio_seconds)
    return subset_path


def _args(root: Path, *extra: str):
    return build_parser().parse_args(
        [
            "--root",
            str(root),
            "--output",
            str(root / "all_splits.pth"),
            "--report-dir",
            str(root / "report"),
            "--min-frames",
            "1",
            "--minitrain-size",
            "2",
            *extra,
        ]
    )


def test_csv_parsing_and_validation_normalization(tmp_path: Path) -> None:
    csv_path = tmp_path / "split.csv"
    _write_csv(csv_path, [("a", "train"), ("b", "validation")])
    reports = Reports()
    result = parse_split_csv(csv_path, subset=SUBSET, reports=reports)
    assert [(item.video_id, item.split) for item in result.entries] == [
        ("a", "train"),
        ("b", "val"),
    ]
    assert normalize_split_type(" validation ") == "val"


def test_unknown_and_empty_rows_are_reported_or_strict(tmp_path: Path) -> None:
    csv_path = tmp_path / "split.csv"
    _write_csv(csv_path, [("a", "mystery"), ("", "train")])
    reports = Reports()
    result = parse_split_csv(csv_path, subset=SUBSET, reports=reports)
    assert result.entries == []
    assert len(reports.unknown_split) == 1 and len(reports.skipped) == 2
    with pytest.raises(BEAT2BuildError):
        parse_split_csv(csv_path, subset=SUBSET, reports=Reports(), strict=True)


@pytest.mark.parametrize(
    ("rows", "conflict"),
    [
        ([('a', 'train'), ('a', 'train')], False),
        ([('a', 'train'), ('a', 'test')], True),
    ],
)
def test_duplicate_rows_and_split_conflicts_are_detected(
    tmp_path: Path, rows: list[tuple[str, str]], conflict: bool
) -> None:
    csv_path = tmp_path / "split.csv"
    _write_csv(csv_path, rows)
    reports = Reports()
    result = parse_split_csv(csv_path, subset=SUBSET, reports=reports)
    assert result.entries == [] and result.duplicate_ids == {"a"}
    assert reports.duplicates[0]["split_conflict"] is conflict


def test_all_four_splits_and_default_additional_policy(tmp_path: Path) -> None:
    rows = [("a", "train"), ("b", "validation"), ("c", "test"), ("d", "additional")]
    _create_subset(tmp_path, rows)
    reports = Reports()
    splits = build_splits(_args(tmp_path), reports)
    assert [item["video_id"] for item in splits["train"]] == ["a"]
    assert [item["video_id"] for item in splits["val"]] == ["b"]
    assert [item["video_id"] for item in splits["test"]] == ["c"]
    assert [item["video_id"] for item in splits["additional"]] == ["d"]
    assert "d" not in {item["video_id"] for item in splits["train"]}


def test_include_additional_as_train_keeps_additional_list(tmp_path: Path) -> None:
    _create_subset(tmp_path, [("z", "train"), ("a", "additional")])
    splits = build_splits(_args(tmp_path, "--include-additional-as-train"), Reports())
    assert [item["video_id"] for item in splits["train"]] == ["a", "z"]
    assert [item["video_id"] for item in splits["additional"]] == ["a"]
    validate_split_artifact(splits, tmp_path, include_additional_as_train=True)


def test_same_video_id_in_different_subsets_is_allowed(tmp_path: Path) -> None:
    _create_subset(tmp_path, [("shared", "train")], subset="beat_english_v2.0.0")
    _create_subset(tmp_path, [("shared", "train")], subset="beat_spanish_v2.0.0")
    splits = build_splits(_args(tmp_path), Reports())
    assert len(splits["train"]) == 2
    assert {item["subset"] for item in splits["train"]} == {
        "beat_english_v2.0.0",
        "beat_spanish_v2.0.0",
    }


@pytest.mark.parametrize("missing", ["poses", "betas", "trans", "gender"])
def test_npz_requires_loader_fields(tmp_path: Path, missing: str) -> None:
    path = tmp_path / "motion.npz"
    _write_npz(path, omit=missing)
    with pytest.raises(ValueError, match="missing_fields"):
        validate_npz(path)


def test_npz_length_and_gender_are_normalized(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    _write_npz(path, frames=17, gender=np.array("NEUTRAL"))
    info = validate_npz(path)
    assert info["length"] == 17 and info["gender"] == "neutral"
    assert normalize_gender(" Female ") == "female"


def test_npz_rejects_short_pose_dimension(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    _write_npz(path, pose_dim=65)
    with pytest.raises(ValueError, match="invalid_poses_shape"):
        validate_npz(path)


def test_npz_rejects_trans_length_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    _write_npz(path, frames=12, trans_frames=11)
    with pytest.raises(ValueError, match="invalid_trans_shape"):
        validate_npz(path)


def test_npz_rejects_framewise_betas_and_nonfinite_values(tmp_path: Path) -> None:
    framewise = tmp_path / "framewise.npz"
    _write_npz(framewise, betas=np.zeros((12, 300), dtype=np.float32))
    with pytest.raises(ValueError, match="unsupported_betas_shape"):
        validate_npz(framewise)
    nonfinite = tmp_path / "nonfinite.npz"
    _write_npz(nonfinite, finite=False)
    with pytest.raises(ValueError, match="nonfinite_poses"):
        validate_npz(nonfinite)


def test_gender_loader_incompatible_one_element_array_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    _write_npz(path, gender=np.array(["neutral"]))
    with pytest.raises(ValueError, match="unsupported_gender_loader_repr"):
        validate_npz(path)


def test_too_short_motion_is_skipped(tmp_path: Path) -> None:
    _create_subset(tmp_path, [("a", "train")], frames=5)
    args = _args(tmp_path)
    args.min_frames = 6
    reports = Reports()
    splits = build_splits(args, reports)
    assert splits["train"] == [] and len(reports.too_short) == 1


@pytest.mark.parametrize("missing_kind", ["npz", "wav"])
def test_missing_pair_fails_by_default_and_can_be_skipped(
    tmp_path: Path, missing_kind: str
) -> None:
    kwargs = {f"missing_{missing_kind}": {"a"}}
    _create_subset(tmp_path, [("a", "train")], **kwargs)
    with pytest.raises(BEAT2BuildError, match="missing or invalid"):
        build_splits(_args(tmp_path), Reports())
    reports = Reports()
    splits = build_splits(_args(tmp_path, "--allow-missing-pairs"), reports)
    assert splits["train"] == []
    assert len(getattr(reports, f"missing_{missing_kind}")) == 1


def test_git_lfs_pointer_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    path.write_bytes(builder.LFS_PREFIX + b"\noid sha256:abc\nsize 123\n")
    assert is_lfs_pointer(path)
    with pytest.raises(ValueError, match="lfs_pointer_not_downloaded"):
        validate_npz(path)


def test_short_audio_is_skipped_unless_explicitly_allowed(tmp_path: Path) -> None:
    _create_subset(tmp_path, [("a", "train")], frames=30, audio_seconds=0.5)
    reports = Reports()
    assert build_splits(_args(tmp_path), reports)["train"] == []
    assert len(reports.short_audio) == 1
    allowed = build_splits(_args(tmp_path, "--allow-short-audio"), Reports())
    assert len(allowed["train"]) == 1


def test_long_audio_metadata_is_allowed(tmp_path: Path) -> None:
    path = tmp_path / "long.wav"
    _write_wav(path, seconds=2.0)
    info = read_audio_info(path)
    assert info["samplerate"] == 16000
    assert info["duration_sec"] == pytest.approx(2.0)


def test_orphan_npz_and_wav_are_audited_but_not_indexed(tmp_path: Path) -> None:
    subset = _create_subset(tmp_path, [("a", "train")])
    _write_npz(subset / "smplxflame_30" / "orphan_motion.npz")
    _write_wav(subset / "wave16k" / "orphan_audio.wav")
    reports = Reports()
    splits = build_splits(_args(tmp_path), reports)
    assert [item["video_id"] for item in splits["train"]] == ["a"]
    assert reports.orphan_npz[0]["video_id"] == "orphan_motion"
    assert reports.orphan_wav[0]["video_id"] == "orphan_audio"


def test_minitrain_is_deterministic() -> None:
    train = [
        {"video_id": name, "subset": SUBSET, "length": 120}
        for name in ("a", "b", "c")
    ]
    assert [item["video_id"] for item in make_minitrain(train, 2)] == ["a", "b"]


def test_split_overlap_validation(tmp_path: Path) -> None:
    subset = _create_subset(tmp_path, [("a", "train")])
    del subset
    item = {"video_id": "a", "subset": SUBSET, "length": 12}
    value = {
        "train": [item],
        "val": [dict(item)],
        "test": [],
        "minitrain": [dict(item)],
        "additional": [],
    }
    with pytest.raises(ValueError, match="train/val overlap"):
        validate_split_artifact(value, tmp_path, include_additional_as_train=False)


def test_split_validation_rejects_nonprefix_minitrain(tmp_path: Path) -> None:
    _create_subset(tmp_path, [("a", "train"), ("b", "train")])
    train = [
        {"video_id": name, "subset": SUBSET, "length": 12}
        for name in ("a", "b")
    ]
    value = {
        "train": train,
        "val": [],
        "test": [],
        "minitrain": [dict(train[1])],
        "additional": [],
    }
    with pytest.raises(ValueError, match="deterministic sorted train prefix"):
        validate_split_artifact(value, tmp_path, include_additional_as_train=False)


def test_atomic_save_reloads_and_validates(tmp_path: Path) -> None:
    _create_subset(tmp_path, [("a", "train")])
    splits = build_splits(_args(tmp_path), Reports())
    output = tmp_path / "all_splits.pth"
    size = atomic_save_splits(
        splits,
        output,
        tmp_path,
        overwrite=False,
        include_additional_as_train=False,
    )
    assert size > 0 and set(safe_torch_load(output)) == set(builder.SPLITS)
    assert not output.with_name(output.name + ".tmp").exists()


def test_dry_run_writes_reports_but_not_output(tmp_path: Path) -> None:
    _create_subset(tmp_path, [("a", "train")])
    output = tmp_path / "result" / "all_splits.pth"
    report = tmp_path / "report"
    assert (
        builder.main(
            [
                "--root",
                str(tmp_path),
                "--output",
                str(output),
                "--report-dir",
                str(report),
                "--min-frames",
                "1",
                "--dry-run",
            ]
        )
        == 0
    )
    assert not output.exists()
    summary = json.loads((report / "build_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "dry_run_complete"
