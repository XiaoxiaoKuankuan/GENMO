"""CPU-only tests for the official AIST++ crossmodal annotation builder."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

import tools.data.aistpp.build_annot_aist_official_30fps as official


class _DummyBodyModel:
    def get_skeleton(self, betas: torch.Tensor) -> torch.Tensor:
        return torch.zeros(55, 3, dtype=torch.float32)


def _camera(name: str) -> dict:
    return {
        "name": name,
        "size": [640, 480],
        "matrix": [[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]],
        "distortions": [0.0] * 5,
        "rotation": [0.0, 0.0, 0.0],
        "translation": [0.0, 0.0, 3.0],
    }


def _music(length: int = 4) -> torch.Tensor:
    result = torch.zeros(length, 35, dtype=torch.float32)
    result[::2, 33] = 1.0
    result[1::2, 34] = 1.0
    return result


def _record(length: int = 4) -> dict:
    pose = np.zeros((length, 72), dtype=np.float32)
    trans = np.zeros((length, 3), dtype=np.float32)
    return {
        "smpl_pose_global": pose.copy(),
        "smpl_trans_global": trans.copy(),
        "smpl_pose": pose.copy(),
        "smpl_trans": trans.copy(),
        "bbox_xyxy": np.tile(
            np.array([[10.0, 20.0, 40.0, 80.0]], dtype=np.float32), (length, 1)
        ),
        "intrinsics": torch.eye(3, dtype=torch.float32),
        "T_w2c": torch.eye(4, dtype=torch.float32),
        "contact_supervision_valid": True,
        "height": 480,
        "width": 640,
    }


def _write_sequence(annotations: Path, music: Path, sequence: str) -> None:
    motion = {
        "smpl_poses": np.zeros((8, 72), dtype=np.float32),
        "smpl_scaling": np.array([100.0], dtype=np.float32),
        "smpl_trans": np.zeros((8, 3), dtype=np.float32),
    }
    with (annotations / "motions" / f"{sequence}.pkl").open("wb") as file:
        pickle.dump(motion, file)
    keypoints = np.zeros((9, 8, 17, 3), dtype=np.float32)
    keypoints[..., :4, 0] = np.array([10, 30, 15, 25])
    keypoints[..., :4, 1] = np.array([20, 40, 35, 25])
    keypoints[..., :4, 2] = 1.0
    with (annotations / "keypoints2d" / f"{sequence}.pkl").open("wb") as file:
        pickle.dump({"keypoints2d": keypoints}, file)
    torch.save(_music(), music / f"{sequence}_musicfeat_fps30.pt")


def _synthetic_tree(root: Path) -> tuple[Path, Path, dict[str, list[str]]]:
    annotations = root / "annotations"
    music = root / "musicfeat_v2"
    for directory in ("motions", "keypoints2d", "cameras", "splits"):
        (annotations / directory).mkdir(parents=True, exist_ok=True)
    music.mkdir(parents=True)
    splits = {
        "train": ["seq_train_b", "seq_train_a"],
        "val": ["seq_val"],
        "test": ["seq_test"],
    }
    sequences = splits["train"] + splits["val"] + splits["test"]
    for name, values in splits.items():
        (annotations / "splits" / official.SPLIT_FILENAMES[name]).write_text(
            "\n".join(values) + "\n", encoding="utf-8"
        )
    (annotations / "ignore_list.txt").write_text("", encoding="utf-8")
    (annotations / "cameras" / "environment.json").write_text(
        json.dumps([_camera(f"c{number:02d}") for number in range(1, 10)]),
        encoding="utf-8",
    )
    (annotations / "cameras" / "mapping.txt").write_text(
        "".join(f"{sequence} environment\n" for sequence in sequences),
        encoding="utf-8",
    )
    for sequence in sequences:
        _write_sequence(annotations, music, sequence)
    _write_sequence(annotations, music, "extra_sequence")
    return annotations, music, splits


def _args(annotations: Path, music: Path, output: Path, report: Path) -> list[str]:
    return [
        "--annotations-root",
        str(annotations),
        "--musicfeat-dir",
        str(music),
        "--output-root",
        str(output),
        "--report-dir",
        str(report),
        "--expected-train-count",
        "2",
        "--expected-val-count",
        "1",
        "--expected-test-count",
        "1",
        "--minitrain-size",
        "1",
        "--min-sequence-frames",
        "1",
    ]


def test_ordered_split_reader_preserves_order(tmp_path: Path) -> None:
    path = tmp_path / "split.txt"
    path.write_text("z\n\na\nm\n", encoding="utf-8")
    assert official.read_ordered_unique_ids(path) == ["z", "a", "m"]


def test_ordered_split_reader_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "split.txt"
    path.write_text("a\nb\na\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        official.read_ordered_unique_ids(path)


def test_official_splits_reject_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        official.validate_official_splits(
            ["a"], ["a"], ["c"], expected_train=1, expected_val=1, expected_test=1
        )


def test_official_candidate_set_is_exact_union() -> None:
    result = official.validate_official_splits(
        ["b", "a"], ["v"], ["t"], expected_train=2, expected_val=1, expected_test=1
    )
    assert result == {"a", "b", "v", "t"}


@pytest.fixture()
def built_dataset(tmp_path: Path, monkeypatch):
    annotations, music, source_splits = _synthetic_tree(tmp_path)
    monkeypatch.setattr(official, "make_smplx", lambda _kind: _DummyBodyModel())
    args = official.build_parser().parse_args(
        _args(annotations, music, tmp_path / "output", tmp_path / "report")
    )
    official.validate_args(args)
    reports = official.OfficialBuildReports(summary=official._base_summary(args))
    annot, splits, expected, lengths = official.build_official_dataset(args, reports)
    return tmp_path, args, annot, splits, expected, lengths, reports, source_splits


def test_extra_motion_does_not_enter_annot(built_dataset) -> None:
    assert "extra_sequence" not in built_dataset[2]


def test_extra_music_feature_does_not_enter_annot_or_fail(built_dataset) -> None:
    reports = built_dataset[6]
    assert reports.summary["extra_musicfeat_count"] == 1
    assert reports.extra_data["extra_musicfeat_ids"] == ["extra_sequence"]


@pytest.mark.parametrize(
    ("kind", "relative", "message"),
    [
        ("motion", Path("motions/seq_train_b.pkl"), "missing motion"),
        ("keypoints", Path("keypoints2d/seq_train_b.pkl"), "missing keypoints2d"),
    ],
)
def test_missing_official_local_source_is_fatal(
    tmp_path: Path, monkeypatch, kind: str, relative: Path, message: str
) -> None:
    annotations, music, _ = _synthetic_tree(tmp_path)
    (annotations / relative).unlink()
    monkeypatch.setattr(official, "make_smplx", lambda _kind: _DummyBodyModel())
    with pytest.raises(official.AISTOfficialBuildError, match=message):
        official.main(_args(annotations, music, tmp_path / "out", tmp_path / "report") + ["--dry-run"])


def test_missing_official_music_is_fatal(tmp_path: Path, monkeypatch) -> None:
    annotations, music, _ = _synthetic_tree(tmp_path)
    (music / "seq_train_b_musicfeat_fps30.pt").unlink()
    monkeypatch.setattr(official, "make_smplx", lambda _kind: _DummyBodyModel())
    with pytest.raises(official.AISTOfficialBuildError, match="missing musicfeat"):
        official.main(_args(annotations, music, tmp_path / "out", tmp_path / "report") + ["--dry-run"])


def test_missing_official_camera_mapping_is_fatal(tmp_path: Path, monkeypatch) -> None:
    annotations, music, _ = _synthetic_tree(tmp_path)
    mapping = annotations / "cameras" / "mapping.txt"
    mapping.write_text(
        mapping.read_text(encoding="utf-8").replace("seq_train_b environment\n", ""),
        encoding="utf-8",
    )
    monkeypatch.setattr(official, "make_smplx", lambda _kind: _DummyBodyModel())
    with pytest.raises(official.AISTOfficialBuildError, match="missing camera"):
        official.main(_args(annotations, music, tmp_path / "out", tmp_path / "report") + ["--dry-run"])


def test_official_ignore_intersection_is_fatal(tmp_path: Path, monkeypatch) -> None:
    annotations, music, _ = _synthetic_tree(tmp_path)
    (annotations / "ignore_list.txt").write_text("seq_val\n", encoding="utf-8")
    monkeypatch.setattr(official, "make_smplx", lambda _kind: _DummyBodyModel())
    report = tmp_path / "report"
    with pytest.raises(official.AISTOfficialBuildError, match="ignore_list"):
        official.main(_args(annotations, music, tmp_path / "out", report) + ["--dry-run"])
    payload = json.loads((report / "missing_required_data.json").read_text())
    assert payload["ignored_official_ids"] == ["seq_val"]


def test_explicit_authorization_allows_official_ignore_intersection(
    tmp_path: Path, monkeypatch
) -> None:
    annotations, music, _ = _synthetic_tree(tmp_path)
    (annotations / "ignore_list.txt").write_text("seq_val\n", encoding="utf-8")
    monkeypatch.setattr(official, "make_smplx", lambda _kind: _DummyBodyModel())
    report = tmp_path / "report"
    assert (
        official.main(
            _args(annotations, music, tmp_path / "out", report)
            + ["--dry-run", "--allow-ignored-official"]
        )
        == 0
    )
    payload = json.loads((report / "missing_required_data.json").read_text())
    assert payload["ignored_official_authorized"] is True
    summary = json.loads((report / "build_summary.json").read_text())
    assert summary["successfully_built_count"] == 4
    assert summary["allow_ignored_official"] is True


def test_output_splits_and_annot_preserve_exact_source_order(built_dataset) -> None:
    _, _, annot, splits, _, _, _, source = built_dataset
    assert splits["train"] == source["train"]
    assert splits["val"] == source["val"]
    assert splits["test"] == source["test"]
    assert list(annot) == source["train"] + source["val"] + source["test"]


def test_minitrain_is_deterministic_in_train_source_order() -> None:
    annot = {"b": _record(100), "a": _record(130), "c": _record(140)}
    assert official.choose_ordered_minitrain(["b", "a", "c"], annot, 2, 120) == ["a", "c"]


def test_standard_output_filenames() -> None:
    args = official.build_parser().parse_args([])
    paths = official.output_paths(args)
    assert {name: path.name for name, path in paths.items()} == official.DEFAULT_FILENAMES


def test_dry_run_writes_reports_only_and_preserves_partial_files(
    tmp_path: Path, monkeypatch
) -> None:
    annotations, music, _ = _synthetic_tree(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    partials = {
        output / "annot_aist_30fps_partial.pt": b"annot partial sentinel",
        output / "train_partial.pt": b"train partial sentinel",
        output / "val_partial.pt": b"val partial sentinel",
        output / "test_partial.pt": b"test partial sentinel",
        output / "minitrain_partial.pt": b"minitrain partial sentinel",
    }
    for path, payload in partials.items():
        path.write_bytes(payload)
    monkeypatch.setattr(official, "make_smplx", lambda _kind: _DummyBodyModel())
    assert official.main(_args(annotations, music, output, tmp_path / "report") + ["--dry-run"]) == 0
    assert all(path.read_bytes() == payload for path, payload in partials.items())
    assert not any((output / filename).exists() for filename in official.DEFAULT_FILENAMES.values())


def test_limit_cannot_publish_standard_default_outputs(monkeypatch) -> None:
    args = official.build_parser().parse_args(["--limit", "1"])
    monkeypatch.setattr(Path, "is_dir", lambda _self: True)
    monkeypatch.setattr(Path, "is_file", lambda _self: True)
    with pytest.raises(ValueError, match="cannot publish"):
        official.validate_args(args)


def test_atomic_save_reloads_valid_official_outputs(built_dataset) -> None:
    tmp_path, _, annot, splits, expected, _, _, _ = built_dataset
    root = tmp_path / "published"
    paths = {
        "annot": root / "annot.pt",
        "train": root / "train.pt",
        "val": root / "val.pt",
        "test": root / "test.pt",
        "minitrain": root / "minitrain.pt",
    }
    official.atomic_save_official_outputs(annot, splits, expected, paths, overwrite=False)
    loaded = official.safe_torch_load(paths["annot"])
    loaded_splits = {name: official.safe_torch_load(paths[name]) for name in splits}
    assert official.validate_official_outputs(loaded, loaded_splits, expected)[0] == 4
    assert not list(root.glob("*.tmp"))


def test_atomic_save_failure_leaves_no_partial_final_outputs(
    built_dataset, monkeypatch
) -> None:
    tmp_path, _, annot, splits, expected, _, _, _ = built_dataset
    root = tmp_path / "failed_publish"
    paths = {
        "annot": root / "annot.pt",
        "train": root / "train.pt",
        "val": root / "val.pt",
        "test": root / "test.pt",
        "minitrain": root / "minitrain.pt",
    }
    real_replace = official.os.replace
    calls = 0

    def fail_second_publish(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(official.os, "replace", fail_second_publish)
    with pytest.raises(OSError, match="injected"):
        official.atomic_save_official_outputs(annot, splits, expected, paths, overwrite=False)
    assert not any(path.exists() for path in paths.values())
    assert not list(root.glob("*.tmp"))


def test_extra_data_report_contains_required_disclaimer(built_dataset) -> None:
    assert "ignored" in built_dataset[6].extra_data["note"]
    assert "official crossmodal" in built_dataset[6].extra_data["note"]


def test_validation_rejects_annot_with_extra_key(built_dataset) -> None:
    _, _, annot, splits, expected, _, _, _ = built_dataset
    bad = dict(annot)
    bad["extra"] = _record()
    with pytest.raises(ValueError, match="insertion order"):
        official.validate_official_outputs(bad, splits, expected)
