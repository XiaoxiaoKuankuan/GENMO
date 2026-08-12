"""CPU tests for the AIST++ music-only preflight report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tools.data.aistpp.preflight_music_only import run_preflight


def _record(frames: int) -> dict:
    poses = np.zeros((frames, 72), dtype=np.float32)
    translations = np.zeros((frames, 3), dtype=np.float32)
    return {
        "smpl_pose_global": poses.copy(),
        "smpl_trans_global": translations.copy(),
        "smpl_pose": poses.copy(),
        "smpl_trans": translations.copy(),
        "bbox_xyxy": np.tile(
            np.array([[10.0, 10.0, 50.0, 100.0]], dtype=np.float32), (frames, 1)
        ),
        "intrinsics": torch.tensor(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        "T_w2c": torch.eye(4),
    }


def _write_subset(root: Path, *, music_difference: int = 0) -> None:
    root.mkdir()
    music_root = root / "musicfeat_v2"
    music_root.mkdir()
    splits = {"train": ["train_seq"], "val": ["val_seq"], "test": ["test_seq"]}
    annot = {sequence: _record(120) for values in splits.values() for sequence in values}
    torch.save(annot, root / "annot_aist_30fps.pt")
    for name, values in splits.items():
        torch.save(values, root / f"{name}.pt")
        for sequence in values:
            music = torch.zeros(120 + music_difference, 35)
            music[::2, 33] = 1
            music[1::2, 34] = 1
            torch.save(music, music_root / f"{sequence}_musicfeat_fps30.pt")


def test_preflight_subset_passes_all_contracts(tmp_path: Path) -> None:
    root = tmp_path / "AIST++"
    _write_subset(root)
    report = run_preflight(root, strict=True, allow_subset=True)
    assert report["final_pass"] is True
    assert report["split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert report["alignment_stats"]["total"]["exact_match_count"] == 3
    assert report["hours"]["total"] == 360 / 30 / 3600


def test_preflight_strict_rejects_alignment_over_two(tmp_path: Path) -> None:
    root = tmp_path / "AIST++"
    _write_subset(root, music_difference=3)
    report = run_preflight(root, strict=True, allow_subset=True)
    assert report["final_pass"] is False
    assert report["alignment_stats"]["total"]["mismatch_gt_2_count"] == 3


def test_preflight_official_counts_required_without_allow_subset(tmp_path: Path) -> None:
    root = tmp_path / "AIST++"
    _write_subset(root)
    report = run_preflight(root, strict=True, allow_subset=False)
    assert report["final_pass"] is False
    assert any("official train count" in issue for issue in report["blocking_issues"])
