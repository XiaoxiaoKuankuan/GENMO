"""End-to-end contracts for the four-dataset human motion curation tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from gem.utils.rotation_conversions import axis_angle_to_matrix
from tools.data.music_dance.curation.apply_review_results import apply_results
from tools.data.music_dance.curation.common import (
    DECISION_COLUMNS,
    Z_UP_TO_Y_UP,
    read_csv,
    read_jsonl,
    transform_z_up_to_y_up,
    write_csv,
)
from tools.data.music_dance.curation.export_motion_review import export_package
from tools.data.music_dance.curation.validate_curated_datasets import validate_curated
from tools.data.music_dance.curation.validate_review_package import validate_package
from tools.data.music_dance.curation.validate_review_results import validate_decisions


REPO_ROOT = Path(__file__).resolve().parents[1]


def _music(frames: int) -> torch.Tensor:
    value = torch.zeros(frames, 35, dtype=torch.float32)
    value[::2, 33] = 1
    value[1::2, 34] = 1
    return value


def _motion(frames: int, offset: float = 0.0) -> dict:
    pose = torch.zeros(frames, 66, dtype=torch.float32)
    transl = torch.zeros(frames, 3, dtype=torch.float32)
    transl[:, 0] = torch.arange(frames, dtype=torch.float32) / 100 + offset
    return {
        "pose": pose,
        "global_orient": pose[:, :3].clone(),
        "body_pose": pose[:, 3:].clone(),
        "transl": transl,
        "betas": torch.zeros(frames, 10, dtype=torch.float32),
    }


def _write_manifest_dataset(
    root: Path, dataset: str, samples: list[tuple[str, str, str, int, dict]]
) -> None:
    (root / "motions").mkdir(parents=True)
    (root / "musicfeat_v2").mkdir()
    (root / "manifests").mkdir()
    music_written: set[str] = set()
    rows = {split: [] for split in ("train", "val", "test")}
    for sample_id, group_id, music_name, frames, extras in samples:
        torch.save(_motion(frames, float(len(rows["train"]))), root / "motions" / f"{sample_id}.pt")
        if music_name not in music_written:
            torch.save(_music(frames), root / "musicfeat_v2" / f"{music_name}.pt")
            music_written.add(music_name)
        split = extras.pop("split", "train")
        row = {
            "sample_id": sample_id,
            "dataset": dataset,
            "group_id": group_id,
            "motion_path": f"motions/{sample_id}.pt",
            "music_feature_path": f"musicfeat_v2/{music_name}.pt",
            "fps": 30,
            "num_frames": frames,
            "split": split,
            **extras,
        }
        rows[split].append(row)
    for split, values in rows.items():
        (root / "manifests" / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in values), encoding="utf-8"
        )


def _write_four_roots(base: Path) -> dict[str, Path]:
    roots = {
        "aistpp": base / "AIST++",
        "aioz_gdance": base / "AIOZ-GDANCE",
        "finedance": base / "FineDance",
        "compas3d": base / "CoMPAS3D",
    }
    aist = roots["aistpp"]
    (aist / "musicfeat_v2").mkdir(parents=True)
    frames = 12
    pose = np.zeros((frames, 72), dtype=np.float32)
    transl = np.zeros((frames, 3), dtype=np.float32)
    annotation = {
        "gBR_sBM_cAll_d01_mBR0_ch01": {
            "smpl_pose_global": pose,
            "smpl_trans_global": transl,
        }
    }
    torch.save(annotation, aist / "annot_aist_30fps.pt")
    torch.save(["gBR_sBM_cAll_d01_mBR0_ch01"], aist / "train.pt")
    torch.save([], aist / "val.pt")
    torch.save([], aist / "test.pt")
    torch.save(
        _music(frames),
        aist / "musicfeat_v2" / "gBR_sBM_cAll_d01_mBR0_ch01_musicfeat_fps30.pt",
    )
    _write_manifest_dataset(
        roots["aioz_gdance"],
        "aioz_gdance",
        [
            ("group_dancer_00", "group", "group_music", 12, {"person_id": 0}),
            ("group_dancer_01", "group", "group_music", 12, {"person_id": 1}),
        ],
    )
    _write_manifest_dataset(
        roots["finedance"], "finedance", [("001", "001", "001_music", 12, {})]
    )
    _write_manifest_dataset(
        roots["compas3d"],
        "compas3d",
        [
            ("Pair1_song1_take1_leader", "Pair1_song1_take1", "pair_music", 12, {"role": "leader"}),
            ("Pair1_song1_take1_follower", "Pair1_song1_take1", "pair_music", 12, {"role": "follower"}),
        ],
    )
    return roots


def _export_args(roots: dict[str, Path], output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_root=str(output),
        export_id="fixture_v1",
        aist_root=str(roots["aistpp"]),
        aioz_root=str(roots["aioz_gdance"]),
        finedance_root=str(roots["finedance"]),
        compas3d_root=str(roots["compas3d"]),
        splits=["train", "val", "test"],
        review_coordinate="y_up",
        aist_forward_checks=0,
        seed=1,
        limit_per_dataset=None,
        overwrite=False,
    )


def _filled_decisions(export_root: Path, path: Path) -> None:
    master = read_jsonl(export_root / "index" / "master.jsonl")
    values = {
        "aistpp": "reject",
        "aioz_gdance__group_dancer_00": "reject",
        "aioz_gdance__group_dancer_01": "keep",
        "finedance": "keep",
        "compas3d": "reject",
    }
    rows = []
    for row in master:
        decision = values.get(row["review_id"], values.get(row["dataset"]))
        rows.append(
            {
                "export_id": row["export_id"],
                "review_id": row["review_id"],
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "duration_sec": f"{row['duration_sec']:.6f}",
                "decision": decision,
                "issue_codes": "jitter" if decision == "reject" else "",
                "reviewer": "unit-test",
                "notes": "",
            }
        )
    write_csv(path, rows, DECISION_COLUMNS)


def test_z_up_review_transform_preserves_rotation_and_pelvis_path() -> None:
    source = _motion(2)
    source["global_orient"][1] = torch.tensor([0.1, -0.2, 0.3])
    source["pose"][:, :3] = source["global_orient"]
    pelvis = torch.tensor([0.0, -0.3, 0.02])
    target = transform_z_up_to_y_up(source, pelvis)
    torch.testing.assert_close(
        axis_angle_to_matrix(target["global_orient"]),
        Z_UP_TO_Y_UP @ axis_angle_to_matrix(source["global_orient"]),
    )
    expected_translation = (
        Z_UP_TO_Y_UP @ (source["transl"] + pelvis).unsqueeze(-1)
    ).squeeze(-1) - pelvis
    torch.testing.assert_close(target["transl"], expected_translation)
    torch.testing.assert_close(target["body_pose"], source["body_pose"])


def test_review_export_decision_and_shared_music_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _write_four_roots(tmp_path / "sources")
    export_root = tmp_path / "review"
    monkeypatch.setattr(
        "tools.data.music_dance.curation.export_motion_review._neutral_pelvis_and_model",
        lambda: (torch.zeros(3), None),
    )
    report = export_package(_export_args(roots, export_root))
    assert report["sample_count"] == 6
    package = validate_package(export_root, verify_checksums=True, write_report=False)
    assert package["final_pass"]
    assert package["counts_by_dataset"] == {
        "aistpp": 1,
        "aioz_gdance": 2,
        "finedance": 1,
        "compas3d": 2,
    }
    assert not list(export_root.rglob("*.pt"))

    decisions = tmp_path / "decisions.csv"
    _filled_decisions(export_root, decisions)
    decision_report, _ = validate_decisions(
        export_root, decisions, strict=True, write_report=False
    )
    assert decision_report["decision_counts"] == {"reject": 4, "keep": 2}

    curated = tmp_path / "curated"
    dry = apply_results(
        argparse.Namespace(
            export_root=str(export_root),
            decisions=str(decisions),
            output_root=str(curated),
            dry_run=True,
            apply=False,
            quarantine_zero_ref_music=True,
        )
    )
    assert dry["zero_reference_music_count"] == 2
    assert not curated.exists()
    applied = apply_results(
        argparse.Namespace(
            export_root=str(export_root),
            decisions=str(decisions),
            output_root=str(curated),
            dry_run=False,
            apply=True,
            quarantine_zero_ref_music=True,
        )
    )
    assert applied["accepted_sample_count"] == 2
    assert applied["source_roots_modified"] is False

    # One rejected AIOZ dancer does not remove music still used by the kept dancer.
    assert (curated / "AIOZ-GDANCE/musicfeat_v2/group_music.pt").is_file()
    assert not (curated / "quarantine/AIOZ-GDANCE/musicfeat_v2/group_music.pt").exists()
    # Both CoMPAS roles were rejected, so their shared music has no remaining reference.
    assert not (curated / "CoMPAS3D/musicfeat_v2/pair_music.pt").exists()
    assert (curated / "quarantine/CoMPAS3D/musicfeat_v2/pair_music.pt").is_file()
    # Source artifacts remain intact and recoverable.
    assert (roots["compas3d"] / "musicfeat_v2/pair_music.pt").is_file()
    validated = validate_curated(curated, strict=True, loader_smoke=False)
    assert validated["final_pass"]
    assert validated["active_music_feature_count"] == 2


def test_strict_review_result_rejects_missing_unknown_and_unsure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _write_four_roots(tmp_path / "sources")
    export_root = tmp_path / "review"
    monkeypatch.setattr(
        "tools.data.music_dance.curation.export_motion_review._neutral_pelvis_and_model",
        lambda: (torch.zeros(3), None),
    )
    export_package(_export_args(roots, export_root))
    columns, rows = read_csv(export_root / "review" / "decisions.csv")
    assert set(DECISION_COLUMNS) <= set(columns)
    rows = rows[:-1]
    rows[0]["decision"] = "unsure"
    bad = tmp_path / "bad.csv"
    write_csv(bad, rows, DECISION_COLUMNS)
    with pytest.raises(ValueError, match="blank/unsure|missing"):
        validate_decisions(export_root, bad, strict=True, write_report=False)


def test_curated_experiment_composes_without_changing_condition_contract() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=["exp=gem_smpl_music_only_4set_curated"])
    assert list(cfg.train_datasets) == [
        "aistpp_train", "aioz_gdance_train", "finedance_train", "compas3d_train"
    ]
    assert list(cfg.pipeline.args.in_attr) == ["encoded_music"]
    assert list(cfg.pipeline.args.train_modes) == ["diffusion"]
    assert cfg.train_datasets.aistpp_train.root.endswith("/AIST++")
    assert cfg.test_datasets.aistpp_music_eval.root.endswith("/AIST++")

