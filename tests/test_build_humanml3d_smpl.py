"""CPU-only tests for the HumanML3D SMPL-X build tool."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from gem.utils.rotation_conversions import axis_angle_to_matrix
from tools.data.humanml3d.build_humanml3d_smpl import (
    PREFIX_TRIM_SECONDS,
    convert_crop_bounds,
    group_text_annotations,
    main,
    make_output_record,
    make_segment_key,
    mirror_smpl_pose,
    mirror_translation,
    parse_text_line,
    prefix_trim_seconds,
    slice_motion_tensor,
    validate_output_record,
)


def test_time_conversion_20fps_to_30fps() -> None:
    crop = convert_crop_bounds(0, 100, "kit", 20, 30, 1000)
    assert crop.start == 0
    assert crop.end == 150
    assert crop.expected_frames == 150
    assert crop.actual_frames == 150


@pytest.mark.parametrize(
    ("family", "collapsed", "seconds"),
    [
        ("eyes_japan_dataset", "eyesjapandataset", 3.0),
        ("mpi_hdm05", "mpihdm05", 3.0),
        ("totalcapture", "totalcapture", 1.0),
        ("mpi_limits", "mpilimits", 1.0),
        ("transitions_mocap", "transitionsmocap", 0.5),
    ],
)
def test_official_prefix_trim_families(
    family: str, collapsed: str, seconds: float
) -> None:
    assert PREFIX_TRIM_SECONDS[family] == seconds
    assert prefix_trim_seconds(family) == seconds
    assert prefix_trim_seconds(collapsed) == seconds
    crop = convert_crop_bounds(0, 100, collapsed, 20, 30, 1000)
    assert crop.start == round(seconds * 30)
    assert crop.end == round((seconds + 5.0) * 30)


def test_text_line_parse_and_full_grouping() -> None:
    first = parse_text_line("a person walks#walk/VERB person/NOUN#0#0")
    second = parse_text_line("someone strolls#someone/PRON stroll/VERB#0.0#0.0")
    assert first.caption == "a person walks"
    assert first.tokens == ["walk/VERB", "person/NOUN"]
    grouped = group_text_annotations([first, second])
    assert list(grouped) == [(0.0, 0.0)]
    assert len(grouped[(0.0, 0.0)]) == 2


def test_subclip_grouping_and_deterministic_key() -> None:
    first = parse_text_line("turn left#turn/VERB left/ADV#1.5#4.2")
    second = parse_text_line("a left turn#a/DET left/ADJ turn/NOUN#1.5#4.2")
    third = parse_text_line("wave#wave/VERB#4.2#5.0")
    grouped = group_text_annotations([first, second, third])
    assert len(grouped[(1.5, 4.2)]) == 2
    assert len(grouped[(4.2, 5.0)]) == 1
    assert make_segment_key("000004", 1.5, 4.2) == "000004__seg_1500_4200"
    assert make_segment_key("000004", 1.5, 4.2) == make_segment_key(
        "000004", 1.5, 4.2
    )


def test_mirror_swaps_left_and_right_joint_rotations() -> None:
    pose = torch.zeros(1, 66)
    pose[0, 3:6] = torch.tensor([0.2, -0.3, 0.4])  # left hip only
    mirrored = mirror_smpl_pose(pose).reshape(1, 22, 3)
    mirrored_rot = axis_angle_to_matrix(mirrored)
    source_rot = axis_angle_to_matrix(pose.reshape(1, 22, 3))
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    expected_right = reflection @ source_rot[:, 1] @ reflection
    assert torch.allclose(mirrored_rot[:, 2], expected_right, atol=1e-6)
    assert torch.allclose(mirrored_rot[:, 1], torch.eye(3).unsqueeze(0), atol=1e-6)


def test_double_mirror_recovers_rotation_matrices() -> None:
    generator = torch.Generator().manual_seed(123)
    pose = torch.randn(12, 66, generator=generator) * 0.4
    restored = mirror_smpl_pose(mirror_smpl_pose(pose))
    original_rot = axis_angle_to_matrix(pose.reshape(-1, 22, 3))
    restored_rot = axis_angle_to_matrix(restored.reshape(-1, 22, 3))
    assert torch.max(torch.abs(original_rot - restored_rot)).item() < 1e-4


def test_mirror_translation_only_flips_x() -> None:
    trans = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    mirrored = mirror_translation(trans)
    assert torch.equal(mirrored[:, 0], -trans[:, 0])
    assert torch.equal(mirrored[:, 1:], trans[:, 1:])
    assert torch.equal(mirror_translation(mirrored), trans)


def test_sliced_output_does_not_share_large_source_storage() -> None:
    source = torch.randn(1000, 66)
    sliced = slice_motion_tensor(source, 100, 120)
    assert sliced.shape == (20, 66)
    assert sliced.is_contiguous()
    assert sliced.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    source[100:120].zero_()
    assert torch.count_nonzero(sliced) > 0


def test_output_record_contract() -> None:
    annotation = parse_text_line("walk forward#walk/VERB forward/ADV#0#0")
    record = make_output_record(
        torch.zeros(30, 66),
        torch.zeros(30, 3),
        torch.arange(16, dtype=torch.float64),
        "neutral",
        [annotation],
    )
    validate_output_record(record, "motion")
    assert set(record) == {"pose", "trans", "beta", "gender", "text_data"}
    assert record["pose"].shape == (30, 66)
    assert record["trans"].shape == (30, 3)
    assert record["beta"].shape == (10,)
    assert record["beta"].dtype == torch.float32
    assert record["text_data"] == [
        {"caption": "walk forward", "tokens": ["walk/VERB", "forward/ADV"]}
    ]


def _write_synthetic_inputs(root: Path) -> tuple[Path, Path, Path]:
    humanml = root / "HumanML3D_official"
    dataset = humanml / "HumanML3D"
    texts = dataset / "texts"
    texts.mkdir(parents=True)
    (humanml / "index.csv").write_text(
        "source_path,start_frame,end_frame,new_name\n./KIT/a,0,100,000001.npy\n",
        encoding="utf-8",
    )
    (dataset / "train.txt").write_text("000001\nM000001\n", encoding="utf-8")
    text = (
        "a person walks#person/NOUN walk/VERB#0#0\n"
        "the person waves#person/NOUN wave/VERB#1#3\n"
    )
    (texts / "000001.txt").write_text(text, encoding="utf-8")
    (texts / "M000001.txt").write_text(text, encoding="utf-8")

    amass_path = root / "amass.pth"
    torch.save(
        {
            "amass/key": {
                "pose": torch.zeros(180, 66),
                "trans": torch.zeros(180, 3),
                "beta": torch.zeros(10),
                "gender": "neutral",
            }
        },
        amass_path,
    )
    mapping_path = root / "mapping.csv"
    fields = [
        "new_name",
        "source_path",
        "start_frame",
        "end_frame",
        "in_train",
        "match_status",
        "amass_key",
        "normalized_family",
    ]
    with mapping_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "new_name": "000001.npy",
                "source_path": "./KIT/a",
                "start_frame": 0,
                "end_frame": 100,
                "in_train": True,
                "match_status": "exact_family_path",
                "amass_key": "amass/key",
                "normalized_family": "kit",
            }
        )
    return humanml, amass_path, mapping_path


def test_dry_run_checks_everything_without_writing_main_pth(tmp_path: Path) -> None:
    humanml, amass, mapping = _write_synthetic_inputs(tmp_path)
    output = tmp_path / "result" / "train.pth"
    report = tmp_path / "report"
    assert (
        main(
            [
                "--humanml-root",
                str(humanml),
                "--amass-file",
                str(amass),
                "--mapping-csv",
                str(mapping),
                "--output",
                str(output),
                "--report-dir",
                str(report),
                "--dry-run",
            ]
        )
        == 0
    )
    assert not output.exists()
    summary = json.loads((report / "build_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "dry_run_complete"
    assert summary["total_records"] == 4  # original/mirror full + one subclip each
    assert summary["output_size_bytes"] == 0

