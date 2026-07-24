from __future__ import annotations

import json

import pytest

from tests.data.motionxpp_fixtures import make_motionxpp_root
from tools.data.motionxpp.common import build_asset_index, parse_keypoint_asset
from tools.data.motionxpp.inspect_motionxpp import inspect_motionxpp


@pytest.mark.parametrize("zipped", [True, False])
def test_inspect_zip_and_extracted_directory(tmp_path, zipped):
    root = make_motionxpp_root(tmp_path / "Motion-Xplusplus", zipped=zipped)
    output = tmp_path / "report"
    result = inspect_motionxpp(root, output, sample_count=1)
    assert result["pairing"]["total_motion"] == 2
    assert result["pairing"]["total_text"] == 2
    assert result["pairing"]["total_motion_text_pairs"] == 2
    assert result["recommended"] == ["toy"]
    assert (output / "inventory.json").is_file()
    assert (output / "pairing_report.json").is_file()
    assert (output / "schema_report.json").is_file()
    assert (output / "overlap_report.json").is_file()
    assert (output / "recommended_subsets.txt").read_text().strip() == "toy"
    schema = json.loads((output / "schema_report.json").read_text())
    assert schema["keypoint_training_conclusion"]["condition_on_keypoints"] is False


def test_keypoint_schema_has_coco17_but_no_camera(tmp_path):
    root = make_motionxpp_root(tmp_path / "Motion-Xplusplus")
    index = build_asset_index(root, "keypoints", "toy")
    audited = parse_keypoint_asset(index.assets["walk_clip1"])
    assert audited["kp2d"].shape == (40, 17, 3)
    assert audited["confidence_values"] == [0.0, 1.0]
    assert audited["image_size"] is None
    assert audited["has_camera_intrinsics"] is False
