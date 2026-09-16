"""验证四库人体 Z-up 交付的数据语义与完整性门禁。

测试仅在 pytest 的系统临时目录中构造小型四库数据，不依赖生产动作或人体模型。
重点验证旋转作用方向、平移轴变换、局部姿态不变、时变形状完整保留，以及源哈希、
坐标和 FPS 错误拒绝；完整打包测试还核对归档成员和拒绝覆盖行为。临时目录由
pytest 外层 TemporaryDirectory 统一清理，不向正式训练或数据目录写入文件。
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.data.music_dance.curation import package_smplx_zup as packager


def source_arrays(dataset: str = "aioz_gdance") -> tuple[dict, dict]:
    rng = np.random.default_rng(42)
    source = {
        "pose": rng.normal(0, 0.2, (4, 66)).astype(np.float32),
        "transl": np.asarray([[1, 2, 3], [-4, 5, 6], [0, -1, 2], [3, 2, 1]], dtype=np.float32),
        "betas": rng.normal(0, 0.2, (4, 10)).astype(np.float32),
        "fps": np.asarray(30.0, dtype=np.float32),
        "num_frames": np.asarray(4),
        "coordinate_system": np.asarray(packager.Y_UP),
        "dataset": np.asarray(dataset),
        "sample_id": np.asarray("dance"),
        "review_id": np.asarray(f"{dataset}__dance"),
    }
    row = {
        "dataset": dataset,
        "sample_id": "dance",
        "review_id": f"{dataset}__dance",
        "num_frames": 4,
        "fps": 30.0,
        "split": "train",
        "music_key": f"{dataset}::song",
        "review_motion_path": f"motions/{dataset}/dance.npz",
        "review_sha256": "pending",
    }
    return source, row


def make_source(root: Path) -> Path:
    rows = []
    for dataset in packager.COUNTS:
        source, row = source_arrays(dataset)
        path = root / row["review_motion_path"]
        path.parent.mkdir(parents=True)
        np.savez_compressed(path, **source)
        row["review_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
    (root / "index").mkdir()
    (root / "index/master.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return root


def test_rotation_direction_and_shape_preservation() -> None:
    source, row = source_arrays()
    output = packager.convert_motion(source, row)
    np.testing.assert_array_equal(output["trans"][0], [1, -3, 2])
    # 对任意向量核验主动旋转的左乘次序，防止轴角直接相加或右乘。
    vector = np.asarray([0.4, 0.8, -0.5])
    before = Rotation.from_rotvec(source["pose"][:, :3]).apply(vector)
    after = Rotation.from_rotvec(output["root_orient"]).apply(vector)
    np.testing.assert_allclose(
        after, np.stack([before[:, 0], -before[:, 2], before[:, 1]], axis=-1), atol=2e-7
    )
    np.testing.assert_array_equal(output["pose_body"], source["pose"][:, 3:])
    np.testing.assert_array_equal(output["source_betas"], source["betas"])
    np.testing.assert_array_equal(output["betas"][:10], source["betas"][0])
    assert output["source_betas_time_varying"]
    assert packager.validate_conversion(source, output)["root_matrix_max_abs_error"] < 5e-7


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fps", np.asarray(50.0), "fps"),
        ("coordinate_system", np.asarray(packager.Z_UP), "Y-up"),
        ("gender", np.asarray("female"), "neutral"),
    ],
)
def test_invalid_source_metadata_is_rejected(field: str, value: np.ndarray, message: str) -> None:
    source, row = source_arrays()
    source[field] = value
    with pytest.raises(ValueError, match=message):
        packager.convert_motion(source, row)


def test_corrupted_body_or_original_shape_is_rejected() -> None:
    source, row = source_arrays()
    output = packager.convert_motion(source, row)
    output["source_betas"][2, 0] += 0.1
    with pytest.raises(ValueError, match="per-frame betas"):
        packager.validate_conversion(source, output)


def test_full_package_and_archive_checks(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source")
    destination = tmp_path / "delivery"
    result = packager.package_dataset(
        source, destination, "four_set", dict.fromkeys(packager.COUNTS, 1)
    )
    assert result["status"] == "passed" and result["total_samples"] == 4
    assert result["total_frames"] == 16
    package = Path(result["package_root"])
    with tarfile.open(result["archive"], "r:gz") as archive:
        assert all(Path(name).parts[0] == "four_set" for name in archive.getnames())
    hashes = {
        p.relative_to(package).as_posix(): packager.sha256_file(p)
        for p in package.rglob("*")
        if p.is_file()
    }
    hashes["README.md"] = "0" * 64
    with pytest.raises(ValueError, match="archive SHA256 mismatch"):
        packager.verify_archive(Path(result["archive"]), "four_set", hashes)
    with pytest.raises(FileExistsError):
        packager.package_dataset(source, destination, "four_set", dict.fromkeys(packager.COUNTS, 1))
    assert not list(destination.glob(".*.staging-*"))


def test_source_checksum_failure_cleans_staging(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source")
    first = source / "motions/aistpp/dance.npz"
    with first.open("ab") as handle:
        handle.write(b"changed")
    destination = tmp_path / "delivery"
    with pytest.raises(ValueError, match="source SHA256 mismatch"):
        packager.package_dataset(source, destination, "four_set", dict.fromkeys(packager.COUNTS, 1))
    assert not list(destination.iterdir())


def test_complete_source_inventory_required(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source")
    with pytest.raises(ValueError, match="counts mismatch"):
        packager.package_dataset(source, tmp_path / "delivery", "four_set")
