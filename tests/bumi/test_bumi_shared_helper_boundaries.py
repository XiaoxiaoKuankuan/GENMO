"""回归验证当前 BUMI producer 只依赖中性共享 helper。

本测试不读取服务器数据，也不执行训练、仿真或控制器回放。它检查 UMR、
robot_retargeter、CSV 和 transfer filelist 的源码 import closure 不会重新指向已经退役的
legacy/SONIC producer，并确认运行时实际绑定的是中性 hash、manifest、配对、发布、
四元数、落地、质量判定和报告实现。测试还用临时目录执行这些关键路径，但不把离线
通过解释为动力学可跟踪或实机安全。
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gem.robots.bumi import motion_utils, quality_common
from gem.robots.bumi.kinematics import BumiKinematics
from tools.data.bumi import (
    build_bumi_music_dataset_from_csv as csv_builder,
)
from tools.data.bumi import (
    build_bumi_music_dataset_from_robot_retargeter_npz as robot_builder,
)
from tools.data.bumi import (
    build_bumi_transfer_filelists as transfer_filelists,
)
from tools.data.bumi import (
    dataset_publish_utils,
    npz_quality_utils,
    qpos_resample_utils,
    umr_qpos_adapter,
)
from tools.data.bumi import (
    filter_robot_retargeter_npz_motions as robot_filter,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CURRENT_PRODUCERS = (
    REPO_ROOT / "tools/data/bumi/umr_qpos_adapter.py",
    REPO_ROOT / "tools/data/bumi/filter_robot_retargeter_npz_motions.py",
    REPO_ROOT / "tools/data/bumi/build_bumi_music_dataset_from_robot_retargeter_npz.py",
    REPO_ROOT / "tools/data/bumi/build_bumi_music_dataset_from_csv.py",
    REPO_ROOT / "tools/data/bumi/build_bumi_transfer_filelists.py",
)
FORBIDDEN_PRODUCER_IMPORTS = {
    "gem.robots.bumi.legacy_motion",
    "gem.robots.bumi.quality_filter",
    "tools.data.bumi.build_bumi_music_dataset",
    "tools.data.bumi.build_bumi_music_dataset_from_sonic_npz",
    "tools.data.bumi.filter_sonic_npz_motions",
}
RETIRED_PRODUCTION_PATHS = (
    "configs/bumi/quality_filter_gmr_manual_q1_v3.yaml",
    "configs/exp/gem_bumi_music_only_5set_manual_q1_v3_qpos30_contact_50k.yaml",
    "configs/exp/gem_bumi_music_only_5set_manual_q1_v3_qpos30_contact_scratch_350k.yaml",
    "configs/pipeline/music_only_bumi_qpos30_contact_v2.yaml",
    "gem/robots/bumi/legacy_motion.py",
    "gem/robots/bumi/quality_filter.py",
    "scripts/build_bumi_hq_original_comparison.py",
    "scripts/export_smplx_to_bumi3_offline_npz.py",
    "tools/data/bumi/build_bumi_music_dataset.py",
    "tools/data/bumi/build_bumi_music_dataset_from_sonic_npz.py",
    "tools/data/bumi/filter_legacy_bumi_motions.py",
    "tools/data/bumi/filter_sonic_npz_motions.py",
    "tools/data/bumi/prepare_gmr_manual_q1_selected_root.py",
    "tools/eval/render_legacy_bumi_motion.py",
)


def test_retired_legacy_gmr_and_sonic_production_paths_stay_absent() -> None:
    """旧生产入口只能从 Git 历史复现，不能再次混入现行工作树。"""

    assert [
        relative for relative in RETIRED_PRODUCTION_PATHS if (REPO_ROOT / relative).exists()
    ] == []


def test_current_producer_import_closure_has_no_legacy_or_sonic_edge() -> None:
    """当前 producer 不能通过直接 import 再次依赖旧 producer。"""

    violations: list[str] = []
    for path in CURRENT_PRODUCERS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in FORBIDDEN_PRODUCER_IMPORTS:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}:{node.module}")
    assert violations == []


def test_current_producers_bind_the_neutral_implementations() -> None:
    """运行时绑定也必须指向中性模块，而不是复制或转调旧实现。"""

    assert umr_qpos_adapter.sha256_file is motion_utils.sha256_file
    assert umr_qpos_adapter.NumpyCompatibleUnpickler is motion_utils.NumpyCompatibleUnpickler
    assert umr_qpos_adapter.root_tilt_statistics is motion_utils.root_tilt_statistics
    assert umr_qpos_adapter.evaluate_motion is npz_quality_utils.evaluate_motion
    assert umr_qpos_adapter.build_summary is npz_quality_utils.build_summary
    assert umr_qpos_adapter.write_reports is npz_quality_utils.write_reports
    assert umr_qpos_adapter._central_difference is npz_quality_utils.central_difference
    assert umr_qpos_adapter._materialize is dataset_publish_utils.materialize_file

    assert robot_filter.sha256_file is motion_utils.sha256_file
    assert robot_filter.QualityStatus is quality_common.QualityStatus
    assert robot_filter.evaluate_motion is npz_quality_utils.evaluate_motion
    assert robot_filter.build_summary is npz_quality_utils.build_summary
    assert robot_filter.write_reports is npz_quality_utils.write_reports
    assert robot_builder.sha256_file is motion_utils.sha256_file
    assert robot_builder.root_tilt_statistics is motion_utils.root_tilt_statistics
    assert robot_builder._materialize is dataset_publish_utils.materialize_file
    assert robot_builder._music_tensor is dataset_publish_utils.load_music_tensor
    assert (
        robot_builder.make_quaternion_continuous_np
        is qpos_resample_utils.make_quaternion_continuous_np
    )

    assert csv_builder.sha256_file is motion_utils.sha256_file
    assert csv_builder._slerp_pairs is qpos_resample_utils.slerp_pairs
    assert (
        csv_builder.make_quaternion_continuous_np
        is qpos_resample_utils.make_quaternion_continuous_np
    )
    assert (
        csv_builder.normalize_body_origin_ground is qpos_resample_utils.normalize_body_origin_ground
    )

    assert transfer_filelists.DATASET_SPECS is dataset_publish_utils.DATASET_SPECS
    assert transfer_filelists._read_jsonl is dataset_publish_utils.read_jsonl
    assert transfer_filelists._mapping is dataset_publish_utils.require_dataset_mapping
    assert transfer_filelists.load_human_indices is dataset_publish_utils.load_human_indices


def test_neutral_file_tilt_quaternion_and_ground_helpers(
    tmp_path: Path, test_kinematics_path: Path
) -> None:
    """用小型输入确认迁移后的共享数值与发布基础行为。"""

    source = tmp_path / "source.bin"
    source.write_bytes(b"BUMI-neutral-helper-regression")
    assert motion_utils.sha256_file(source) == hashlib.sha256(source.read_bytes()).hexdigest()

    stats = motion_utils.root_tilt_statistics(np.asarray([0.0, 30.0, 60.0]))
    assert stats == {
        "num_frames": 3,
        "median_deg": 30.0,
        "p95_deg": pytest.approx(57.0),
        "max_deg": 60.0,
        "over_45deg_fraction": pytest.approx(1.0 / 3.0),
    }

    manifest = tmp_path / "nested/manifest.jsonl"
    rows = [{"sample_id": "a", "split": "train"}, {"sample_id": "b", "split": "test"}]
    dataset_publish_utils.write_jsonl(manifest, rows)
    assert dataset_publish_utils.read_jsonl(manifest) == rows
    destination = tmp_path / "published/source.bin"
    assert dataset_publish_utils.materialize_file(source, destination) in {"hardlink", "copy"}
    assert dataset_publish_utils.materialize_file(source, destination) == "existing"

    quaternion = np.asarray([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]])
    continuous = qpos_resample_utils.make_quaternion_continuous_np(quaternion)
    assert np.all(np.sum(continuous[1:] * continuous[:-1], axis=-1) >= 0.0)
    np.testing.assert_allclose(np.linalg.norm(continuous, axis=-1), 1.0)

    kinematics = BumiKinematics(test_kinematics_path)
    qpos = torch.zeros(4, 28)
    qpos[:, 2] = 1.0
    qpos[:, 3] = 1.0
    normalized, before, after = qpos_resample_utils.normalize_body_origin_ground(qpos, kinematics)
    assert before == pytest.approx(0.5)
    assert after == pytest.approx(0.0, abs=1.0e-6)
    torch.testing.assert_close(normalized[:, 2], torch.full((4,), 0.5))


def test_neutral_quality_intervals_keep_half_open_halo_semantics() -> None:
    """公共区间 helper 必须保持左闭右开和 halo 后最短片段语义。"""

    mask = np.zeros(400, dtype=np.bool_)
    mask[150:170] = True
    assert quality_common.mask_to_intervals(mask) == ((150, 170),)
    assert quality_common.safe_intervals_from_bad_mask(
        mask, halo_frames=15, minimum_frames=120
    ) == ((0, 135), (185, 400))


def test_current_umr_adapter_uses_neutral_central_difference() -> None:
    """直接执行当前 UMR 数组适配器，确认共享差分进入实际输出字段。"""

    kinematics = BumiKinematics(
        REPO_ROOT / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json"
    )
    config = robot_filter.load_config(robot_filter.DEFAULT_CONFIG)
    qpos = kinematics.default_qpos.repeat(6, 1)
    qpos[:, 0] = torch.arange(6, dtype=torch.float32) / 30.0

    arrays = umr_qpos_adapter.qpos_arrays(qpos, kinematics, config)

    expected_joint_velocity = npz_quality_utils.central_difference(arrays["joint_pos"], 30)
    expected_body_velocity = npz_quality_utils.central_difference(arrays["body_pos_w"], 30)
    np.testing.assert_array_equal(arrays["joint_vel"], expected_joint_velocity)
    np.testing.assert_array_equal(arrays["body_lin_vel_w"], expected_body_velocity)
    np.testing.assert_allclose(arrays["body_lin_vel_w"][:, 0, 0], 1.0, atol=1.0e-6)


def test_current_robot_builder_reorders_named_joints_with_neutral_quaternion() -> None:
    """直接执行当前 robot_retargeter builder 的 qpos30 生产路径。"""

    kinematics = BumiKinematics(
        REPO_ROOT / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json"
    )
    source_order = tuple(reversed(kinematics.joint_order))
    frames = 3
    body_pos = np.zeros((frames, 22, 3), dtype=np.float32)
    body_pos[:, 0, 2] = np.asarray([0.7, 0.8, 0.9], dtype=np.float32)
    body_quat = np.zeros((frames, 22, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    body_quat[1, 0, 0] = -1.0
    arrays = {
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "joint_pos": np.repeat(np.arange(21, dtype=np.float32)[None], frames, axis=0),
    }

    qpos = robot_builder.qpos30_from_npz(arrays, source_order, kinematics)

    assert qpos.shape == (frames, 28)
    torch.testing.assert_close(qpos[:, 2], torch.tensor([0.7, 0.8, 0.9]))
    torch.testing.assert_close(qpos[:, 3], torch.ones(frames))
    torch.testing.assert_close(qpos[:, 7:], torch.arange(20, -1, -1).repeat(frames, 1).float())


def test_current_csv_builder_uses_neutral_slerp_in_50hz_resample() -> None:
    """直接执行当前 CSV producer，覆盖最短弧四元数和 50→30 Hz 插值。"""

    source = np.zeros((6, 28), dtype=np.float64)
    source[:, 0] = np.arange(6, dtype=np.float64) / 50.0
    source[:, 3] = 1.0
    source[1::2, 3] = -1.0
    source[:, 3:7] = csv_builder.make_quaternion_continuous_np(source[:, 3:7])

    qpos = csv_builder.resample_qpos_to_30hz(source, source_fps=50, target_frames=4)

    torch.testing.assert_close(qpos[:, 0], torch.tensor([0.0, 1 / 30, 2 / 30, 0.1]))
    torch.testing.assert_close(qpos[:, 3], torch.ones(4))
    torch.testing.assert_close(qpos[:, 4:7], torch.zeros(4, 3))


def test_neutral_report_writer_preserves_expected_artifacts(tmp_path: Path) -> None:
    """共享报告器仍原子生成旧入口承诺的完整文件集合。"""

    row = {
        "dataset": "aistpp",
        "sample_id": "aistpp/sample",
        "source_relative_path": "aistpp/sample.npz",
        "source_sha256": "a" * 64,
        "status": quality_common.QualityStatus.PASS.value,
        "status_without_joint_limit": quality_common.QualityStatus.PASS.value,
        "quality_accepted": True,
        "reason_codes": [],
        "metrics": {"num_frames": 4},
        "error_type": None,
        "error_message": None,
    }
    config_path = tmp_path / "quality.yaml"
    config_path.write_text("contract_version: synthetic\n", encoding="utf-8")
    output = tmp_path / "reports"
    npz_quality_utils.write_reports(
        output,
        [row],
        {"report_contract_version": "synthetic.v1", "sequences": 1},
        config_path,
        overwrite=False,
    )

    assert {path.name for path in output.iterdir()} == set(npz_quality_utils.REPORT_FILENAMES)
    assert (output / "strict_pass.txt").read_text(encoding="utf-8") == "aistpp/sample.npz\n"
    assert (
        json.loads((output / "quality_report.jsonl").read_text(encoding="utf-8"))[
            "quality_accepted"
        ]
        is True
    )
    with pytest.raises(FileExistsError, match="--overwrite"):
        npz_quality_utils.write_reports(
            output,
            [row],
            {"report_contract_version": "synthetic.v1", "sequences": 1},
            config_path,
            overwrite=False,
        )
