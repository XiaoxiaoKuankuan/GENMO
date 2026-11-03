"""验证 BUMI v2 CSV 到 SuperMimic UMR NPZ 的数据契约与 30→50 Hz 插值。

用解析平移、旋转和关节轨迹检查四元数异号、关节重排、CSV 乱序表头和末帧保留，
并覆盖单帧/过短动作拒绝、非有限输入、默认输出位置及覆盖保护。
测试只依赖 NumPy、SciPy、pytest，不启动仿真；SuperMimic 原生 read_umr_npz、
MotionLib 和最新 bumi_v2 XML 的兼容性另用实际太极 CSV 做完整加载验证。
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/csv_to_npz_bumi_v2.py"
spec = importlib.util.spec_from_file_location("csv_to_npz_bumi_v2", SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def trajectory(frames=31):
    times = np.arange(frames) / 30.0
    motion = np.zeros((frames, 28))
    motion[:, :3] = np.array([0.2, -0.1, 0.5]) + times[:, None] * [0.3, -0.2, 0.1]
    motion[:, 3:7] = Rotation.from_rotvec(times[:, None] * [0.1, 0.2, 0.3]).as_quat()
    motion[1::2, 3:7] *= -1.0
    motion[:, 7:] = np.arange(21) / 100 + times[:, None] * np.linspace(-0.2, 0.2, 21)
    return motion


def test_named_csv_reorders_all_columns_and_normalizes_quaternions(tmp_path):
    motion = trajectory()
    motion[:, 3:7] *= 1.001
    inverse_aliases = {name: alias for alias, name in converter.CSV_ALIASES.items()}
    names = list(converter.ROOT_COLUMNS) + [inverse_aliases[n] for n in converter.RETARGET_JOINT_NAMES]
    permutation = np.random.default_rng(42).permutation(28)
    path = tmp_path / "太极.csv"
    np.savetxt(path, motion[:, permutation], delimiter=",", encoding="utf-8-sig",
               header=",".join(names[i] for i in permutation), comments="")
    loaded = converter.load_motion(path, (2, 7), root_z_offset=0.05)
    expected = trajectory()[1:7]
    expected[:, 2] += 0.05
    np.testing.assert_allclose(loaded, expected, atol=1e-12)


def test_headerless_supermimic_order(tmp_path):
    expected = trajectory(2)
    source = expected.copy()
    source[:, 7:] = expected[:, [7 + converter.RETARGET_JOINT_NAMES.index(n)
                                 for n in converter.SUPERMIMIC_JOINT_NAMES]]
    path = tmp_path / "motion.csv"
    np.savetxt(path, source, delimiter=",")
    np.testing.assert_allclose(converter.load_motion(path, csv_joint_order="supermimic"), expected)


def test_linear_motion_shortest_arc_and_named_joint_order():
    qpos = converter.resample_motion(trajectory())
    times = np.arange(51) / 50.0
    assert qpos.shape == (51, 28)
    assert qpos.dtype == np.float32
    np.testing.assert_allclose(qpos[:, :3], [0.2, -0.1, 0.5] + times[:, None] * [0.3, -0.2, 0.1], atol=1e-7)
    expected_q = np.arange(21) / 100 + times[:, None] * np.linspace(-0.2, 0.2, 21)
    order = [converter.RETARGET_JOINT_NAMES.index(n) for n in converter.SUPERMIMIC_JOINT_NAMES]
    np.testing.assert_allclose(qpos[:, 7:], expected_q[:, order], atol=1e-7)
    actual = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])
    expected = Rotation.from_rotvec(times[:, None] * [0.1, 0.2, 0.3])
    assert np.max((actual * expected.inv()).magnitude()) < 1e-7
    assert np.all(np.sum(qpos[1:, 3:7] * qpos[:-1, 3:7], axis=1) > 0)


@pytest.mark.parametrize("frames,expected_count", [(2, 2), (4, 6), (31, 51), (120, 199), (10000, 16666)])
def test_supermimic_timeline_includes_aligned_endpoint(frames, expected_count):
    qpos = converter.resample_motion(trajectory(frames))
    assert qpos.shape == (expected_count, 28)
    assert np.isfinite(qpos).all()
    assert (expected_count - 1) / 50 <= (frames - 1) / 30
    assert expected_count / 50 > (frames - 1) / 30


def test_equal_fps_keeps_all_frames():
    motion = trajectory(51)
    qpos = converter.resample_motion(motion, input_fps=50)
    assert len(qpos) == len(motion)
    np.testing.assert_allclose(qpos[:, :3], motion[:, :3], atol=1e-7)
    expected = Rotation.from_quat(motion[:, 3:7])
    actual = Rotation.from_quat(qpos[:, [4, 5, 6, 3]])
    assert np.max((actual * expected.inv()).magnitude()) < 1e-7


def test_invalid_inputs(tmp_path):
    path = tmp_path / "motion.csv"
    data = trajectory(3)
    np.savetxt(path, data, delimiter=",")
    with pytest.raises(ValueError, match="frame_range"):
        converter.load_motion(path, (1, 4))
    with pytest.raises(ValueError, match="帧率"):
        converter.resample_motion(data, input_fps=0)
    with pytest.raises(ValueError, match="至少需要两帧"):
        converter.resample_motion(converter.load_motion(path, (2, 2)))
    with pytest.raises(ValueError, match="不足"):
        converter.resample_motion(data, input_fps=1000)
    data[0, 3:7] = 0
    np.savetxt(path, data, delimiter=",")
    with pytest.raises(ValueError, match="四元数"):
        converter.load_motion(path)
    data[0, 0] = np.nan
    np.savetxt(path, data, delimiter=",")
    with pytest.raises(ValueError, match="有限"):
        converter.load_motion(path)


def test_cli_writes_supermimic_contract_and_protects_existing_file(tmp_path, monkeypatch):
    source = tmp_path / "太极.csv"
    np.savetxt(source, trajectory(), delimiter=",")
    argv = [str(SCRIPT), "--input_file", str(source)]
    monkeypatch.setattr(sys, "argv", argv)
    converter.main()
    output = tmp_path / "太极_50hz.npz"
    with np.load(output, allow_pickle=False) as data:
        assert data["qpos"].shape == (51, 28)
        assert data["qpos"].dtype == np.float32
        np.testing.assert_array_equal(data["qpos"][0, 3:7], [1, 0, 0, 0])
        assert float(data["fps"][0]) == 50.0
        assert data["robot_name"].item() == "bumi_v2"
        assert tuple(data["robot_joint_names"]) == converter.SUPERMIMIC_JOINT_NAMES
        assert data["robot_joint_names"].dtype.kind == "U"
        assert data["source_frames"].tolist() == [31]
        assert data["source_fps"].tolist() == [30]
        assert data["quaternion_order"].item() == "wxyz"
        assert all(not data[name].dtype.hasobject for name in data.files)
    original = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        converter.main()
    assert error.value.code == 2
    assert output.read_bytes() == original
    monkeypatch.setattr(sys, "argv", argv + ["--overwrite", "--frame_range", "1", "4"])
    converter.main()
    with np.load(output, allow_pickle=False) as data:
        assert data["qpos"].shape == (6, 28)
