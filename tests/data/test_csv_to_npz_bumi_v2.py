"""验证 BUMI v2 CSV 转换的关节映射、时间轴和旋转速度，不启动 Isaac Sim。

使用具有解析解的平移/旋转轨迹检验 30→50 Hz 插值，覆盖四元数异号、乱序表头、
单帧/两帧短动作和错误输入。真实 articulation 的连杆位置及质心速度另以实际
CSV 转换后对照 URDF 运动学核验，避免只检查数组形状就声称运动学正确。
"""

import importlib.util
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


def test_headerless_isaaclab_order(tmp_path):
    expected = trajectory(2)
    source = expected.copy()
    source[:, 7:] = expected[:, [7 + converter.RETARGET_JOINT_NAMES.index(n)
                                 for n in converter.ISAACLAB_JOINT_NAMES]]
    path = tmp_path / "motion.csv"
    np.savetxt(path, source, delimiter=",")
    np.testing.assert_allclose(converter.load_motion(path, csv_joint_order="isaaclab"), expected)


def test_linear_motion_and_shortest_arc_world_angular_velocity():
    sampled = converter.resample_motion(trajectory())
    times = np.arange(50) / 50.0
    assert sampled["joint_pos"].shape == (50, 21)
    np.testing.assert_allclose(sampled["root_pos"], [0.2, -0.1, 0.5] + times[:, None] * [0.3, -0.2, 0.1], atol=1e-7)
    np.testing.assert_allclose(sampled["joint_vel"], np.broadcast_to(np.linspace(-0.2, 0.2, 21), (50, 21)), atol=1e-6)
    np.testing.assert_allclose(sampled["root_lin_vel"], np.broadcast_to([0.3, -0.2, 0.1], (50, 3)), atol=1e-6)
    np.testing.assert_allclose(sampled["root_ang_vel"], np.broadcast_to([0.1, 0.2, 0.3], (50, 3)), atol=1e-6)
    actual = Rotation.from_quat(sampled["root_quat"][:, [1, 2, 3, 0]])
    expected = Rotation.from_rotvec(times[:, None] * [0.1, 0.2, 0.3])
    assert np.max((actual * expected.inv()).magnitude()) < 1e-7
    assert np.all(np.sum(sampled["root_quat"][1:] * sampled["root_quat"][:-1], axis=1) > 0)


@pytest.mark.parametrize("frames,expected_count", [(1, 1), (2, 2), (4, 5), (120, 199), (10000, 16665)])
def test_short_sequences_and_half_open_timeline(frames, expected_count):
    sampled = converter.resample_motion(trajectory(frames))
    assert all(len(value) == expected_count and np.isfinite(value).all() for value in sampled.values())
    if frames == 1:
        for key in ("joint_vel", "root_lin_vel", "root_ang_vel"):
            np.testing.assert_array_equal(sampled[key], 0.0)
    else:
        assert (expected_count - 1) / 50 < (frames - 1) / 30
        assert expected_count / 50 >= (frames - 1) / 30


def test_invalid_inputs_fail_before_simulator_launch(tmp_path):
    path = tmp_path / "motion.csv"
    data = trajectory(3)
    np.savetxt(path, data, delimiter=",")
    with pytest.raises(ValueError, match="frame_range"):
        converter.load_motion(path, (1, 4))
    with pytest.raises(ValueError, match="帧率"):
        converter.resample_motion(data, input_fps=0)
    data[0, 3:7] = 0
    np.savetxt(path, data, delimiter=",")
    with pytest.raises(ValueError, match="四元数"):
        converter.load_motion(path)
    data[0, 0] = np.nan
    np.savetxt(path, data, delimiter=",")
    with pytest.raises(ValueError, match="有限"):
        converter.load_motion(path)
