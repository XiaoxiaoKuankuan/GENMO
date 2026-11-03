"""将 BUMI v2 的 30 Hz CSV 轨迹转换为 SuperMimic 可读取的 50 Hz UMR NPZ。

参照 legged_lab/scripts/csv_to_npz_bumi_4340.py 的 CSV 读取和重采样思路，
输出契约以 supermimic_vibed/supermimic/motion_cache.py 的 read_umr_npz 为准。
输入为根位置 xyz、根四元数 xyzw、21 个关节角（米、弧度）；有表头时按名称
匹配，支持 leg.pitch.l 等重定向列名及 l_leg_pitch_joint 等 URDF 关节名。
无表头时默认使用左腿、右腿、腰、左臂、右臂的 retarget 顺序；可显式指定
supermimic 规范顺序。根坐标和关节角保持原值，仅按显式参数添加根高度偏移。

平移和关节角采用线性插值，根旋转采用归一化、最短弧 SLERP，并保持四元数符号
连续。按 SuperMimic NPZ 加载器约定，目标帧数为 floor((N-1)*目标Hz/源Hz)+1，
保留落在目标时间网格上的末帧，不外推；输入与输出均至少两帧，供下游计算速度。
--frame_range 使用从 1 开始、两端包含的帧索引；默认帧率为 30→50 Hz。

NPZ 核心字段为 qpos[T,28] float32、fps[1]、robot_joint_names[21] 和
robot_name='bumi_v2'。qpos 依次为世界系根位置 xyz、根四元数 wxyz，以及按
robot_joint_names 标注顺序排列的关节角；该顺序不是 MJCF 内部树遍历顺序。
字符串使用 Unicode 数组，读取无需 pickle；额外保存源帧率、源帧数及四元数顺序。
速度与连杆正运动学交给 SuperMimic 按其当前 bumi_v2 XML 计算，避免混入另一套
URDF 的连杆位置、惯性或质心速度。本脚本不进行物理修正，也不构建训练缓存。

仅依赖 NumPy、SciPy，使用 CPU，无需 Isaac Sim、GPU 或 SuperMimic 的 Python 环境。
示例：python scripts/csv_to_npz_bumi_v2.py --input_file /path/motion.csv
    --input_fps 30 --output_fps 50 --save_to /path/motion_50hz.npz
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT_COLUMNS = (
    "root_pos_x", "root_pos_y", "root_pos_z",
    "root_rot_x", "root_rot_y", "root_rot_z", "root_rot_w",
)
RETARGET_JOINT_NAMES = tuple(
    f"{side}_{part}_joint"
    for side in ("l", "r")
    for part in ("leg_pitch", "leg_roll", "leg_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")
) + ("waist_yaw_joint",) + tuple(
    f"{side}_{part}_joint"
    for side in ("l", "r")
    for part in ("arm_pitch", "arm_roll", "arm_yaw", "elbow_pitch")
)
SUPERMIMIC_JOINT_NAMES = (
    "l_leg_pitch_joint", "r_leg_pitch_joint", "waist_yaw_joint",
    "l_leg_roll_joint", "r_leg_roll_joint", "l_arm_pitch_joint", "r_arm_pitch_joint",
    "l_leg_yaw_joint", "r_leg_yaw_joint", "l_arm_roll_joint", "r_arm_roll_joint",
    "l_knee_pitch_joint", "r_knee_pitch_joint", "l_arm_yaw_joint", "r_arm_yaw_joint",
    "l_ankle_pitch_joint", "r_ankle_pitch_joint", "l_elbow_pitch_joint",
    "r_elbow_pitch_joint", "l_ankle_roll_joint", "r_ankle_roll_joint",
)
CSV_ALIASES = {
    ".".join((*name.split("_")[1:-1], name.split("_")[0])): name
    for name in RETARGET_JOINT_NAMES if name != "waist_yaw_joint"
}
CSV_ALIASES["waist.yaw"] = "waist_yaw_joint"


def load_motion(path, frame_range=None, csv_joint_order="retarget", root_z_offset=0.0):
    """读取并一次性校验 CSV，返回统一 retarget 关节顺序及 xyzw 根四元数。"""
    path = Path(path).expanduser()
    with path.open(encoding="utf-8-sig", newline="") as file:
        first = next(csv.reader(file), [])
    if len(first) != 28:
        raise ValueError(f"CSV 应有 28 列（根位置3+四元数4+关节21），实际 {len(first)} 列")
    try:
        [float(value) for value in first]
        has_header = False
    except ValueError:
        has_header = True
    motion = np.loadtxt(path, delimiter=",", skiprows=int(has_header), ndmin=2, encoding="utf-8-sig")
    if motion.shape[0] == 0 or motion.shape[1] != 28 or not np.isfinite(motion).all():
        raise ValueError("CSV 必须包含非空、有限的 28 列数值轨迹")
    if has_header:
        names = [CSV_ALIASES.get(value.strip(), value.strip()) for value in first]
        expected = ROOT_COLUMNS + RETARGET_JOINT_NAMES
        if len(set(names)) != 28 or set(names) != set(expected):
            raise ValueError(f"CSV 表头与 BUMI v2 不匹配：缺少 {set(expected)-set(names)}，多出 {set(names)-set(expected)}")
        motion = motion[:, [names.index(name) for name in expected]]
    elif csv_joint_order == "supermimic":
        motion[:, 7:] = motion[:, [7 + SUPERMIMIC_JOINT_NAMES.index(name) for name in RETARGET_JOINT_NAMES]]
    elif csv_joint_order != "retarget":
        raise ValueError("csv_joint_order 必须为 retarget 或 supermimic")
    if frame_range is not None:
        start, end = frame_range
        if not 1 <= start <= end <= len(motion):
            raise ValueError(f"frame_range 必须满足 1 <= START <= END <= {len(motion)}")
        motion = motion[start - 1:end].copy()
    if not np.isfinite(root_z_offset):
        raise ValueError("root_z_offset 必须为有限数值")
    motion[:, 2] += root_z_offset
    norms = np.linalg.norm(motion[:, 3:7], axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("CSV 包含零长度根四元数")
    motion[:, 3:7] /= norms
    return motion


def resample_motion(motion, input_fps=30, output_fps=50):
    """重采样为规范关节顺序、wxyz 四元数的 qpos，时间轴与 SuperMimic 一致。"""
    if input_fps <= 0 or output_fps <= 0:
        raise ValueError("输入、输出帧率必须为正整数")
    frames = len(motion)
    if frames < 2:
        raise ValueError("SuperMimic 动作至少需要两帧，不能计算单帧动作的速度")
    # 整数计算 floor+1；末帧落在目标时间网格上时必须保留。
    count = (frames - 1) * output_fps // input_fps + 1
    if count < 2:
        raise ValueError("动作时长不足以生成目标帧率下的两帧参考")
    times = np.arange(count, dtype=np.float64) / output_fps
    source_frame = times * input_fps
    left = np.minimum(np.floor(source_frame).astype(np.int64), frames - 2)
    blend = np.clip(source_frame - left, 0.0, 1.0)[:, None]
    sampled = motion[left] * (1.0 - blend) + motion[left + 1] * blend
    rotations = Rotation.from_quat(motion[:, 3:7])
    sampled[:, 3:7] = Slerp(np.arange(frames) / input_fps, rotations)(times).as_quat()
    # 连续选取四元数符号，避免等价 q/-q 在下游差分或显示中产生符号跳变。
    quats = sampled[:, 3:7]
    flips = np.where(np.sum(quats[1:] * quats[:-1], axis=1) < 0.0, -1.0, 1.0)
    quats[1:] *= np.cumprod(flips)[:, None]
    columns = [0, 1, 2, 6, 3, 4, 5] + [
        7 + RETARGET_JOINT_NAMES.index(name) for name in SUPERMIMIC_JOINT_NAMES
    ]
    return sampled[:, columns].astype(np.float32)


def main():
    """校验并重采样 CSV，在 CPU 上写出 SuperMimic 动作文件。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_file", type=Path, required=True, help="输入 BUMI v2 CSV")
    parser.add_argument("--input_fps", type=int, default=30)
    parser.add_argument("--output_fps", type=int, default=50)
    parser.add_argument("--frame_range", nargs=2, type=int, metavar=("START", "END"))
    parser.add_argument("--save_to", type=Path, help="输出目录或完整 .npz 路径；默认输入文件所在目录")
    parser.add_argument("--output_name", help="输出文件名，不含 .npz；默认原名加输出帧率后缀")
    parser.add_argument("--root_z_offset", type=float, default=0.0)
    parser.add_argument("--csv_joint_order", choices=("retarget", "supermimic"), default="retarget", help="仅无表头时使用；有表头按列名映射")
    parser.add_argument("--overwrite", action="store_true", help="允许替换已存在的目标 NPZ")
    args = parser.parse_args()
    motion = load_motion(args.input_file, args.frame_range, args.csv_joint_order, args.root_z_offset)
    sampled = resample_motion(motion, args.input_fps, args.output_fps)
    output = args.save_to.expanduser() if args.save_to else args.input_file.expanduser().parent
    if output.suffix.lower() != ".npz":
        name = args.output_name or f"{args.input_file.stem}_{args.output_fps}hz"
        output /= name if name.endswith(".npz") else name + ".npz"
    output = output.resolve()
    if output.exists() and not args.overwrite:
        parser.error(f"目标文件已存在：{output}；如需替换，请使用 --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, qpos=sampled, fps=np.asarray([args.output_fps], dtype=np.float32),
        robot_joint_names=np.asarray(SUPERMIMIC_JOINT_NAMES), robot_name=np.asarray("bumi_v2"),
        source_fps=np.asarray([args.input_fps], dtype=np.float32),
        source_frames=np.asarray([len(motion)], dtype=np.int64), quaternion_order=np.asarray("wxyz"),
    )
    print(f"[输入] {len(motion)} 帧，{args.input_fps} Hz")
    print(f"[输出] SuperMimic / bumi_v2，qpos {sampled.shape}，{args.output_fps} Hz；"
          f"首末采样跨度 {(len(sampled) - 1) / args.output_fps:.3f} 秒")
    print(f"[完成] {output}")


if __name__ == "__main__":
    main()
