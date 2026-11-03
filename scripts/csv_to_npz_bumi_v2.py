"""将 BUMI v2 的 30 Hz CSV 轨迹转换为 Isaac Lab 运动跟踪使用的 50 Hz NPZ。

参照 legged_lab/scripts/csv_to_npz_bumi_4340.py 的数据流程，加载独立的
robots/bumi_v2/urdf/bumi.urdf，并通过 Isaac Lab 运动学更新获得全部连杆状态。
输入为根位置 xyz、根四元数 xyzw、21 个关节角（米、弧度）；有表头时按名称
匹配，支持 leg.pitch.l 等重定向列名及 l_leg_pitch_joint 等 URDF 关节名。
无表头时默认使用左腿、右腿、腰、左臂、右臂的 retarget 顺序。

平移和关节角采用线性插值，根旋转采用归一化、最短弧 SLERP；速度在目标时间轴
计算。与参考脚本一致，采样区间为 [0, (输入帧数-1)/输入帧率)，不外推末帧；
单帧输入原样输出，速度为零。--frame_range 使用从 1 开始、两端包含的帧索引。

NPZ 保存 fps、joint_names、body_names、joint_pos、joint_vel、body_pos_w、
body_quat_w、body_lin_vel_w、body_ang_vel_w。输出关节/连杆顺序取实际加载的
Isaac Lab articulation，四元数为 wxyz，速度在世界系。body_pos_w 是连杆原点，
body_lin_vel_w 按参考脚本的 Isaac Lab 字段语义记录连杆质心线速度。根速度则按
CSV 根连杆原点写入，由 Isaac Lab 完成质心转换，避免旋转时混淆两个参考点。

脚本只回放指定状态，不推进物理、不修正轨迹、不上传数据；转换完成自动退出。
需在已安装 Isaac Sim / Isaac Lab、NumPy、SciPy 的环境运行，例如：
python scripts/csv_to_npz_bumi_v2.py --input_file /path/motion.csv --headless \
    --legged_lab_root /home/weili/legged_lab --save_to /path/motion_50hz.npz
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
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
ISAACLAB_JOINT_NAMES = (
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
    elif csv_joint_order == "isaaclab":
        motion[:, 7:] = motion[:, [7 + ISAACLAB_JOINT_NAMES.index(name) for name in RETARGET_JOINT_NAMES]]
    elif csv_joint_order != "retarget":
        raise ValueError("csv_joint_order 必须为 retarget 或 isaaclab")
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
    """保持原时间轴重采样；根角速度使用世界系相对旋转的中心/单边差分。"""
    if input_fps <= 0 or output_fps <= 0:
        raise ValueError("输入、输出帧率必须为正整数")
    frames = len(motion)
    # 整数计算 ceil，避免浮点 arange 在整倍数时意外增减一帧。
    count = max(1, ((frames - 1) * output_fps + input_fps - 1) // input_fps)
    times = np.arange(count, dtype=np.float64) / output_fps
    if frames == 1:
        sampled = motion.copy()
    else:
        source_frame = times * input_fps
        left = np.floor(source_frame).astype(np.int64)
        right = np.minimum(left + 1, frames - 1)
        blend = (source_frame - left)[:, None]
        sampled = motion[left] * (1.0 - blend) + motion[right] * blend
        rotations = Rotation.from_quat(motion[:, 3:7])
        sampled[:, 3:7] = Slerp(np.arange(frames) / input_fps, rotations)(times).as_quat()
    # 连续选取四元数符号，避免等价 q/-q 在下游差分或显示中产生符号跳变。
    quats = sampled[:, 3:7]
    flips = np.where(np.sum(quats[1:] * quats[:-1], axis=1) < 0.0, -1.0, 1.0)
    quats[1:] *= np.cumprod(flips)[:, None]
    velocities = np.zeros_like(sampled)
    angular = np.zeros((count, 3))
    if count > 1:
        velocities = np.gradient(sampled, 1.0 / output_fps, axis=0)
        previous = np.maximum(np.arange(count) - 1, 0)
        following = np.minimum(np.arange(count) + 1, count - 1)
        rotations = Rotation.from_quat(quats)
        relative = rotations[following] * rotations[previous].inv()
        angular = relative.as_rotvec() / ((following - previous) / output_fps)[:, None]
    return {
        "root_pos": sampled[:, :3].astype(np.float32),
        "root_quat": quats[:, [3, 0, 1, 2]].astype(np.float32),
        "root_lin_vel": velocities[:, :3].astype(np.float32),
        "root_ang_vel": angular.astype(np.float32),
        "joint_pos": sampled[:, 7:].astype(np.float32),
        "joint_vel": velocities[:, 7:].astype(np.float32),
    }


def convert_in_isaaclab(args, motion, output_path, usd_dir):
    """使用真正的 BUMI v2 articulation 输出位置、旋转及质心速度，不执行物理步。"""
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass
    from NoetixRobot.assets.robots.bumi_v2.bumi import BUMI_21DOF_CFG

    robot_cfg = BUMI_21DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_cfg.spawn.usd_dir = usd_dir
    robot_cfg.spawn.usd_file_name = "bumi_v2.usd"

    @configclass
    class ReplaySceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg = robot_cfg
        ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=750.0))

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / args.output_fps, device=args.device))
    sim.set_camera_view(eye=(2.0, 2.0, 1.2), target=(0.0, 0.0, 0.5))
    scene = InteractiveScene(ReplaySceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]
    if len(robot.joint_names) != 21 or set(robot.joint_names) != set(RETARGET_JOINT_NAMES):
        raise ValueError(f"实际加载的 BUMI v2 关节不匹配：{robot.joint_names}")
    indexes = robot.find_joints(list(RETARGET_JOINT_NAMES), preserve_order=True)[0]
    tensors = {key: torch.as_tensor(value, device=sim.device) for key, value in motion.items()}
    count = len(motion["joint_pos"])
    body_count = len(robot.body_names)
    arrays = {key: np.empty((count, 21), dtype=np.float32) for key in ("joint_pos", "joint_vel")}
    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        arrays[key] = np.empty((count, body_count, 4 if key == "body_quat_w" else 3), dtype=np.float32)
    root = robot.data.default_root_state.clone()
    joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    for frame in range(count):
        root[:, :3] = tensors["root_pos"][frame]
        root[:, 3:7] = tensors["root_quat"][frame]
        root[:, 7:10] = tensors["root_lin_vel"][frame]
        root[:, 10:] = tensors["root_ang_vel"][frame]
        robot.write_root_link_state_to_sim(root)
        joint_pos[:, indexes] = tensors["joint_pos"][frame]
        joint_vel[:, indexes] = tensors["joint_vel"][frame]
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.forward()
        if not args.headless:
            sim.render()
        scene.update(sim.get_physics_dt())
        for key, value in arrays.items():
            value[frame] = getattr(robot.data, key)[0].cpu().numpy()
        if frame == 0 or (frame + 1) % 1000 == 0 or frame + 1 == count:
            print(f"[转换] {frame + 1}/{count}", flush=True)
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Isaac Lab 输出包含非有限数值，未保存 NPZ")
    np.savez_compressed(
        output_path, **arrays, fps=np.asarray([args.output_fps]),
        joint_names=np.asarray(robot.joint_names), body_names=np.asarray(robot.body_names),
        robot_model=np.asarray("bumi_v2"), source_fps=np.asarray([args.input_fps]),
        quaternion_order=np.asarray("wxyz"), body_linear_velocity_point=np.asarray("center_of_mass"),
    )
    print(f"[完成] {output_path}\n[输出] {count} 帧，{args.output_fps} Hz，播放时长 {count / args.output_fps:.3f} 秒", flush=True)


def main():
    """沿用参考脚本主要参数，转换前校验输入，转换结束关闭仿真并清理临时 USD。"""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_fps", type=int, default=30)
    parser.add_argument("--output_fps", type=int, default=50)
    parser.add_argument("--frame_range", nargs=2, type=int, metavar=("START", "END"))
    parser.add_argument("--save_to", type=Path, help="输出目录或完整 .npz 路径；默认输入文件所在目录")
    parser.add_argument("--output_name", help="输出文件名，不含 .npz；默认原名加输出帧率后缀")
    parser.add_argument("--root_z_offset", type=float, default=0.0)
    parser.add_argument("--csv_joint_order", choices=("retarget", "isaaclab"), default="retarget", help="仅无表头时使用；有表头按列名映射")
    parser.add_argument("--legged_lab_root", type=Path, default=Path.home() / "legged_lab", help="包含 BUMI v2 资产的 legged_lab 根目录")
    parser.add_argument("--overwrite", action="store_true", help="允许替换已存在的目标 NPZ")
    AppLauncher.add_app_launcher_args(parser)
    # AppLauncher 内部会预解析参数；必选项后加，保证单独 --help 可以正常退出。
    parser.add_argument("--input_file", type=Path, required=True, help="输入 BUMI v2 CSV")
    args = parser.parse_args()
    motion = load_motion(args.input_file, args.frame_range, args.csv_joint_order, args.root_z_offset)
    sampled = resample_motion(motion, args.input_fps, args.output_fps)
    package_root = args.legged_lab_root.expanduser().resolve() / "source/NoetixRobot"
    urdf = package_root / "NoetixRobot/assets/robots/bumi_v2/urdf/bumi.urdf"
    if not urdf.is_file():
        parser.error(f"找不到 BUMI v2 URDF：{urdf}；请设置 --legged_lab_root")
    sys.path.insert(0, str(package_root))
    output = args.save_to.expanduser() if args.save_to else args.input_file.expanduser().parent
    if output.suffix.lower() != ".npz":
        name = args.output_name or f"{args.input_file.stem}_{args.output_fps}hz"
        output /= name if name.endswith(".npz") else name + ".npz"
    output = output.resolve()
    if output.exists() and not args.overwrite:
        parser.error(f"目标文件已存在：{output}；如需替换，请使用 --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[输入] {len(motion)} 帧，{args.input_fps} Hz；重采样为 {len(sampled['joint_pos'])} 帧")
    app = AppLauncher(args).app
    try:
        with tempfile.TemporaryDirectory(prefix="bumi_v2_csv_usd_") as usd_dir:
            convert_in_isaaclab(args, sampled, output, usd_dir)
    finally:
        # 文件及临时 USD 已关闭/清理；避免默认退出再次初始化 Replicator 后长时间阻塞。
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
