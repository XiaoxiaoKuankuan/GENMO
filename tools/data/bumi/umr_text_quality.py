"""UMR 文本动作的逐帧数值与运动学质量评估核心。

本模块复用音乐筛选器 evaluate_motion 的限位、连续性与姿态统计，使用真实 BUMI
XML 的 MuJoCo FK 补充完整脚网格穿地、独立支撑候选滑移、悬空和凸包自碰撞指标。
机器人关节按名称映射，并取 XML 与 UMR 生成配置限位的交集，避免遗漏双膝0.1rad
下界。每个批处理 worker 常驻一个模型，逐帧 FK 不启动渲染、不使用 GPU。

文本完整动作默认只记录坐躺等姿态，不执行音乐库的站立风格淘汰。所有坏帧区间为
左闭右开，仅供定位，不裁剪动作或复用整段 caption 标注局部动作。碰撞为凸包近似，
脚滑为网格高度与垂向速度推定支撑后的诊断，PASS不代表动力学或实机质量验收。
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

from gem.robots.bumi.kinematics import BumiKinematics, sha256_file
from gem.robots.bumi.quality_filter import mask_to_intervals
from tools.data.bumi.filter_sonic_npz_motions import (
    SonicNpzQualityConfig,
    _angular_speed_wxyz,
    _central_difference,
    _longest_true_run,
    _signal_metrics,
    evaluate_motion,
)

CONFIG_SCHEMA = "genmo.bumi_umr_text_quality_config.v1"
POSTURE_CODES = {"FLOOR_STYLE_SUSTAINED", "FLOOR_STYLE_FRAGMENTED", "LOW_ROOT_REVIEW"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_rules(path):
    raw = yaml.safe_load(Path(path).read_text())
    require(raw["schema"] == CONFIG_SCHEMA, "不支持的UMR文本质量配置")
    require(raw["training_frames"] == [60, 300], "文本分支要求完整60..300帧")
    require(raw["posture_policy"] in {"diagnostic", "standing"}, "未知姿态策略")
    require(
        raw["source_up"] == "y" and raw["source_format"] == "motionmillion_272",
        "此适配器需要明确的MotionMillion Y-up源契约",
    )
    require(np.isfinite(raw["ground_height_m"]), "地面高度必须有限")
    require(
        set(raw["dynamics"])
        == {
            "joint_velocity_l2",
            "joint_acceleration_l2",
            "joint_jerk_l2",
            "root_linear_velocity",
            "root_angular_velocity",
        },
        "导数指标必须完整",
    )
    require(
        set(raw["feet"])
        == {
            "penetration_review_m",
            "penetration_reject_m",
            "penetration_severe_m",
            "penetration_consecutive_frames",
            "support_height_min_m",
            "support_height_max_m",
            "support_vertical_speed_max_m_s",
            "support_min_frames",
            "slide_review_m_s",
            "slide_reject_m_s",
            "slide_consecutive_frames",
            "airborne_height_m",
            "airborne_review_frames",
        },
        "脚部配置字段必须完整",
    )
    require(
        set(raw["collision"])
        == {
            "review_extra_depth_m",
            "reject_extra_depth_m",
            "consecutive_frames",
            "allowed_body_pairs",
        },
        "碰撞配置字段必须完整",
    )
    for name, values in raw["dynamics"].items():
        require(
            len(values) == 2 and np.isfinite(values[0]) and values[0] > 0 and values[1],
            f"非法导数阈值: {name}",
        )
    for group in ("core", "feet", "collision"):
        for name, value in raw[group].items():
            if isinstance(value, list):
                continue
            require(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and np.isfinite(value),
                f"非法阈值: {group}.{name}",
            )
            if name.endswith("frames"):
                require(isinstance(value, int) and value >= 1, f"帧阈值必须为正整数: {name}")
    require(raw["core"]["minimum_frames"] >= 4, "导数检查至少四帧")
    require(0 <= raw["core"]["exceed_ratio_max"] <= 1, "超限比例必须在0..1")
    for group, review, reject in (
        ("feet", "penetration_review_m", "penetration_reject_m"),
        ("feet", "slide_review_m_s", "slide_reject_m_s"),
        ("collision", "review_extra_depth_m", "reject_extra_depth_m"),
    ):
        require(0 < raw[group][review] <= raw[group][reject], f"{group}阈值顺序错误")
    require(
        raw["feet"]["support_height_min_m"] < raw["feet"]["support_height_max_m"],
        "支撑高度区间错误",
    )
    require(
        raw["feet"]["penetration_severe_m"] >= raw["feet"]["penetration_reject_m"]
        and raw["feet"]["airborne_height_m"] > 0
        and raw["feet"]["support_vertical_speed_max_m_s"] > 0,
        "脚部阈值非法",
    )
    return raw


def verify_asset_files(robot_xml, kinematics, asset_manifest, rules):
    """主进程核验XML和清单中的全部网格；输出指纹用于续跑与发布绑定。"""
    xml, kin, manifest_path = map(
        lambda p: Path(p).resolve(strict=True), (robot_xml, kinematics, asset_manifest)
    )
    manifest = json.loads(manifest_path.read_text())
    require(manifest["format"] == "umr_bumi3_asset_manifest_v1", "UMR资产清单格式错误")
    require((manifest_path.parent / manifest["xml"]).resolve() == xml, "资产清单XML路径错误")
    actual = {}
    for relative, expected in manifest["files"].items():
        path = (manifest_path.parent / relative).resolve(strict=True)
        require(path.is_relative_to(manifest_path.parent), "资产路径越界")
        digest = sha256_file(path)
        require(digest == expected, f"资产指纹不匹配: {relative}")
        actual[relative] = digest
    require(sha256_file(xml) == rules["robot_xml_sha256"], "XML与质量配置指纹不同")
    require(sha256_file(kin) == rules["kinematics_sha256"], "运动学与质量配置指纹不同")
    return dict(
        robot_xml_sha256=sha256_file(xml),
        kinematics_sha256=sha256_file(kin),
        asset_manifest_sha256=sha256_file(manifest_path),
        asset_files=actual,
    )


class QualityEngine:
    """每个worker常驻的真实模型与共享音乐评估器适配器。"""

    def __init__(self, rules, robot_xml, kinematics, retarget_config, batch_config=None):
        self.rules = rules
        self.xml = Path(robot_xml).resolve(strict=True)
        self.kin = BumiKinematics(kinematics)
        require(
            self.kin.source_mjcf_sha256 == rules["robot_xml_sha256"]
            and sha256_file(self.xml) == rules["robot_xml_sha256"],
            "XML身份不一致",
        )
        require(self.kin.kinematics_sha256 == rules["kinematics_sha256"], "运动学身份不一致")
        original = json.loads(Path(retarget_config).read_text())
        robot = original["robot"]
        batch = json.loads(Path(batch_config).read_text()) if batch_config else {}
        require(not batch.get("robot"), "batch中有robot覆盖项，需先显式合并生成配置")
        require(
            (Path(retarget_config).resolve().parent / robot["xml"]).resolve() == self.xml,
            "UMR配置引用了不同XML",
        )
        require(
            robot.get("name") == "bumi3" and robot.get("to_smpl_frame") is True,
            "需要已核验的BUMI3 UMR坐标配置",
        )
        motion = {**original.get("motion", {}), **batch.get("motion", {})}
        require(
            all(
                motion.get(k, v) == v
                for k, v in {"start": 0, "end": -1, "stride": 1, "max_frames": 0}.items()
            ),
            "文本完整动作适配器不接受截帧/降采样生成配置",
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.xml))
        require((self.model.nq, self.model.nv, self.model.nu) == (28, 27, 21), "模型维度错误")
        self.data = mujoco.MjData(self.model)
        names = self.kin.joint_order
        self.addresses = [int(self.model.jnt_qposadr[self.model.joint(n).id]) for n in names]
        require(self.addresses == list(range(7, 28)), "MuJoCo与运动学关节顺序不一致")
        lower, upper = [], []
        overrides = robot.get("joint_limits", {})
        require(set(overrides) <= set(names), "生成配置包含未知限位关节")
        for i, name in enumerate(names):
            lo, hi = self.model.jnt_range[self.model.joint(name).id]
            require(
                np.allclose(
                    [lo, hi],
                    [float(self.kin.joint_lower_limits[i]), float(self.kin.joint_upper_limits[i])],
                    atol=1e-6,
                ),
                f"XML与运动学限位不一致: {name}",
            )
            if name in overrides:
                a, b = map(float, overrides[name])
                require(not np.isnan([a, b]).any(), "自定义限位含NaN")
                lo, hi = max(lo, a), min(hi, b)
            require(lo < hi, f"有效关节限位无交集: {name}")
            lower.append(lo)
            upper.append(hi)
        self.config = SonicNpzQualityConfig(
            motion_contract_version="umr.bumi_text_qpos30.v1",
            fps=30,
            required_keys=(),
            robot_xml_sha256=rules["robot_xml_sha256"],
            preset_sha256=sha256_file(retarget_config),
            kinematics_sha256=self.kin.kinematics_sha256,
            joint_order=names,
            joint_lower_limits=np.asarray(lower),
            joint_upper_limits=np.asarray(upper),
            body_order=self.kin.body_order,
            dynamics=rules["dynamics"],
            **rules["core"],
        )
        self.body_ids = [self.model.body(name).id for name in self.kin.body_order]
        for group in (
            self.config.torso_proxy_bodies,
            self.config.upper_non_hand_bodies,
            self.config.ankle_bodies,
        ):
            require(bool(group) and set(group) <= set(self.kin.body_order), "质量配置刚体名称错误")
        self.feet = []
        for name in self.config.ankle_bodies:
            body = self.model.body(name).id
            geoms = [
                g
                for g in range(self.model.ngeom)
                if self.model.geom_bodyid[g] == body
                and self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            ]
            require(len(geoms) == 1, f"脚网格数量错误: {name}")
            geom = geoms[0]
            mid = self.model.geom_dataid[geom]
            start, count = self.model.mesh_vertadr[mid], self.model.mesh_vertnum[mid]
            vertices = np.array(self.model.mesh_vert[start : start + count], dtype=np.float64)
            self.feet.append((geom, vertices, vertices.mean(axis=0)))
        allowed = rules["collision"]["allowed_body_pairs"]
        self.allowed = set()
        for pair in allowed:
            require(len(pair) == 2 and pair[0] != pair[1], "允许碰撞对格式错误")
            for name in pair:
                self.model.body(name)  # 未知名称必须失败。
            self.allowed.add(" / ".join(sorted(pair)))
        mujoco.mj_forward(self.model, self.data)
        self.reference_depths = self.collision_depths()

    def collision_depths(self):
        depths = {}
        model = self.model
        for contact in self.data.contact:
            if contact.dist >= -0.001:
                continue
            a, b = [int(model.geom_bodyid[g]) for g in contact.geom]
            if (
                a == b
                or min(a, b) == 0
                or model.body_parentid[a] == b
                or model.body_parentid[b] == a
            ):
                continue
            key = " / ".join(sorted((model.body(a).name, model.body(b).name)))
            depths[key] = max(depths.get(key, 0.0), -float(contact.dist))
        return depths

    def evaluate(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float64)
        n = len(qpos)
        body = np.empty((n, 22, 3))
        quat = np.empty((n, 22, 4))
        heights, centers = np.empty((n, 2)), np.empty((n, 2, 3))
        collision_series = {}
        for frame, pose in enumerate(qpos):
            self.data.qpos[:] = pose
            mujoco.mj_forward(self.model, self.data)
            body[frame], quat[frame] = self.data.xpos[self.body_ids], self.data.xquat[self.body_ids]
            for side, (geom, vertices, center) in enumerate(self.feet):
                rotation = self.data.geom_xmat[geom].reshape(3, 3)
                position = self.data.geom_xpos[geom]
                heights[frame, side] = np.min(vertices @ rotation[2]) + position[2]
                centers[frame, side] = center @ rotation.T + position
            for pair, depth in self.collision_depths().items():
                if pair not in collision_series:
                    collision_series[pair] = np.zeros(n)
                collision_series[pair][frame] = depth
        # 保留输入四元数符号，核心最短弧算法会正确处理 q/-q。
        quat[:, 0] = qpos[:, 3:7]
        arrays = dict(
            joint_pos=qpos[:, 7:],
            joint_vel=_central_difference(qpos[:, 7:], 30),
            body_pos_w=body,
            body_quat_w=quat,
            body_lin_vel_w=_central_difference(body, 30),
        )
        decision = evaluate_motion(arrays, self.config)
        reasons = dict(decision["reason_statuses"])
        diagnostic_reasons = {}
        if self.rules["posture_policy"] == "diagnostic":
            for code in POSTURE_CODES:
                if code in reasons:
                    diagnostic_reasons[code] = reasons.pop(code)
        bad = np.zeros(n, dtype=bool)
        issues = {}

        def mark(code, level, mask):
            nonlocal bad
            reasons[code] = level
            bad |= mask
            issues[code] = [list(x) for x in mask_to_intervals(mask)]

        # 旧核心的 valid_intervals 仅覆盖floor，不能冒充完整坏帧并集。
        for code in list(reasons):
            mask = np.zeros(n, dtype=bool)
            if code == "SOURCE_JOINT_LIMIT":
                v = np.maximum(
                    self.config.joint_lower_limits - qpos[:, 7:],
                    qpos[:, 7:] - self.config.joint_upper_limits,
                )
                mask = (v > self.config.joint_limit_violation_max).any(axis=1)
            elif code == "ROOT_HEIGHT_BELOW_ABSOLUTE_BOUND":
                mask = qpos[:, 2] < self.config.root_height_min_absolute
            elif code == "ROOT_HEIGHT_ABOVE_ABSOLUTE_BOUND":
                mask = qpos[:, 2] > self.config.root_height_max_absolute
            elif code in POSTURE_CODES:
                if code == "LOW_ROOT_REVIEW":
                    mask = qpos[:, 2] < self.config.low_root_review_height
                else:
                    for a, b in decision["floor_intervals"]:
                        mask[a:b] = True
            else:
                for name, (threshold, _) in self.config.dynamics.items():
                    if not code.startswith(name.upper()):
                        continue
                    if name == "root_angular_velocity":
                        values, order = _angular_speed_wxyz(qpos[:, 3:7], 30), 1
                    elif name == "root_linear_velocity":
                        values, order = np.linalg.norm(np.diff(qpos[:, :3], axis=0) * 30, axis=1), 1
                    else:
                        order = {
                            "joint_velocity_l2": 1,
                            "joint_acceleration_l2": 2,
                            "joint_jerk_l2": 3,
                        }[name]
                        values = np.linalg.norm(
                            np.diff(qpos[:, 7:], n=order, axis=0) * 30**order, axis=1
                        )
                    for offset in range(order + 1):
                        mask[offset : offset + len(values)] |= values > threshold
            mark(code, reasons[code], mask)
        foot_metrics, foot_flags = foot_diagnostics(heights, centers, self.rules)
        # 文本包含躺/跪/手撑等动作；双脚离地不等于全身悬空。这里只用非足部
        # body原点的低高度作为保守的支撑迹象，不把它当作精确接触标签。
        nonfoot = [
            i for i, name in enumerate(self.kin.body_order) if name not in self.config.ankle_bodies
        ]
        root_quat = qpos[:, 3:7] / np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
        tilt_cos = 1 - 2 * (root_quat[:, 1] ** 2 + root_quat[:, 2] ** 2)
        low_posture = (
            qpos[:, 2] - self.rules["ground_height_m"] < self.config.floor_gate_root_height
        ) | (tilt_cos < np.cos(np.deg2rad(self.config.floor_gate_tilt_degrees)))
        nonfoot_support = low_posture & (
            np.min(body[:, nonfoot, 2], axis=1) - self.rules["ground_height_m"]
            <= self.config.upper_body_ground_height
        )
        foot_metrics["nonfoot_support_inferred_fraction"] = float(nonfoot_support.mean())
        for code, level, mask in foot_flags:
            if code == "LONG_AIRBORNE_REVIEW" and self.rules["posture_policy"] == "diagnostic":
                mask = mask & ~nonfoot_support
                if _longest_true_run(mask) < self.rules["feet"]["airborne_review_frames"]:
                    diagnostic_reasons["FEET_AIRBORNE_WITH_NONFOOT_SUPPORT"] = "REVIEW"
                    continue
            mark(code, level, mask)
        collision_metrics = {}
        cfg = self.rules["collision"]
        for pair, depth in collision_series.items():
            baseline = self.reference_depths.get(pair, 0.0)
            extra = np.maximum(depth - baseline, 0)
            collision_metrics[pair] = dict(
                max_depth_m=float(depth.max()),
                reference_depth_m=baseline,
                extra_depth=_signal_metrics(extra, cfg["review_extra_depth_m"]),
                allowed=pair in self.allowed,
            )
            if pair in self.allowed:
                continue
            for level, threshold in (
                ("REVIEW", cfg["review_extra_depth_m"]),
                ("REJECT", cfg["reject_extra_depth_m"]),
            ):
                mask = extra > threshold
                if _longest_true_run(mask) >= cfg["consecutive_frames"]:
                    mark(f"SELF_COLLISION_{pair}", level, mask)
        metrics = decision["metrics"]
        metrics.update(
            feet=foot_metrics,
            collisions=collision_metrics,
            reference_collision_depths=self.reference_depths,
            joint_near_limit_fraction_by_name={
                name: float(
                    np.mean(
                        np.minimum(
                            qpos[:, i + 7] - self.config.joint_lower_limits[i],
                            self.config.joint_upper_limits[i] - qpos[:, i + 7],
                        )
                        < 0.01
                    )
                )
                for i, name in enumerate(self.config.joint_order)
            },
            repeated_pose_pair_fraction=float(
                np.mean(np.max(np.abs(np.diff(qpos, axis=0)), axis=1) < 1e-7)
            ),
            root_travel_m=float(np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1).sum()),
        )
        status = "REJECT" if "REJECT" in reasons.values() else "REVIEW" if reasons else "PASS"
        return dict(
            status=status,
            metrics=metrics,
            reason_codes=list(reasons),
            reason_statuses=reasons,
            diagnostic_reasons=diagnostic_reasons,
            issue_intervals=issues,
            bad_intervals=[list(x) for x in mask_to_intervals(bad)],
            full_sequence=True,
            crop_count=0,
        )


def foot_diagnostics(heights, centers, rules):
    """支撑候选独立于水平速度，防止先排除滑动再宣称无脚滑。"""
    cfg, fps = rules["feet"], 30
    heights = np.asarray(heights) - rules["ground_height_m"]
    n = len(heights)
    speed = np.linalg.norm(np.diff(centers[..., :2], axis=0), axis=-1) * fps
    vertical = np.abs(np.diff(centers[..., 2], axis=0)) * fps
    result, flags = {}, []
    for side, label in enumerate(("left", "right")):
        h = heights[:, side]
        low = (h >= cfg["support_height_min_m"]) & (h <= cfg["support_height_max_m"])
        candidate = (
            low[1:] & low[:-1] & (vertical[:, side] <= cfg["support_vertical_speed_max_m_s"])
        )
        support = np.zeros_like(candidate)
        for a, b in mask_to_intervals(candidate):
            if b - a + 1 >= cfg["support_min_frames"]:
                support[a:b] = True
        result[label] = dict(
            min_surface_height_m=float(h.min()),
            max_surface_height_m=float(h.max()),
            support_edge_fraction=float(support.mean()),
            slide=_signal_metrics(speed[support, side], cfg["slide_review_m_s"]),
            support_displacement_m=float(speed[support, side].sum() / fps),
        )
        result[label]["slide"]["max_consecutive_exceed_frames"] = _longest_true_run(
            support & (speed[:, side] > cfg["slide_review_m_s"])
        )
        for level, threshold in (
            ("REVIEW", cfg["penetration_review_m"]),
            ("REJECT", cfg["penetration_reject_m"]),
        ):
            mask = h < -threshold
            if _longest_true_run(mask) >= cfg["penetration_consecutive_frames"]:
                flags.append((f"FOOT_PENETRATION_{label}_{level}", level, mask))
        severe = h < -cfg["penetration_severe_m"]
        if severe.any():
            flags.append((f"FOOT_PENETRATION_{label}_SEVERE", "REJECT", severe))
        for level, threshold in (
            ("REVIEW", cfg["slide_review_m_s"]),
            ("REJECT", cfg["slide_reject_m_s"]),
        ):
            edges = support & (speed[:, side] > threshold)
            if _longest_true_run(edges) >= cfg["slide_consecutive_frames"]:
                mask = np.zeros(n, dtype=bool)
                mask[:-1] |= edges
                mask[1:] |= edges
                flags.append((f"FOOT_SLIDE_{label}_{level}", level, mask))
    airborne = (heights > cfg["airborne_height_m"]).all(axis=1)
    result["both_airborne"] = dict(
        fraction=float(airborne.mean()), longest_frames=_longest_true_run(airborne)
    )
    if _longest_true_run(airborne) >= cfg["airborne_review_frames"]:
        flags.append(("LONG_AIRBORNE_REVIEW", "REVIEW", airborne))
    return result, flags
