"""Synthetic tests for the independent GEM2 SMPL-direct path."""

from __future__ import annotations

import copy
import json
import math
import socket
import unittest
from pathlib import Path

import numpy as np

from gem.gmr_udp_bridge import (
    GEM2_MAGIC,
    GEM2_VERSION,
    HEADER,
    MAGIC,
    PACKET_BYTES,
    PAYLOAD,
    VERSION,
    GMRUDPBridge,
)
from gem.smpl_direct_adapter import (
    AXIS_CONVERT_AY_TO_ZUP,
    TARGET_JOINT_INDICES,
    TARGET_NAMES,
    SMPLDirectAdapter,
    TargetPose,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/gmr/smpl_direct_e1_adapter.json"


def standing_joints_zup() -> np.ndarray:
    joints = np.zeros((22, 3), dtype=np.float64)
    values = {
        0: (0.0, 0.0, 1.00),
        1: (0.0, 0.12, 0.95),
        2: (0.0, -0.12, 0.95),
        4: (0.0, 0.12, 0.55),
        5: (0.0, -0.12, 0.55),
        7: (0.0, 0.12, 0.10),
        8: (0.0, -0.12, 0.10),
        9: (0.0, 0.0, 1.35),
        10: (0.20, 0.12, 0.05),
        11: (0.20, -0.12, 0.05),
        12: (0.0, 0.0, 1.55),
        16: (0.0, 0.25, 1.40),
        17: (0.0, -0.25, 1.40),
        18: (0.0, 0.25, 1.05),
        19: (0.0, -0.25, 1.05),
        20: (0.0, 0.25, 0.75),
        21: (0.0, -0.25, 0.75),
    }
    for index, value in values.items():
        joints[index] = value
    return joints


def t_pose_joints_zup() -> np.ndarray:
    joints = standing_joints_zup()
    joints[18] = (0.0, 0.60, 1.40)
    joints[19] = (0.0, -0.60, 1.40)
    joints[20] = (0.0, 0.90, 1.40)
    joints[21] = (0.0, -0.90, 1.40)
    return joints


def squat_joints_zup() -> np.ndarray:
    joints = standing_joints_zup()
    upper = (0, 1, 2, 9, 12, 16, 17, 18, 19, 20, 21)
    joints[np.asarray(upper), 2] -= 0.30
    joints[4] = (0.16, 0.12, 0.38)
    joints[5] = (0.16, -0.12, 0.38)
    return joints


def zup_to_ay(joints: np.ndarray) -> np.ndarray:
    return np.einsum("ij,nj->ni", AXIS_CONVERT_AY_TO_ZUP.T, joints)


def identity_rotations() -> np.ndarray:
    return np.repeat(np.eye(3, dtype=np.float64)[None], 22, axis=0)


def rotation_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.asarray(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )


class SMPLDirectAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text())

    def make_adapter(self, **kwargs) -> SMPLDirectAdapter:
        return SMPLDirectAdapter(copy.deepcopy(self.config), **kwargs)

    def adapt(self, joints: np.ndarray, **kwargs):
        adapter = self.make_adapter(**kwargs)
        return adapter, adapter.adapt(zup_to_ay(joints), identity_rotations())

    def assert_proper_frames(self, frame) -> None:
        self.assertEqual(tuple(frame.raw_targets), TARGET_NAMES)
        for target in frame.raw_targets.values():
            np.testing.assert_allclose(
                target.rotation_zup.T @ target.rotation_zup,
                np.eye(3),
                atol=1e-6,
            )
            self.assertAlmostEqual(np.linalg.det(target.rotation_zup), 1.0, places=6)

    def test_target_table_and_positions_are_exact_joint_centers(self) -> None:
        _, frame = self.adapt(standing_joints_zup())
        self.assertEqual(set(TARGET_JOINT_INDICES), set(TARGET_NAMES))
        for name, joint_index in TARGET_JOINT_INDICES.items():
            np.testing.assert_allclose(
                frame.raw_targets[name].position_zup,
                frame.joints_zup[joint_index],
                atol=1e-10,
            )
        # Unit base scales mean the transmitted targets are also joint centers.
        for name in TARGET_NAMES:
            np.testing.assert_allclose(
                frame.scaled_targets[name].position_zup,
                frame.raw_targets[name].position_zup,
                atol=1e-10,
            )

    def test_standing_frames_follow_documented_axes(self) -> None:
        _, frame = self.adapt(standing_joints_zup())
        self.assert_proper_frames(frame)
        np.testing.assert_allclose(
            frame.raw_targets["SMPL_Pelvis"].rotation_zup, np.eye(3), atol=1e-6
        )
        for name in (
            "SMPL_LeftHip",
            "SMPL_RightHip",
            "SMPL_LeftKnee",
            "SMPL_RightKnee",
            "SMPL_LeftShoulder",
            "SMPL_RightShoulder",
            "SMPL_LeftElbow",
            "SMPL_RightElbow",
        ):
            np.testing.assert_allclose(
                frame.raw_targets[name].rotation_zup[:, 2],
                (0.0, 0.0, -1.0),
                atol=1e-6,
            )

    def test_t_pose_is_left_right_symmetric(self) -> None:
        _, frame = self.adapt(t_pose_joints_zup())
        left = frame.raw_targets["SMPL_LeftShoulder"].rotation_zup[:, 2]
        right = frame.raw_targets["SMPL_RightShoulder"].rotation_zup[:, 2]
        np.testing.assert_allclose(left, (0.0, 1.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(right, -left, atol=1e-6)
        for left_name, right_name in (
            ("SMPL_LeftHip", "SMPL_RightHip"),
            ("SMPL_LeftKnee", "SMPL_RightKnee"),
            ("SMPL_LeftAnkle", "SMPL_RightAnkle"),
            ("SMPL_LeftShoulder", "SMPL_RightShoulder"),
            ("SMPL_LeftElbow", "SMPL_RightElbow"),
            ("SMPL_LeftWrist", "SMPL_RightWrist"),
        ):
            left_position = frame.raw_targets[left_name].position_zup
            right_position = frame.raw_targets[right_name].position_zup
            self.assertAlmostEqual(left_position[0], right_position[0], places=7)
            self.assertAlmostEqual(left_position[1], -right_position[1], places=7)
            self.assertAlmostEqual(left_position[2], right_position[2], places=7)

    def test_single_arm_raise_does_not_rotate_other_arm(self) -> None:
        standing = standing_joints_zup()
        raised = standing.copy()
        raised[18] = (0.0, 0.60, 1.40)
        raised[20] = (0.0, 0.90, 1.40)
        _, first = self.adapt(standing)
        _, second = self.adapt(raised)
        self.assertGreater(
            np.linalg.norm(
                first.raw_targets["SMPL_LeftShoulder"].rotation_zup
                - second.raw_targets["SMPL_LeftShoulder"].rotation_zup
            ),
            0.5,
        )
        np.testing.assert_allclose(
            first.raw_targets["SMPL_RightShoulder"].rotation_zup,
            second.raw_targets["SMPL_RightShoulder"].rotation_zup,
            atol=1e-7,
        )

    def test_foot_lock_squat_keeps_feet_and_lowers_pelvis(self) -> None:
        adapter = self.make_adapter(vertical_mode="foot_lock")
        standing = adapter.adapt(zup_to_ay(standing_joints_zup()), identity_rotations())
        squat = adapter.adapt(zup_to_ay(squat_joints_zup()), identity_rotations())
        np.testing.assert_allclose(squat.joints_zup[[10, 11], 2], 0.0, atol=1e-7)
        self.assertLess(
            squat.raw_targets["SMPL_Pelvis"].position_zup[2],
            standing.raw_targets["SMPL_Pelvis"].position_zup[2] - 0.2,
        )

    def test_foot_lock_removes_whole_body_vertical_drift(self) -> None:
        adapter = self.make_adapter(vertical_mode="foot_lock")
        first = adapter.adapt(zup_to_ay(standing_joints_zup()), identity_rotations())
        drifted = standing_joints_zup()
        drifted[:, 2] += 0.75
        second = adapter.adapt(zup_to_ay(drifted), identity_rotations())
        np.testing.assert_allclose(second.joints_zup, first.joints_zup, atol=1e-7)

    def test_contact_preserves_jump_and_relocks_on_landing(self) -> None:
        adapter = self.make_adapter(vertical_mode="contact")
        base = standing_joints_zup()
        first = adapter.adapt(zup_to_ay(base), identity_rotations(), timestamp_ns=1_000_000_000)
        jumped = base.copy()
        jumped[:, 2] += 0.30
        airborne = adapter.adapt(
            zup_to_ay(jumped), identity_rotations(), timestamp_ns=1_033_333_333
        )
        self.assertEqual(airborne.contact_mask, (False, False))
        self.assertGreater(
            airborne.raw_targets["SMPL_Pelvis"].position_zup[2]
            - first.raw_targets["SMPL_Pelvis"].position_zup[2],
            0.25,
        )
        adapter.adapt(zup_to_ay(base), identity_rotations(), timestamp_ns=1_066_666_666)
        landed = adapter.adapt(zup_to_ay(base), identity_rotations(), timestamp_ns=1_099_999_999)
        self.assertEqual(landed.contact_mask, (True, True))
        np.testing.assert_allclose(landed.joints_zup[[10, 11], 2], 0.0, atol=1e-7)

    def test_degenerate_limb_reuses_previous_frame(self) -> None:
        adapter = self.make_adapter()
        first = adapter.adapt(zup_to_ay(standing_joints_zup()), identity_rotations())
        degenerate = standing_joints_zup()
        degenerate[18] = degenerate[16]
        second = adapter.adapt(zup_to_ay(degenerate), identity_rotations())
        np.testing.assert_allclose(
            second.raw_targets["SMPL_LeftShoulder"].rotation_zup,
            first.raw_targets["SMPL_LeftShoulder"].rotation_zup,
            atol=1e-7,
        )

    def test_gem2_packet_layout_and_quaternion_sign_continuity(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(0.3)
        bridge = GMRUDPBridge("127.0.0.1", receiver.getsockname()[1])
        try:
            _, frame = self.adapt(standing_joints_zup())
            first_targets = dict(frame.scaled_targets)
            second_targets = dict(frame.scaled_targets)
            first_rotation = rotation_z(179.0)
            second_rotation = rotation_z(181.0)
            for name in TARGET_NAMES:
                position = frame.scaled_targets[name].position_zup
                first_targets[name] = TargetPose(position, first_rotation)
                second_targets[name] = TargetPose(position, second_rotation)

            first_packet = bridge.send_smpl_targets(first_targets, source_stamp_ns=123)
            second_packet = bridge.send_smpl_targets(second_targets, source_stamp_ns=456)
            self.assertEqual(receiver.recvfrom(2048)[0], first_packet)
            self.assertEqual(receiver.recvfrom(2048)[0], second_packet)
            self.assertEqual(len(first_packet), PACKET_BYTES)
            self.assertEqual(
                HEADER.unpack_from(first_packet),
                (GEM2_MAGIC, GEM2_VERSION, len(TARGET_NAMES), 0, 123),
            )
            first_values = PAYLOAD.unpack_from(first_packet, HEADER.size)
            second_values = PAYLOAD.unpack_from(second_packet, HEADER.size)
            np.testing.assert_allclose(
                first_values[:3],
                frame.scaled_targets["SMPL_Pelvis"].position_zup,
                atol=1e-6,
            )
            first_quaternion = np.asarray(first_values[3:7])
            second_quaternion = np.asarray(second_values[3:7])
            self.assertGreater(float(np.dot(first_quaternion, second_quaternion)), 0.99)
        finally:
            bridge.close()
            receiver.close()

    def test_gem1_send_segments_remains_compatible(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(0.3)
        bridge = GMRUDPBridge("127.0.0.1", receiver.getsockname()[1])
        try:
            targets = {
                name: TargetPose(np.zeros(3), np.eye(3))
                for name in (
                    "Pelvis",
                    "Chest",
                    "Left_UpperLeg",
                    "Right_UpperLeg",
                    "Left_LowerLeg",
                    "Right_LowerLeg",
                    "Left_Foot",
                    "Right_Foot",
                    "Left_UpperArm",
                    "Right_UpperArm",
                    "Left_Forearm",
                    "Right_Forearm",
                    "Left_Hand",
                    "Right_Hand",
                )
            }
            packet = bridge.send_segments(targets)
            self.assertEqual(receiver.recvfrom(2048)[0], packet)
            magic, version, count, _, _ = HEADER.unpack_from(packet)
            self.assertEqual((magic, version, count), (MAGIC, VERSION, 14))
        finally:
            bridge.close()
            receiver.close()


if __name__ == "__main__":
    unittest.main()
