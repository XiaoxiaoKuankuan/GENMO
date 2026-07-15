"""Synthetic anatomical checks for the GEM joint-to-segment adapter."""

from __future__ import annotations

import copy
import json
import math
import socket
import unittest
from pathlib import Path

import numpy as np

from gem.gmr_segment_adapter import SEGMENT_NAMES, BetaStabilizer, GMRSegmentAdapter
from gem.gmr_udp_bridge import HEADER, PACKET_BYTES, PAYLOAD, GMRUDPBridge

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/gmr/e1_segment_adapter.json"
AXIS_CONVERT = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))


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


def zup_to_ay(joints: np.ndarray) -> np.ndarray:
    return np.einsum("ij,nj->ni", AXIS_CONVERT.T, joints)


def identity_rotations() -> np.ndarray:
    return np.repeat(np.eye(3)[None], 22, axis=0)


def rotate_z(joints: np.ndarray, degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    rotation = np.asarray(
        (
            (math.cos(angle), -math.sin(angle), 0.0),
            (math.sin(angle), math.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    return np.einsum("ij,nj->ni", rotation, joints)


class GMRSegmentAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text())

    def adapt(self, joints_zup: np.ndarray, **kwargs):
        adapter = GMRSegmentAdapter(copy.deepcopy(self.config), **kwargs)
        return adapter, adapter.adapt(zup_to_ay(joints_zup), identity_rotations())

    def assert_proper_frames(self, frame) -> None:
        self.assertEqual(tuple(frame.scaled_segments), SEGMENT_NAMES)
        for pose in frame.scaled_segments.values():
            np.testing.assert_allclose(
                pose.rotation_zup.T @ pose.rotation_zup, np.eye(3), atol=1e-6
            )
            self.assertAlmostEqual(np.linalg.det(pose.rotation_zup), 1.0, places=6)

    def test_natural_standing_arms_point_down_and_frames_are_proper(self) -> None:
        _, frame = self.adapt(standing_joints_zup())
        self.assert_proper_frames(frame)
        for name in ("Left_UpperArm", "Right_UpperArm", "Left_Forearm", "Right_Forearm"):
            primary = frame.raw_segments[name].rotation_zup[:, 2]
            np.testing.assert_allclose(primary, [0.0, 0.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(frame.raw_segments["Pelvis"].rotation_zup, np.eye(3), atol=1e-6)

    def test_t_pose_upper_arm_axes_are_horizontal_outward_and_symmetric(self) -> None:
        _, frame = self.adapt(t_pose_joints_zup())
        left = frame.raw_segments["Left_UpperArm"].rotation_zup[:, 2]
        right = frame.raw_segments["Right_UpperArm"].rotation_zup[:, 2]
        np.testing.assert_allclose(left, [0.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(right, [0.0, -1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(left, -right, atol=1e-6)
        self.assert_proper_frames(frame)

    def test_single_arm_raise_uses_bone_geometry_not_elbow_joint_rotation(self) -> None:
        standing = standing_joints_zup()
        raised = standing.copy()
        raised[18] = (0.0, 0.60, 1.40)
        raised[20] = (0.0, 0.90, 1.40)
        _, standing_frame = self.adapt(standing)
        _, raised_frame = self.adapt(raised)
        self.assertGreater(
            np.linalg.norm(
                standing_frame.raw_segments["Left_UpperArm"].rotation_zup
                - raised_frame.raw_segments["Left_UpperArm"].rotation_zup
            ),
            0.5,
        )
        np.testing.assert_allclose(
            standing_frame.raw_segments["Right_UpperArm"].rotation_zup,
            raised_frame.raw_segments["Right_UpperArm"].rotation_zup,
            atol=1e-6,
        )

        rotations = identity_rotations()
        rotations[18] = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        adapter = GMRSegmentAdapter(copy.deepcopy(self.config))
        changed_elbow = adapter.adapt(zup_to_ay(raised), rotations)
        np.testing.assert_allclose(
            changed_elbow.raw_segments["Left_UpperArm"].rotation_zup,
            raised_frame.raw_segments["Left_UpperArm"].rotation_zup,
            atol=1e-6,
        )

    def test_origins_are_configured_midpoints(self) -> None:
        _, frame = self.adapt(standing_joints_zup())
        pairs = {
            "Left_UpperArm": (16, 18),
            "Right_UpperArm": (17, 19),
            "Left_Forearm": (18, 20),
            "Right_Forearm": (19, 21),
            "Left_Foot": (7, 10),
            "Right_Foot": (8, 11),
        }
        for name, (proximal, distal) in pairs.items():
            expected = 0.5 * (frame.joints_zup[proximal] + frame.joints_zup[distal])
            np.testing.assert_allclose(frame.raw_segments[name].position_zup, expected, atol=1e-7)
        np.testing.assert_allclose(
            frame.raw_segments["Left_Hand"].position_zup,
            frame.joints_zup[20],
            atol=1e-7,
        )

    def test_hierarchical_scale_changes_edge_length_not_direction(self) -> None:
        config = copy.deepcopy(self.config)
        config["edge_scales"]["Chest->Left_UpperArm"] = 1.8
        adapter = GMRSegmentAdapter(config)
        frame = adapter.adapt(zup_to_ay(standing_joints_zup()), identity_rotations())
        raw_edge = (
            frame.raw_segments["Left_UpperArm"].position_zup
            - frame.raw_segments["Chest"].position_zup
        )
        scaled_edge = (
            frame.scaled_segments["Left_UpperArm"].position_zup
            - frame.scaled_segments["Chest"].position_zup
        )
        self.assertAlmostEqual(np.linalg.norm(scaled_edge) / np.linalg.norm(raw_edge), 1.8)
        np.testing.assert_allclose(
            scaled_edge / np.linalg.norm(scaled_edge),
            raw_edge / np.linalg.norm(raw_edge),
            atol=1e-7,
        )

    def test_degenerate_bone_reuses_previous_frame_without_axis_flip(self) -> None:
        adapter = GMRSegmentAdapter(copy.deepcopy(self.config))
        first = adapter.adapt(zup_to_ay(standing_joints_zup()), identity_rotations())
        degenerate = standing_joints_zup()
        degenerate[18] = degenerate[16]
        second = adapter.adapt(zup_to_ay(degenerate), identity_rotations())
        np.testing.assert_allclose(
            second.raw_segments["Left_UpperArm"].rotation_zup,
            first.raw_segments["Left_UpperArm"].rotation_zup,
            atol=1e-7,
        )
        fresh = GMRSegmentAdapter(copy.deepcopy(self.config)).adapt(
            zup_to_ay(degenerate), identity_rotations()
        )
        np.testing.assert_allclose(
            fresh.raw_segments["Left_UpperArm"].rotation_zup,
            fresh.raw_segments["Pelvis"].rotation_zup,
            atol=1e-7,
        )

    def test_mean_betas_freeze_after_warmup(self) -> None:
        stabilizer = BetaStabilizer("mean", warmup=3)
        outputs = [stabilizer.update(np.full(10, value)) for value in (1.0, 2.0, 3.0)]
        frozen = stabilizer.update(np.full(10, 100.0))
        np.testing.assert_allclose(outputs[-1], np.full(10, 2.0))
        np.testing.assert_allclose(frozen, outputs[-1])
        self.assertTrue(stabilizer.frozen)
        self.assertAlmostEqual(1.0 + 0.1 * frozen[0], 1.2)

    def test_initial_heading_aligns_front_90_and_180_to_e1_forward(self) -> None:
        reference = None
        for degrees in (0.0, 90.0, 180.0):
            _, frame = self.adapt(rotate_z(t_pose_joints_zup(), degrees))
            pelvis_rotation = frame.raw_segments["Pelvis"].rotation_zup
            np.testing.assert_allclose(pelvis_rotation[:, 0], [1.0, 0.0, 0.0], atol=1e-6)
            origins = np.stack([frame.raw_segments[name].position_zup for name in SEGMENT_NAMES])
            if reference is None:
                reference = origins
            else:
                np.testing.assert_allclose(origins, reference, atol=1e-6)

        adapter = GMRSegmentAdapter(copy.deepcopy(self.config))
        adapter.adapt(zup_to_ay(t_pose_joints_zup()), identity_rotations())
        turned = adapter.adapt(
            zup_to_ay(rotate_z(t_pose_joints_zup(), 90.0)), identity_rotations()
        )
        np.testing.assert_allclose(
            turned.raw_segments["Pelvis"].rotation_zup[:, 0],
            [0.0, 1.0, 0.0],
            atol=1e-6,
        )

    def test_contact_ground_handles_drift_jump_and_landing(self) -> None:
        adapter = GMRSegmentAdapter(copy.deepcopy(self.config), ground_mode="contact")
        base = standing_joints_zup()
        first = adapter.adapt(zup_to_ay(base), identity_rotations(), timestamp_ns=1_000_000_000)
        initial_height = first.raw_segments["Pelvis"].position_zup[2]
        frame = first
        # Simulate a full minute of slow estimator drift while the feet remain in contact.
        for index in range(1, 1801):
            drifted = base.copy()
            drifted[:, 2] += index * 0.0001
            frame = adapter.adapt(
                zup_to_ay(drifted),
                identity_rotations(),
                timestamp_ns=1_000_000_000 + index * 33_333_333,
            )
        self.assertLess(
            abs(frame.raw_segments["Pelvis"].position_zup[2] - initial_height),
            0.02,
        )

        drift_baseline = base.copy()
        drift_baseline[:, 2] += 0.18
        jumped = drift_baseline.copy()
        jumped[:, 2] += 0.30
        jump_frame = adapter.adapt(
            zup_to_ay(jumped), identity_rotations(), timestamp_ns=61_100_000_000
        )
        self.assertGreater(
            jump_frame.raw_segments["Pelvis"].position_zup[2] - initial_height,
            0.25,
        )
        adapter.adapt(
            zup_to_ay(drift_baseline),
            identity_rotations(),
            timestamp_ns=61_133_333_333,
        )
        landed = adapter.adapt(
            zup_to_ay(drift_baseline),
            identity_rotations(),
            timestamp_ns=61_166_666_666,
        )
        self.assertLess(
            abs(landed.raw_segments["Pelvis"].position_zup[2] - initial_height),
            0.02,
        )

    def test_segment_udp_packet_remains_exactly_412_bytes(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(0.3)
        bridge = GMRUDPBridge("127.0.0.1", receiver.getsockname()[1])
        try:
            _, frame = self.adapt(standing_joints_zup())
            packet = bridge.send_segments(frame.scaled_segments, source_stamp_ns=123)
            received, _ = receiver.recvfrom(2048)
            self.assertEqual(packet, received)
            self.assertEqual(len(packet), PACKET_BYTES)
            header = HEADER.unpack_from(packet)
            self.assertEqual(header[-1], 123)
            values = PAYLOAD.unpack_from(packet, HEADER.size)
            left_upper_arm = values[8 * 7 : 9 * 7]
            np.testing.assert_allclose(
                left_upper_arm[:3],
                frame.scaled_segments["Left_UpperArm"].position_zup,
                atol=1e-6,
            )
            self.assertFalse(np.allclose(left_upper_arm[:3], frame.joints_zup[16], atol=1e-6))
        finally:
            bridge.close()
            receiver.close()


if __name__ == "__main__":
    unittest.main()
