"""Lightweight GEM1 bridge tests; no model weights or SMPL-X model required."""

from __future__ import annotations

import math
import socket
import struct
import unittest

import numpy as np
import torch

from gem.gmr_udp_bridge import (
    BONE_NAMES,
    HEADER,
    MAGIC,
    PACKET_BYTES,
    PAYLOAD,
    VERSION,
    GMRUDPBridge,
)


class GMRUDPBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.bind(("127.0.0.1", 0))
        self.receiver.settimeout(0.3)
        self.bridge = GMRUDPBridge("127.0.0.1", self.receiver.getsockname()[1])

    def tearDown(self) -> None:
        self.bridge.close()
        self.receiver.close()

    @staticmethod
    def _identity_frame() -> tuple[torch.Tensor, torch.Tensor]:
        joints = torch.zeros((22, 3), dtype=torch.float32)
        rotations = torch.eye(3, dtype=torch.float32).repeat(22, 1, 1)
        return joints, rotations

    def _receive_values(self) -> tuple[tuple[object, ...], tuple[float, ...]]:
        packet, _ = self.receiver.recvfrom(2048)
        self.assertEqual(len(packet), PACKET_BYTES)
        return HEADER.unpack_from(packet), PAYLOAD.unpack_from(packet, HEADER.size)

    def test_packet_layout_axis_conversion_and_quaternion(self) -> None:
        joints, rotations = self._identity_frame()
        # Chest is joint 9. AY [x,y,z] must become Z-up [x,-z,y].
        joints[9] = torch.tensor([1.0, 2.0, 3.0])
        self.bridge.send_fk(joints, rotations, source_stamp_ns=123456)

        header, values = self._receive_values()
        self.assertEqual(PACKET_BYTES, 412)
        self.assertEqual(header, (MAGIC, VERSION, len(BONE_NAMES), 0, 123456))
        self.assertEqual(len(values), 14 * 7)

        chest = values[7:14]
        np.testing.assert_allclose(chest[:3], [1.0, -3.0, 2.0], atol=1e-6)
        np.testing.assert_allclose(chest[3:], [1.0, 0.0, 0.0, 0.0], atol=1e-6)

    def test_identity_input_produces_identity_pelvis_quaternion(self) -> None:
        joints, rotations = self._identity_frame()
        self.bridge.send_fk(joints, rotations)

        _, values = self._receive_values()
        np.testing.assert_allclose(values[3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-7)

    def test_initial_yaw_is_cancelled(self) -> None:
        joints, _ = self._identity_frame()
        axis_convert = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
            dtype=np.float32,
        )
        yaw = math.radians(63.0)
        rz = np.asarray(
            (
                (math.cos(yaw), -math.sin(yaw), 0.0),
                (math.sin(yaw), math.cos(yaw), 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        )
        # C @ R_ay @ C.T == Rz(yaw), so normalization outputs identity.
        rotation_ay = torch.from_numpy(axis_convert.T @ rz @ axis_convert)
        rotations = rotation_ay.repeat(22, 1, 1)
        self.bridge.send_fk(joints, rotations)

        _, values = self._receive_values()
        pelvis_quat = values[3:7]
        np.testing.assert_allclose(pelvis_quat, [1.0, 0.0, 0.0, 0.0], atol=1e-5)

    def test_ground_uses_toes_but_transmits_ankle_height(self) -> None:
        self.bridge.close()
        self.bridge = GMRUDPBridge("127.0.0.1", self.receiver.getsockname()[1], scale=1.7)
        joints, rotations = self._identity_frame()
        # AY is Y-up: joints 7/8 are transmitted ankles; 10/11 are toes/ground.
        joints[7:9, 1] = 0.10
        joints[10:12, 1] = 0.0
        self.bridge.send_fk(joints, rotations)

        _, values = self._receive_values()
        left_foot = values[6 * 7 : 7 * 7]
        right_foot = values[7 * 7 : 8 * 7]
        self.assertAlmostEqual(left_foot[2], 0.10 * 1.7, places=6)
        self.assertAlmostEqual(right_foot[2], 0.10 * 1.7, places=6)

    def test_identity_frame_preserves_left_right_symmetry(self) -> None:
        joints, rotations = self._identity_frame()
        left_right_pairs = ((1, 2), (4, 5), (7, 8), (16, 17), (18, 19), (20, 21))
        for pair_index, (left, right) in enumerate(left_right_pairs, start=1):
            joints[left] = torch.tensor([0.1 * pair_index, 0.2, 0.3])
            joints[right] = torch.tensor([-0.1 * pair_index, 0.2, 0.3])
        self.bridge.send_fk(joints, rotations)

        _, values = self._receive_values()
        payload_pairs = ((2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13))
        for left, right in payload_pairs:
            left_data = np.asarray(values[left * 7 : (left + 1) * 7])
            right_data = np.asarray(values[right * 7 : (right + 1) * 7])
            self.assertAlmostEqual(left_data[0], -right_data[0], places=6)
            np.testing.assert_allclose(left_data[1:3], right_data[1:3], atol=1e-6)
            np.testing.assert_allclose(left_data[3:], right_data[3:], atol=1e-6)

    def test_improper_rotation_is_rejected_without_udp_packet(self) -> None:
        joints, rotations = self._identity_frame()
        rotations[3, 0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "not a proper rotation"):
            self.bridge.send_fk(joints, rotations)
        with self.assertRaises(socket.timeout):
            self.receiver.recvfrom(2048)

    def test_nan_frame_is_rejected_without_udp_packet(self) -> None:
        joints, rotations = self._identity_frame()
        joints[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            self.bridge.send_fk(joints, rotations)
        with self.assertRaises(socket.timeout):
            self.receiver.recvfrom(2048)

    def test_protocol_structs_are_explicit_little_endian(self) -> None:
        self.assertEqual(HEADER.format, "<4sHHIQ")
        self.assertEqual(PAYLOAD.size, 14 * 7 * struct.calcsize("<f"))


if __name__ == "__main__":
    unittest.main()
