"""Synthetic tests for original-GMR SMPL-X targets and the SMP1 packet."""

from __future__ import annotations

import socket
import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - project runtime normally provides torch.
    torch = None

from gem.gmr_udp_bridge import (
    HEADER,
    PACKET_BYTES,
    PAYLOAD,
    SMPLX_MAGIC,
    SMPLX_TARGET_NAMES,
    SMPLX_VERSION,
    GMRUDPBridge,
)
from gem.smplx_gmr_reference import (
    AXIS_CONVERT_AY_TO_ZUP,
    TARGET_JOINT_INDICES,
    TARGET_NAMES,
    BetaStabilizer,
    SMPLXGMRReference,
)


def standing_frame() -> tuple[np.ndarray, np.ndarray]:
    joints = np.zeros((22, 3), dtype=np.float64)
    joints[0] = (0.0, 1.0, 0.0)
    joints[9] = (0.0, 1.35, 0.0)
    joints[1] = (0.0, 0.95, -0.12)
    joints[2] = (0.0, 0.95, 0.12)
    joints[4] = (0.0, 0.55, -0.12)
    joints[5] = (0.0, 0.55, 0.12)
    joints[10] = (0.20, 0.0, -0.12)
    joints[11] = (0.20, 0.0, 0.12)
    joints[16] = (0.0, 1.40, -0.25)
    joints[17] = (0.0, 1.40, 0.25)
    joints[18] = (0.0, 1.20, -0.50)
    joints[19] = (0.0, 1.20, 0.50)
    joints[20] = (0.0, 1.00, -0.70)
    joints[21] = (0.0, 1.00, 0.70)
    rotations = np.repeat(np.eye(3, dtype=np.float64)[None], 22, axis=0)
    return joints, rotations


class SMPLXGMRReferenceTest(unittest.TestCase):
    def test_beta_zero_numpy_preserves_shape_and_dtype(self) -> None:
        stabilizer = BetaStabilizer("zero", warmup=30)
        betas = np.arange(20, dtype=np.float32).reshape(2, 10)
        result = stabilizer.update(betas)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, betas.shape)
        self.assertEqual(result.dtype, betas.dtype)
        self.assertEqual(np.count_nonzero(result), 0)
        self.assertTrue(stabilizer.frozen)
        self.assertEqual(stabilizer.count, 0)

    @unittest.skipUnless(torch is not None, "Torch is not installed")
    def test_beta_zero_torch_cpu_preserves_shape_dtype_and_device(self) -> None:
        stabilizer = BetaStabilizer("zero")
        betas = torch.randn(3, 10, dtype=torch.float64)
        result = stabilizer.update(betas)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, betas.shape)
        self.assertEqual(result.dtype, betas.dtype)
        self.assertEqual(result.device, betas.device)
        self.assertEqual(torch.count_nonzero(result).item(), 0)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is not available")
    def test_beta_zero_torch_cuda_stays_on_cuda(self) -> None:
        stabilizer = BetaStabilizer("zero")
        betas = torch.randn(1, 10, device="cuda", dtype=torch.float16)
        result = stabilizer.update(betas)
        self.assertEqual(result.device.type, "cuda")
        self.assertEqual(result.dtype, betas.dtype)
        self.assertEqual(result.shape, betas.shape)
        self.assertEqual(torch.count_nonzero(result).item(), 0)

    def test_beta_zero_is_constant_across_updates(self) -> None:
        stabilizer = BetaStabilizer("zero", warmup=30)
        for value in (-10.0, 1.0, 99.0):
            result = stabilizer.update(np.full((1, 10), value, dtype=np.float32))
            np.testing.assert_array_equal(result, np.zeros((1, 10), dtype=np.float32))
        self.assertTrue(stabilizer.frozen)
        self.assertEqual(stabilizer.count, 0)

    def test_beta_zero_still_rejects_nonfinite_input(self) -> None:
        stabilizer = BetaStabilizer("zero")
        for bad_value in (np.nan, np.inf, -np.inf):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                    stabilizer.update(np.full((1, 10), bad_value, dtype=np.float32))

    def test_beta_mean_freezes_after_warmup(self) -> None:
        stabilizer = BetaStabilizer("mean", warmup=3)
        stabilizer.update(np.zeros(10))
        stabilizer.update(np.ones(10))
        frozen = stabilizer.update(np.full(10, 2.0))
        np.testing.assert_allclose(frozen, np.ones(10))
        np.testing.assert_allclose(stabilizer.update(np.full(10, 99.0)), frozen)
        self.assertTrue(stabilizer.frozen)

    def test_names_indices_and_joint_centers(self) -> None:
        joints, rotations = standing_frame()
        frame = SMPLXGMRReference().adapt(joints, rotations)
        self.assertEqual(TARGET_NAMES, SMPLX_TARGET_NAMES)
        self.assertEqual(tuple(frame.scaled_targets), TARGET_NAMES)
        self.assertEqual(set(TARGET_JOINT_INDICES), set(TARGET_NAMES))
        for name, index in TARGET_JOINT_INDICES.items():
            expected = AXIS_CONVERT_AY_TO_ZUP @ joints[index]
            expected -= np.asarray((0.0, 0.0, 0.0))
            np.testing.assert_allclose(
                frame.scaled_targets[name].position_zup, expected, atol=1e-12
            )

    def test_global_rotation_uses_world_only_transform(self) -> None:
        joints, rotations = standing_frame()
        angle = np.deg2rad(31.0)
        rotation = np.asarray(
            (
                (np.cos(angle), 0.0, np.sin(angle)),
                (0.0, 1.0, 0.0),
                (-np.sin(angle), 0.0, np.cos(angle)),
            )
        )
        rotations[16] = rotation
        frame = SMPLXGMRReference().adapt(joints, rotations)
        actual = frame.scaled_targets["left_shoulder"].rotation_zup
        np.testing.assert_allclose(actual, AXIS_CONVERT_AY_TO_ZUP @ rotation, atol=1e-12)
        self.assertFalse(
            np.allclose(
                actual,
                AXIS_CONVERT_AY_TO_ZUP @ rotation @ AXIS_CONVERT_AY_TO_ZUP.T,
            )
        )

    def test_initial_origin_uses_pelvis_xy_and_smplx_feet(self) -> None:
        joints, rotations = standing_frame()
        joints[:, 0] += 4.0
        joints[:, 2] -= 3.0
        frame = SMPLXGMRReference(global_scale=1.5).adapt(joints, rotations)
        pelvis = frame.scaled_targets["pelvis"].position_zup
        left_foot = frame.scaled_targets["left_foot"].position_zup
        right_foot = frame.scaled_targets["right_foot"].position_zup
        np.testing.assert_allclose(pelvis[:2], (0.0, 0.0), atol=1e-12)
        self.assertAlmostEqual(min(left_foot[2], right_foot[2]), 0.0)
        self.assertAlmostEqual(pelvis[2], 1.5)

    def test_smp1_packet_layout(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(0.5)
        bridge = GMRUDPBridge("127.0.0.1", receiver.getsockname()[1])
        try:
            joints, rotations = standing_frame()
            frame = SMPLXGMRReference().adapt(joints, rotations, frame_id=7, timestamp_ns=123456789)
            returned = bridge.send_smplx_targets(
                frame.scaled_targets, source_stamp_ns=frame.timestamp_ns
            )
            packet, _ = receiver.recvfrom(2048)
            self.assertEqual(returned, packet)
            self.assertEqual(len(packet), PACKET_BYTES)
            self.assertEqual(
                HEADER.unpack_from(packet),
                (SMPLX_MAGIC, SMPLX_VERSION, 14, 0, 123456789),
            )
            values = PAYLOAD.unpack_from(packet, HEADER.size)
            self.assertEqual(len(values), 14 * 7)
            np.testing.assert_allclose(values[:3], (0.0, 0.0, 1.0), atol=1e-6)
        finally:
            bridge.close()
            receiver.close()

    def test_invalid_rotation_is_rejected(self) -> None:
        joints, rotations = standing_frame()
        rotations[3, 0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "proper SO\\(3\\)"):
            SMPLXGMRReference().adapt(joints, rotations)


if __name__ == "__main__":
    unittest.main()
