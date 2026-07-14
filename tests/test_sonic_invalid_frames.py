# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Regression tests for recoverable invalid GEM-SMPL source frames."""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import torch
import zmq

from gem.utils.sonic.smpl_converter import InvalidSMPLFrameError, SonicSMPLConverter
from gem.utils.sonic.zmq_publisher import SonicPublisher


def _free_tcp_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


class SonicInvalidFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        default_repo = Path(__file__).resolve().parents[2] / "GR00T-WholeBodyControl"
        cls.sonic_repo = Path(os.environ.get("SONIC_REPO_PATH", default_repo))
        if not (cls.sonic_repo / "gear_sonic/trl/utils/torch_transform.py").is_file():
            raise unittest.SkipTest(f"GR00T-WholeBodyControl not found: {cls.sonic_repo}")

    def setUp(self) -> None:
        self.converter = SonicSMPLConverter(self.sonic_repo)
        self.pose = torch.zeros((1, 63), dtype=torch.float32)
        self.root = torch.zeros((1, 3), dtype=torch.float32)

    def test_zero_pose_converts(self) -> None:
        result = self.converter.convert(
            {"body_pose": self.pose, "global_orient": self.root},
            {"body_pose": self.pose, "global_orient": self.root},
        )
        self.assertEqual(result["smpl_pose"].shape, (1, 21, 3))
        self.assertEqual(result["smpl_joints_local"].shape, (1, 24, 3))
        self.assertEqual(result["body_quat_w"].shape, (1, 4))
        self.assertTrue(all(torch.isfinite(value).all() for value in result.values()))

    def test_incam_body_pose_has_priority_over_invalid_global(self) -> None:
        invalid_global_pose = self.pose.clone()
        invalid_global_pose[0, 0] = torch.nan
        result = self.converter.convert(
            {"body_pose": invalid_global_pose, "global_orient": self.root},
            {"body_pose": self.pose, "global_orient": self.root},
        )
        self.assertTrue(torch.equal(result["smpl_pose"], self.pose.reshape(1, 21, 3)))

    def test_invalid_incam_body_pose_falls_back_to_global(self) -> None:
        invalid_incam_pose = self.pose.clone()
        invalid_incam_pose[0, 0] = torch.nan
        global_pose = self.pose.clone()
        global_pose[0, 1] = 0.25
        result = self.converter.convert(
            {"body_pose": global_pose, "global_orient": self.root},
            {"body_pose": invalid_incam_pose, "global_orient": self.root},
        )
        self.assertTrue(torch.equal(result["smpl_pose"], global_pose.reshape(1, 21, 3)))

    def test_invalid_global_root_falls_back_to_incam(self) -> None:
        invalid_global_root = self.root.clone()
        invalid_global_root[0, 0] = torch.nan
        result = self.converter.convert(
            {"body_pose": self.pose, "global_orient": invalid_global_root},
            {"body_pose": self.pose, "global_orient": self.root},
        )
        self.assertTrue(torch.isfinite(result["smpl_joints_local"]).all())

    def test_both_invalid_roots_raise_recoverable_error(self) -> None:
        invalid_global_root = self.root.clone()
        invalid_global_root[0, 0] = torch.nan
        invalid_incam_root = self.root.clone()
        invalid_incam_root[0, 1] = torch.inf
        with self.assertRaises(InvalidSMPLFrameError) as caught:
            self.converter.convert(
                {"body_pose": self.pose, "global_orient": invalid_global_root},
                {"body_pose": self.pose, "global_orient": invalid_incam_root},
            )
        message = str(caught.exception)
        self.assertIn("global global_orient", message)
        self.assertIn("in-camera global_orient", message)
        self.assertIn("nan=1", message)
        self.assertIn("posinf=1", message)

    def test_missing_pose_candidates_raise_recoverable_error(self) -> None:
        with self.assertRaises(InvalidSMPLFrameError) as caught:
            self.converter.convert(
                {"global_orient": self.root},
                {"global_orient": self.root},
            )
        self.assertIn("body_pose is missing", str(caught.exception))

    def test_nonfinite_fk_output_reports_exact_stage(self) -> None:
        self.converter._compute_human_joints = lambda **_kwargs: torch.full(
            (1, 24, 3),
            torch.nan,
        )
        with self.assertRaises(InvalidSMPLFrameError) as caught:
            self.converter.convert(
                {"global_orient": self.root},
                {"body_pose": self.pose},
            )
        self.assertIn("compute_human_joints output", str(caught.exception))

    def test_one_bad_frame_does_not_stop_worker_and_normal_output_resumes(self) -> None:
        port = _free_tcp_port()
        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"pose")
        subscriber.setsockopt(zmq.RCVTIMEO, 2000)
        subscriber.connect(f"tcp://127.0.0.1:{port}")
        publisher = SonicPublisher(
            host="127.0.0.1",
            port=port,
            queue_size=4,
            sonic_repo_path=self.sonic_repo,
        )
        try:
            publisher.connect()
            time.sleep(0.2)
            base_timestamp = time.monotonic_ns()
            invalid_root = np.full((1, 3), np.nan, dtype=np.float32)
            pose = np.zeros((1, 63), dtype=np.float32)
            invalid_params = {"body_pose": pose, "global_orient": invalid_root}
            self.assertTrue(
                publisher.publish_smpl(
                    invalid_params,
                    invalid_params,
                    base_timestamp,
                )
            )
            self.assertTrue(_wait_until(lambda: publisher._invalid_source_frames == 1))
            self.assertTrue(publisher.check_health())

            num_frames = 6
            normal_pose = np.zeros((num_frames, 63), dtype=np.float32)
            normal_root = np.zeros((num_frames, 3), dtype=np.float32)
            normal_params = {"body_pose": normal_pose, "global_orient": normal_root}
            timestamps = base_timestamp + np.arange(1, num_frames + 1) * 20_000_000
            self.assertTrue(
                publisher.publish_smpl(
                    normal_params,
                    normal_params,
                    timestamps,
                )
            )

            message = subscriber.recv()
            header = json.loads(message[4:1284].rstrip(b"\x00"))
            fields = {field["name"]: field["shape"] for field in header["fields"]}
            self.assertEqual(fields["smpl_pose"], [5, 21, 3])
            self.assertTrue(_wait_until(lambda: publisher._consecutive_invalid_frames == 0))
            self.assertTrue(publisher.check_health())
            self.assertGreaterEqual(publisher._next_frame_index, 5)
        finally:
            publisher.close()
            subscriber.close(0)
            context.term()

    def test_fallback_batch_length_uses_converted_pose_for_scalar_timestamp(self) -> None:
        port = _free_tcp_port()
        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)
        subscriber.setsockopt(zmq.SUBSCRIBE, b"pose")
        subscriber.setsockopt(zmq.RCVTIMEO, 2000)
        subscriber.connect(f"tcp://127.0.0.1:{port}")
        publisher = SonicPublisher(
            host="127.0.0.1",
            port=port,
            queue_size=4,
            sonic_repo_path=self.sonic_repo,
        )
        try:
            publisher.connect()
            time.sleep(0.2)
            invalid_incam = {
                "body_pose": np.full((1, 63), np.nan, dtype=np.float32),
                "global_orient": np.zeros((1, 3), dtype=np.float32),
            }
            global_params = {
                "body_pose": np.zeros((6, 63), dtype=np.float32),
                "global_orient": np.zeros((6, 3), dtype=np.float32),
            }
            self.assertTrue(
                publisher.publish_smpl(
                    invalid_incam,
                    global_params,
                    time.monotonic_ns(),
                )
            )

            message = subscriber.recv()
            header = json.loads(message[4:1284].rstrip(b"\x00"))
            fields = {field["name"]: field["shape"] for field in header["fields"]}
            self.assertEqual(fields["smpl_pose"], [5, 21, 3])
            self.assertEqual(publisher._invalid_source_frames, 0)
            self.assertTrue(publisher.check_health())
        finally:
            publisher.close()
            subscriber.close(0)
            context.term()

    def test_fallback_batch_length_accepts_matching_timestamp_array(self) -> None:
        publisher = SonicPublisher(
            host="127.0.0.1",
            port=_free_tcp_port(),
            queue_size=4,
            sonic_repo_path=self.sonic_repo,
        )
        invalid_incam = {
            "body_pose": np.full((1, 63), np.nan, dtype=np.float32),
            "global_orient": np.zeros((1, 3), dtype=np.float32),
        }
        global_params = {
            "body_pose": np.zeros((6, 63), dtype=np.float32),
            "global_orient": np.zeros((6, 3), dtype=np.float32),
        }
        try:
            publisher.connect()
            base_timestamp = time.monotonic_ns()
            timestamps = base_timestamp + np.arange(6, dtype=np.int64) * 20_000_000
            self.assertTrue(
                publisher.publish_smpl(
                    invalid_incam,
                    global_params,
                    timestamps,
                )
            )
            self.assertTrue(_wait_until(lambda: publisher._next_frame_index == 6))
            self.assertEqual(publisher._invalid_source_frames, 0)
            self.assertTrue(publisher.check_health())
        finally:
            publisher.close()

    def test_valid_bad_valid_preserves_resampler_window_and_frame_index(self) -> None:
        publisher = SonicPublisher(
            host="127.0.0.1",
            port=_free_tcp_port(),
            queue_size=4,
            sonic_repo_path=self.sonic_repo,
        )
        pose = np.zeros((1, 63), dtype=np.float32)
        root = np.zeros((1, 3), dtype=np.float32)
        valid_params = {"body_pose": pose, "global_orient": root}
        invalid_params = {
            "body_pose": pose,
            "global_orient": np.full((1, 3), np.nan, dtype=np.float32),
        }
        try:
            publisher.connect()
            base_timestamp = time.monotonic_ns()
            self.assertTrue(publisher.publish_smpl(valid_params, valid_params, base_timestamp))
            self.assertTrue(_wait_until(lambda: publisher._next_frame_index == 1))
            first_window = [frame["frame_index"] for frame in publisher._frame_window]

            self.assertTrue(
                publisher.publish_smpl(
                    invalid_params,
                    invalid_params,
                    base_timestamp + 20_000_000,
                )
            )
            self.assertTrue(_wait_until(lambda: publisher._invalid_source_frames == 1))
            self.assertEqual(publisher._next_frame_index, 1)
            self.assertEqual(
                [frame["frame_index"] for frame in publisher._frame_window],
                first_window,
            )
            self.assertTrue(publisher.check_health())

            self.assertTrue(
                publisher.publish_smpl(
                    valid_params,
                    valid_params,
                    base_timestamp + 40_000_000,
                )
            )
            self.assertTrue(_wait_until(lambda: publisher._next_frame_index == 3))
            self.assertEqual(
                [frame["frame_index"] for frame in publisher._frame_window],
                [0, 1, 2],
            )
            self.assertEqual(publisher._consecutive_invalid_frames, 0)
            self.assertTrue(publisher.check_health())
        finally:
            publisher.close()

    def test_thirtieth_consecutive_bad_frame_is_fatal(self) -> None:
        publisher = SonicPublisher(
            host="127.0.0.1",
            port=_free_tcp_port(),
            queue_size=2,
            sonic_repo_path=self.sonic_repo,
        )
        try:
            publisher.connect()
            base_timestamp = time.monotonic_ns()
            invalid_root = np.full((1, 3), np.nan, dtype=np.float32)
            pose = np.zeros((1, 63), dtype=np.float32)
            invalid_params = {"body_pose": pose, "global_orient": invalid_root}
            for frame_index in range(30):
                previous_count = publisher._invalid_source_frames
                self.assertTrue(
                    publisher.publish_smpl(
                        invalid_params,
                        invalid_params,
                        base_timestamp + frame_index * 20_000_000,
                    )
                )
                self.assertTrue(
                    _wait_until(
                        lambda previous_count=previous_count: (
                            publisher._invalid_source_frames > previous_count
                            or publisher._worker_error is not None
                        )
                    )
                )
                if frame_index < 29:
                    self.assertTrue(publisher.check_health())

            self.assertTrue(_wait_until(lambda: publisher._worker_error is not None))
            with self.assertRaises(RuntimeError) as caught:
                publisher.check_health()
            self.assertIsInstance(caught.exception.__cause__, RuntimeError)
            self.assertIn(
                "Too many consecutive invalid GEM-SMPL frames",
                str(caught.exception.__cause__),
            )
            self.assertEqual(publisher._invalid_source_frames, 30)
        finally:
            publisher.close()

    def test_bad_frame_snapshots_are_limited_to_three(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            publisher = SonicPublisher(
                sonic_repo_path=self.sonic_repo,
                sonic_bad_frame_dir=directory,
            )
            error = InvalidSMPLFrameError("test invalid frame")
            params = {
                "body_pose": np.zeros((1, 63), dtype=np.float32),
                "global_orient": np.full((1, 3), np.nan, dtype=np.float32),
            }
            for frame_index in range(5):
                publisher._save_bad_frame(
                    params,
                    params,
                    np.array([frame_index], dtype=np.int64),
                    error,
                )

            snapshots = sorted(Path(directory).glob("bad_smpl_frame_*.pt"))
            self.assertEqual(len(snapshots), 3)
            saved = torch.load(snapshots[0], weights_only=False)
            self.assertEqual(saved["body_params_incam"]["body_pose"].device.type, "cpu")
            self.assertEqual(saved["error"], str(error))

    def test_periodic_log_contains_invalid_and_queue_drop_statistics(self) -> None:
        publisher = SonicPublisher(sonic_repo_path=self.sonic_repo)
        publisher._frames_sent = 99
        publisher._started_at = time.perf_counter() - 2.0
        publisher._invalid_source_frames = 4
        publisher._consecutive_invalid_frames = 0
        publisher._frames_dropped = 3
        pose_data = {
            "smpl_pose": np.zeros((5, 21, 3), dtype=np.float32),
            "smpl_joints": np.zeros((5, 24, 3), dtype=np.float32),
            "body_quat": np.tile(
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (5, 1),
            ),
        }
        output = StringIO()
        with redirect_stdout(output):
            publisher._report_sent(pose_data)
        log = output.getvalue()
        self.assertIn("invalid_source_frames=4", log)
        self.assertIn("consecutive_invalid_frames=0", log)
        self.assertIn("dropped_queue_frames=3", log)

    def test_bad_frame_snapshot_failure_is_nonfatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            blocker = Path(directory) / "not_a_directory"
            blocker.write_text("block directory creation", encoding="utf-8")
            publisher = SonicPublisher(
                sonic_repo_path=self.sonic_repo,
                sonic_bad_frame_dir=blocker / "bad_frames",
            )
            output = StringIO()
            with redirect_stdout(output):
                publisher._save_bad_frame(
                    {"body_pose": self.pose},
                    {"global_orient": self.root},
                    np.array([1], dtype=np.int64),
                    InvalidSMPLFrameError("test invalid frame"),
                )
            self.assertIn("failed to save invalid GEM frame", output.getvalue())


if __name__ == "__main__":
    unittest.main()
