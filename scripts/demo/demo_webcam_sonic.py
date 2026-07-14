# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""GEM-SMPL webcam demo with optional asynchronous SONIC ZMQ streaming.

Example::

    CUDA_VISIBLE_DEVICES=0 python scripts/demo/demo_webcam_sonic.py \
        --camera_id 2 --no_imgfeat --sonic_zmq
"""

# ruff: noqa: E402, I001
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SONIC_REPO_PATH = PROJECT_ROOT.parent / "GR00T-WholeBodyControl"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from demo_webcam import WebcamGEMSMPLDemo


class WebcamGEMSMPLSonicDemo(WebcamGEMSMPLDemo):
    """Thin streaming adapter around the unchanged webcam inference demo."""

    def __init__(self, args):
        self._sonic_enabled = bool(args.sonic_zmq)
        self._sonic_publisher = None
        self._last_sonic_params = None

        if self._sonic_enabled:
            # Lazy import keeps this script usable without pyzmq when streaming
            # is disabled and leaves demo_webcam.py completely independent.
            from gem.utils.sonic.zmq_publisher import SonicPublisher

            self._sonic_publisher = SonicPublisher(
                host=args.sonic_host,
                port=args.sonic_port,
                topic=args.sonic_topic,
                sonic_repo_path=args.sonic_repo_path,
                enable_yaw_calibration=args.enable_yaw_calibration,
            )
            self._sonic_publisher.connect()

        try:
            super().__init__(args)
        except BaseException:
            if self._sonic_publisher is not None:
                self._sonic_publisher.close()
            raise

    def process_frame(self, frame_bgr):
        if self._sonic_publisher is not None:
            self._sonic_publisher.check_health()
        result = super().process_frame(frame_bgr)

        if self._sonic_publisher is not None and result is not None and result.get("ready"):
            body_params_incam = result["body_params_incam"]

            # The base async pipeline returns the same pending result until the
            # next backend inference finishes. Its nested params mapping keeps
            # object identity across those shallow copies, so publish it once.
            if body_params_incam is not self._last_sonic_params:
                self._last_sonic_params = body_params_incam
                self._sonic_publisher.publish_smpl(
                    body_params_incam,
                    result["body_params_global"],
                    timestamp_ns=time.monotonic_ns(),
                )

        return result

    def run(self):
        try:
            return super().run()
        finally:
            if self._sonic_publisher is not None:
                self._sonic_publisher.close()


def parse_args():
    """Mirror the base demo CLI and append SONIC-only options."""
    parser = argparse.ArgumentParser(description="GEM-SMPL Webcam Demo with SONIC ZMQ streaming")
    parser.add_argument("--camera_id", type=int, default=0, help="Webcam device ID")
    parser.add_argument("--video", type=str, default=None, help="Video file (overrides camera)")
    parser.add_argument(
        "--context_frames",
        type=int,
        default=120,
        help="Sliding window length (model max=120, shorter=faster)",
    )
    parser.add_argument(
        "--yolo_period",
        type=int,
        default=5,
        help="Run YOLOX every N frames (ByteTrack interpolates between)",
    )
    parser.add_argument(
        "--vitpose_period",
        type=int,
        default=1,
        help="Run ViTPose every N frames (reuse keypoints between)",
    )
    parser.add_argument(
        "--no_imgfeat",
        action="store_true",
        help="Skip HMR2 features; use the no-imgfeat ONNX denoiser variant",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable background rendering in a separate process",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="viser",
        choices=["viser", "opencv"],
        help="Render backend: 'viser' (web 3D world) or 'opencv' (in-camera overlay)",
    )
    parser.add_argument(
        "--render_port",
        type=int,
        default=8012,
        help="Port for Viser web server (only used with --render_mode viser)",
    )
    parser.add_argument(
        "--async_pipeline",
        action="store_true",
        default=True,
        help="Overlap ViTPose, HMR2, and denoiser across frames (default: True)",
    )
    parser.add_argument(
        "--no_async_pipeline",
        action="store_true",
        help="Disable async pipeline (force synchronous mode)",
    )

    parser.add_argument(
        "--sonic_zmq",
        action="store_true",
        help="Publish ready GEM-SMPL poses to SONIC using ZMQ Protocol v3",
    )
    parser.add_argument(
        "--sonic_host",
        type=str,
        default="localhost",
        help="Host/interface on which the SONIC ZMQ publisher binds",
    )
    parser.add_argument(
        "--sonic_port",
        type=int,
        default=5556,
        help="Port on which the SONIC ZMQ publisher binds",
    )
    parser.add_argument(
        "--sonic_topic",
        type=str,
        default="pose",
        help="SONIC ZMQ topic prefix",
    )
    parser.add_argument(
        "--sonic_repo_path",
        type=str,
        default=str(DEFAULT_SONIC_REPO_PATH),
        help="Path to the GR00T-WholeBodyControl repository",
    )
    parser.add_argument(
        "--enable_yaw_calibration",
        action="store_true",
        help="Remove the first valid SONIC root heading from subsequent frames",
    )
    return parser.parse_args()


if __name__ == "__main__":
    demo = WebcamGEMSMPLSonicDemo(parse_args())
    demo.run()
