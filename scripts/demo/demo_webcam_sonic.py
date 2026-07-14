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

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SONIC_REPO_PATH = PROJECT_ROOT.parent / "GR00T-WholeBodyControl"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from demo_webcam import WebcamGEMSMPLDemo


def _shape_text(shape):
    if not shape:
        return "()"
    suffix = "," if len(shape) == 1 else ""
    return f"({','.join(map(str, shape))}{suffix})"


def _nonfinite_summary(value, name):
    """Return a lightweight raw-output diagnostic without altering the value."""
    try:
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if not (tensor.is_floating_point() or tensor.is_complex()):
                return None
            counts = torch.stack(
                (
                    torch.isnan(tensor).sum(),
                    torch.isposinf(tensor).sum(),
                    torch.isneginf(tensor).sum(),
                )
            ).to(device="cpu")
            nan_count, posinf_count, neginf_count = map(int, counts.tolist())
            shape = tuple(tensor.shape)
        else:
            array = np.asarray(value)
            if not np.issubdtype(array.dtype, np.inexact):
                return None
            nan_count = int(np.isnan(array).sum())
            posinf_count = int(np.isposinf(array).sum())
            neginf_count = int(np.isneginf(array).sum())
            shape = array.shape
    except Exception as exc:
        return f"{name}: finite precheck failed ({exc})"

    if nan_count + posinf_count + neginf_count == 0:
        return None
    return (
        f"{name}: shape={_shape_text(shape)}, nan={nan_count}, "
        f"posinf={posinf_count}, neginf={neginf_count}"
    )


def _raw_smpl_nonfinite_details(body_params_incam, body_params_global):
    details = []
    for source_name, params in (
        ("in-camera", body_params_incam),
        ("global", body_params_global),
    ):
        for field_name in ("body_pose", "global_orient"):
            value = params.get(field_name)
            if value is None:
                continue
            detail = _nonfinite_summary(value, f"{source_name} {field_name}")
            if detail is not None:
                details.append(detail)
    return details


class WebcamGEMSMPLSonicDemo(WebcamGEMSMPLDemo):
    """Thin streaming adapter around the unchanged webcam inference demo."""

    def __init__(self, args):
        self._sonic_enabled = bool(args.sonic_zmq)
        self._sonic_publisher = None
        self._last_sonic_params = None
        self._last_raw_nonfinite_warning_time = float("-inf")

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
                sonic_bad_frame_dir=getattr(args, "sonic_bad_frame_dir", None),
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
                body_params_global = result["body_params_global"]
                invalid_details = _raw_smpl_nonfinite_details(
                    body_params_incam,
                    body_params_global,
                )
                now = time.monotonic()
                if invalid_details and now - self._last_raw_nonfinite_warning_time >= 1.0:
                    self._last_raw_nonfinite_warning_time = now
                    print(
                        "\n[SONIC WARNING] raw GEM result contains non-finite values\n"
                        + "\n".join(invalid_details)
                        + "\npublisher will apply fallback/drop policy"
                    )
                self._sonic_publisher.publish_smpl(
                    body_params_incam,
                    body_params_global,
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
    parser.add_argument(
        "--sonic_bad_frame_dir",
        type=str,
        default=None,
        help="Optionally save at most three invalid GEM-SMPL inputs for diagnosis",
    )
    return parser.parse_args()


if __name__ == "__main__":
    demo = WebcamGEMSMPLSonicDemo(parse_args())
    demo.run()
