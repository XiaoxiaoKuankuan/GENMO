# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Reusable resident video model stack and restartable source sessions."""

from __future__ import annotations

import inspect
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from gem.runtime.motion_streamer import SMPLFrame


def _load_endecoder(device: torch.device) -> torch.nn.Module:
    from gem.network.endecoder import EnDecoder

    model = EnDecoder(
        stats_name="MM_V1_AMASS_LOCAL_BEDLAM_CAM",
        encode_type="gvhmr",
        feat_dim=151,
        clip_std=True,
    )
    model.build_obs_indices_dict()
    return model.eval().to(device)


def _load_video_components(
    no_imgfeat: bool,
    device: torch.device,
    shared_endecoder: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Load every video model once; the tracker returned by YOLOX is discarded."""
    from scripts.demo import demo_webcam

    with ThreadPoolExecutor(max_workers=4) as pool:
        yolo_future = pool.submit(demo_webcam.load_yolox)
        pose_future = pool.submit(demo_webcam.load_vitpose)
        denoiser_future = pool.submit(demo_webcam.load_denoiser, no_imgfeat=no_imgfeat)
        endecoder_future = (
            None if shared_endecoder is not None else pool.submit(_load_endecoder, device)
        )
        hmr_future = None if no_imgfeat else pool.submit(demo_webcam.load_hmr2)
        yolox, _initial_tracker, yolo_backend = yolo_future.result()
        vitpose_runner, vitpose_backend = pose_future.result()
        denoiser_runner, denoiser_backend = denoiser_future.result()
        if denoiser_runner is None:
            raise RuntimeError(
                "No realtime ONNX denoiser was found. Export the matching "
                "GEM-SMPL regression model before starting video mode."
            )
        endecoder = shared_endecoder if endecoder_future is None else endecoder_future.result()
        if hmr_future is None:
            hmr2_runner, hmr2_backend = None, "none"
        else:
            hmr2_runner, hmr2_backend = hmr_future.result()
    return {
        "yolox": yolox,
        "yolox_backend": yolo_backend,
        "vitpose_runner": vitpose_runner,
        "vitpose_backend": vitpose_backend,
        "denoiser_runner": denoiser_runner,
        "denoiser_backend": denoiser_backend,
        "endecoder": endecoder,
        "hmr2_runner": hmr2_runner,
        "hmr2_backend": hmr2_backend,
    }


class ResidentVideoModelStack:
    """Own YOLOX, ViTPose, HMR2, ONNX denoiser, and EnDecoder exactly once."""

    def __init__(
        self,
        *,
        no_imgfeat: bool = True,
        context_frames: int = 120,
        device: str | torch.device = "cuda:0",
        warmup_enabled: bool = True,
        warmup_width: int = 1280,
        warmup_height: int = 720,
        shared_endecoder: torch.nn.Module | None = None,
        loader: Callable[..., dict[str, Any]] = _load_video_components,
    ) -> None:
        if context_frames <= 0:
            raise ValueError("context_frames must be > 0")
        if warmup_width <= 0 or warmup_height <= 0:
            raise ValueError("warmup dimensions must be > 0")
        self.no_imgfeat = bool(no_imgfeat)
        self.context_frames = int(context_frames)
        self.device = torch.device(device)
        self.warmup_enabled = bool(warmup_enabled)
        self.warmup_width = int(warmup_width)
        self.warmup_height = int(warmup_height)
        self.shared_endecoder = shared_endecoder
        self._loader = loader
        self._lock = threading.RLock()
        self.initialized = False
        self.load_count = 0
        self.warmup_count = 0
        self.load_seconds = 0.0
        self.yolox: Any | None = None
        self.yolox_backend = "unknown"
        self.vitpose_runner: Any | None = None
        self.vitpose_backend = "unknown"
        self.denoiser_runner: Any | None = None
        self.denoiser_backend = "unknown"
        self.endecoder: Any | None = None
        self.hmr2_runner: Any | None = None
        self.hmr2_backend = "none"

    def initialize(self) -> None:
        """Load and optionally warm the model stack once."""
        with self._lock:
            if self.initialized:
                return
            started = time.perf_counter()
            loader_params = inspect.signature(self._loader).parameters
            if len(loader_params) >= 3:
                components = self._loader(
                    self.no_imgfeat,
                    self.device,
                    self.shared_endecoder,
                )
            else:
                # Backward-compatible injection point for lightweight tests.
                components = self._loader(self.no_imgfeat, self.device)
            required = {
                "yolox",
                "vitpose_runner",
                "vitpose_backend",
                "denoiser_runner",
                "denoiser_backend",
                "endecoder",
                "hmr2_runner",
                "hmr2_backend",
            }
            missing = sorted(required - set(components))
            if missing:
                raise RuntimeError(f"Video model loader omitted fields: {missing}")
            for key, value in components.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            self.load_count += 1
            self.initialized = True
            self.load_seconds = time.perf_counter() - started
            print(f"[Video] Resident model stack loaded once in {self.load_seconds:.2f}s")
            if self.warmup_enabled:
                self.warmup()

    def new_tracker(self) -> Any:
        """Create fresh ByteTrack state for each newly opened source."""
        from gem.utils.yolox_detector import ByteTracker

        return ByteTracker(max_lost=30)

    def warmup(self) -> None:
        """Warm resident video inference without opening a camera or video file."""
        if not self.initialized:
            raise RuntimeError("initialize the video stack before warmup")
        from gem.utils.cam_utils import estimate_K
        from gem.utils.geo_transform import compute_cam_angvel
        from scripts.demo import demo_webcam

        height, width = self.warmup_height, self.warmup_width
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        dummy_bbox = torch.tensor([width / 2, height / 2, min(width, height) * 0.5])
        rotation = torch.eye(3).unsqueeze(0).repeat(self.context_frames, 1, 1)
        K = estimate_K(width, height).float()
        batch = {
            "obs": torch.zeros(1, self.context_frames, 17, 3),
            "bbx_xys": torch.zeros(1, self.context_frames, 3),
            "K_fullimg": K.reshape(1, 1, 3, 3).repeat(1, self.context_frames, 1, 1),
            "f_imgseq": torch.zeros(1, self.context_frames, 1024),
            "f_cam_angvel": compute_cam_angvel(rotation).unsqueeze(0),
        }

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(
                    demo_webcam.run_vitpose_single_frame,
                    self.vitpose_runner,
                    self.vitpose_backend,
                    dummy_frame,
                    dummy_bbox,
                ),
                pool.submit(
                    demo_webcam.run_denoiser,
                    self.denoiser_runner,
                    self.denoiser_backend,
                    batch,
                ),
            ]
            if self.hmr2_runner is not None:
                futures.append(
                    pool.submit(
                        demo_webcam.run_hmr2_single_frame,
                        self.hmr2_runner,
                        self.hmr2_backend,
                        dummy_frame,
                        dummy_bbox,
                    )
                )
            for future in futures:
                future.result()
        self.warmup_count += 1
        print("[Video] Resident model stack warmup complete")

    def status(self) -> dict[str, Any]:
        """Return load identity without exposing model objects."""
        return {
            "initialized": self.initialized,
            "load_count": self.load_count,
            "warmup_count": self.warmup_count,
            "load_seconds": self.load_seconds,
            "no_imgfeat": self.no_imgfeat,
            "context_frames": self.context_frames,
            "denoiser_backend": self.denoiser_backend,
            "vitpose_backend": self.vitpose_backend,
            "hmr2_backend": self.hmr2_backend,
        }

    def close(self) -> None:
        """Release model references only during complete service shutdown."""
        with self._lock:
            self.initialized = False
            self.yolox = None
            self.vitpose_runner = None
            self.denoiser_runner = None
            self.endecoder = None
            self.hmr2_runner = None


@dataclass(slots=True)
class VideoSourceSession:
    """One opened source description and its capture-backed demo."""

    kind: str
    value: int | str
    demo: Any
    started_monotonic: float


def _default_demo_factory(args: Any, **kwargs: Any) -> Any:
    from scripts.demo.demo_webcam import WebcamGEMSMPLDemo

    return WebcamGEMSMPLDemo(args, **kwargs)


class ResidentVideoSession:
    """Restart sources while retaining a single resident video model stack."""

    def __init__(
        self,
        model_stack: ResidentVideoModelStack,
        *,
        frame_sink: Callable[[SMPLFrame], None],
        generation_pause: threading.Event | None = None,
        yolo_period: int = 5,
        vitpose_period: int = 1,
        shape_mode: str = "zero",
        shape_warmup: int = 30,
        demo_factory: Callable[..., Any] = _default_demo_factory,
    ) -> None:
        if yolo_period <= 0 or vitpose_period <= 0:
            raise ValueError("video model periods must be > 0")
        if shape_mode != "zero":
            raise ValueError("unified video service requires shape_mode=zero")
        self.model_stack = model_stack
        self.frame_sink = frame_sink
        self.generation_pause = generation_pause or threading.Event()
        self.yolo_period = int(yolo_period)
        self.vitpose_period = int(vitpose_period)
        self.shape_mode = shape_mode
        self.shape_warmup = int(shape_warmup)
        self._demo_factory = demo_factory
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._reset_requested = threading.Event()
        self._inference_idle = threading.Event()
        self._inference_idle.set()
        self._paused_ack = threading.Event()
        self._thread: threading.Thread | None = None
        self.source: VideoSourceSession | None = None
        self.frames_read = 0
        self.frames_ready = 0
        self.reset_count = 0
        self.last_error: str | None = None

    def _build_args(
        self,
        *,
        camera_id: int | None,
        video_path: str | None,
        rtsp_url: str | None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            camera_id=0 if camera_id is None else int(camera_id),
            video=video_path,
            rtsp_url=rtsp_url,
            context_frames=self.model_stack.context_frames,
            yolo_period=self.yolo_period,
            vitpose_period=self.vitpose_period,
            no_imgfeat=self.model_stack.no_imgfeat,
            render=False,
            render_mode="viser",
            render_port=8012,
            display=False,
            async_pipeline=False,
            no_async_pipeline=True,
            gmr_host=None,
            gmr_port=7006,
            gmr_protocol="smplx1",
            gmr_scale=1.0,
            shape_mode=self.shape_mode,
            shape_warmup=self.shape_warmup,
            smplx_yaw_deg=0.0,
        )

    def start_source(
        self,
        *,
        camera_id: int | None = None,
        video_path: str | Path | None = None,
        rtsp_url: str | None = None,
    ) -> None:
        """Open exactly one source with fresh tracker and rollout state."""
        supplied = sum(value is not None for value in (camera_id, video_path, rtsp_url))
        if supplied != 1:
            raise ValueError(
                "video_start requires exactly one of camera_id, video_path, or rtsp_url"
            )
        self.stop_source()
        self.model_stack.initialize()
        resolved_video = None if video_path is None else str(Path(video_path))
        args = self._build_args(
            camera_id=camera_id,
            video_path=resolved_video,
            rtsp_url=rtsp_url,
        )
        demo = self._demo_factory(
            args,
            frame_sink=self.frame_sink,
            model_stack=self.model_stack,
            create_gmr_bridge=False,
        )
        if camera_id is not None:
            kind, value = "camera", int(camera_id)
        elif video_path is not None:
            kind, value = "video", resolved_video
        else:
            kind, value = "rtsp", str(rtsp_url)
        with self._lock:
            self.frames_read = 0
            self.frames_ready = 0
            self.last_error = None
            self._stop_event.clear()
            self._reset_requested.clear()
            self._paused_ack.clear()
            self.generation_pause.clear()
            self.source = VideoSourceSession(
                kind=kind,
                value=value,
                demo=demo,
                started_monotonic=time.monotonic(),
            )
            self._thread = threading.Thread(
                target=self.process_loop,
                name="genmo-video-session",
                daemon=True,
            )
            self._thread.start()
        print(f"[Video] Source started: {kind}={value}")

    def process_loop(self) -> None:
        """Capture and infer until source stop/end; no UDP work occurs here."""
        with self._lock:
            source = self.source
        if source is None:
            return
        demo = source.demo
        next_video_frame = time.monotonic()
        try:
            while not self._stop_event.is_set():
                if self.generation_pause.is_set():
                    self._inference_idle.set()
                    self._paused_ack.set()
                    self._stop_event.wait(0.005)
                    continue
                self._paused_ack.clear()
                if self._reset_requested.is_set():
                    demo.reset_state()
                    self.reset_count += 1
                    self._reset_requested.clear()
                ok, frame = demo.cap.read()
                if not ok:
                    break
                if demo._video_frame_period is not None:
                    remaining = next_video_frame - time.monotonic()
                    if remaining > 0:
                        self._stop_event.wait(remaining)
                    next_video_frame = time.monotonic() + demo._video_frame_period
                if self._stop_event.is_set():
                    break
                self._inference_idle.clear()
                try:
                    result = demo.process_frame(frame)
                finally:
                    self._inference_idle.set()
                self.frames_read += 1
                if result is not None and result.get("ready", False):
                    self.frames_ready += 1
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[Video ERROR] {self.last_error}")
        finally:
            self._inference_idle.set()

    def pause_inference(self, timeout: float = 30.0) -> None:
        """Pause GPU video work and wait for an in-flight synchronous frame."""
        self.generation_pause.set()
        with self._lock:
            thread = self._thread
        if thread is None or not thread.is_alive():
            self._paused_ack.set()
        if not self._paused_ack.wait(timeout) or not self._inference_idle.wait(timeout):
            raise TimeoutError("Timed out waiting for resident video inference to pause")

    def resume_inference(self) -> None:
        """Allow source processing to continue after GEM generation."""
        self.generation_pause.clear()
        self._paused_ack.clear()

    def reset_state(self) -> None:
        """Reset source tracking/rollout before video output resumes."""
        with self._lock:
            source = self.source
        if source is None:
            return
        if not self.generation_pause.is_set():
            raise RuntimeError("Pause video inference before resetting its state")
        if not self._inference_idle.is_set():
            raise RuntimeError("Video inference is still active during reset")
        source.demo.reset_state()
        self.reset_count += 1

    def request_reset(self) -> None:
        """Request a non-blocking reset before the next captured inference frame."""
        with self._lock:
            if self.source is not None:
                self._reset_requested.set()

    def stop_source(self) -> None:
        """Release only the current capture; keep the model stack resident."""
        with self._lock:
            thread = self._thread
            source = self.source
            self._stop_event.set()
            self._reset_requested.clear()
            self._paused_ack.clear()
            self.generation_pause.clear()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10.0)
            if thread.is_alive():
                raise RuntimeError("Resident video thread did not stop in time")
        if source is not None:
            source.demo.close()
        with self._lock:
            self._thread = None
            self.source = None
            self._inference_idle.set()
        if source is not None:
            print("[Video] Source stopped; resident models remain loaded")

    def status(self) -> dict[str, Any]:
        """Return session and model residency diagnostics."""
        with self._lock:
            source = self.source
            thread = self._thread
            return {
                "active": source is not None and thread is not None and thread.is_alive(),
                "paused": self.generation_pause.is_set(),
                "source_kind": None if source is None else source.kind,
                "source_value": None if source is None else source.value,
                "frames_read": self.frames_read,
                "frames_ready": self.frames_ready,
                "reset_count": self.reset_count,
                "last_error": self.last_error,
                "model_stack": self.model_stack.status(),
            }

    def close(self) -> None:
        """Stop the source and release the resident model stack."""
        self.stop_source()
        self.model_stack.close()
