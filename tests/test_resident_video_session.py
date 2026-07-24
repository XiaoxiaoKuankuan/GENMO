"""CPU tests for the reusable video model stack and source lifecycle."""

from __future__ import annotations

import time

import numpy as np
import torch

from gem.runtime.motion_streamer import SMPLFrame
from gem.runtime.resident_video_session import (
    ResidentVideoModelStack,
    ResidentVideoSession,
)


def dummy_loader(_no_imgfeat, _device, _shared=None):
    return {
        "yolox": object(),
        "yolox_backend": "dummy",
        "vitpose_runner": object(),
        "vitpose_backend": "dummy",
        "denoiser_runner": object(),
        "denoiser_backend": "dummy",
        "endecoder": object() if _shared is None else _shared,
        "hmr2_runner": None,
        "hmr2_backend": "none",
    }


class DummyCapture:
    def __init__(self) -> None:
        self.released = False

    def read(self):
        time.sleep(0.002)
        return (not self.released), np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class DummyDemo:
    def __init__(self, args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.cap = DummyCapture()
        self._video_frame_period = None
        self.processed = 0
        self.resets = 0
        self.closed = False

    def process_frame(self, _frame):
        self.processed += 1
        frame = SMPLFrame(
            torch.zeros(63),
            torch.zeros(3),
            torch.zeros(3),
            torch.zeros(10),
        )
        self.kwargs["frame_sink"](frame)
        return {"ready": True}

    def reset_state(self):
        self.resets += 1

    def close(self):
        self.closed = True
        self.cap.release()


def test_model_stack_loads_exactly_once() -> None:
    stack = ResidentVideoModelStack(
        device="cpu",
        warmup_enabled=False,
        loader=dummy_loader,
    )
    stack.initialize()
    first_endecoder = stack.endecoder
    stack.initialize()
    assert stack.load_count == 1
    assert stack.endecoder is first_endecoder


def test_video_start_stop_reuses_models_and_disables_direct_gmr() -> None:
    stack = ResidentVideoModelStack(
        device="cpu",
        warmup_enabled=False,
        loader=dummy_loader,
    )
    demos = []
    frames = []

    def factory(args, **kwargs):
        demo = DummyDemo(args, **kwargs)
        demos.append(demo)
        return demo

    session = ResidentVideoSession(
        stack,
        frame_sink=frames.append,
        demo_factory=factory,
    )
    session.start_source(camera_id=0)
    time.sleep(0.02)
    session.stop_source()
    session.start_source(camera_id=1)
    time.sleep(0.02)
    session.stop_source()
    assert stack.load_count == 1
    assert len(demos) == 2
    assert all(demo.kwargs["create_gmr_bridge"] is False for demo in demos)
    assert all(demo.kwargs["model_stack"] is stack for demo in demos)
    assert frames
    assert all(torch.count_nonzero(frame.betas) == 0 for frame in frames)


def test_pause_generation_and_nonblocking_reset_request() -> None:
    stack = ResidentVideoModelStack(
        device="cpu",
        warmup_enabled=False,
        loader=dummy_loader,
    )
    demos = []

    def factory(args, **kwargs):
        demo = DummyDemo(args, **kwargs)
        demos.append(demo)
        return demo

    session = ResidentVideoSession(
        stack,
        frame_sink=lambda _frame: None,
        demo_factory=factory,
    )
    session.start_source(camera_id=0)
    time.sleep(0.01)
    session.pause_inference()
    count = demos[0].processed
    time.sleep(0.01)
    assert demos[0].processed == count
    session.resume_inference()
    session.request_reset()
    time.sleep(0.02)
    assert demos[0].resets == 1
    assert session.status()["reset_count"] == 1
    session.close()


def test_every_source_gets_fresh_session_state() -> None:
    stack = ResidentVideoModelStack(
        device="cpu",
        warmup_enabled=False,
        loader=dummy_loader,
    )
    demos = []

    def factory(args, **kwargs):
        demo = DummyDemo(args, **kwargs)
        demos.append(demo)
        return demo

    session = ResidentVideoSession(
        stack,
        frame_sink=lambda _frame: None,
        demo_factory=factory,
    )
    session.start_source(camera_id=2)
    first = session.source
    session.stop_source()
    session.start_source(video_path="server.mp4")
    second = session.source
    assert first is not second
    assert first.kind == "camera" and second.kind == "video"
    assert demos[0].closed
    assert second.demo.args.no_async_pipeline is True
    assert second.demo.args.gmr_host is None
    session.close()


def test_status_exposes_residency_without_udp_objects() -> None:
    stack = ResidentVideoModelStack(
        device="cpu",
        warmup_enabled=False,
        loader=dummy_loader,
    )
    session = ResidentVideoSession(
        stack,
        frame_sink=lambda _frame: None,
        demo_factory=lambda args, **kwargs: DummyDemo(args, **kwargs),
    )
    session.start_source(camera_id=0)
    status = session.status()
    assert status["model_stack"]["load_count"] == 1
    assert not hasattr(session, "gmr_bridge")
    session.close()
