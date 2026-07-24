"""CPU protocol tests for the unified multimodal server and client."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from scripts.demo import demo_multimodal_server as server
from scripts.demo import multimodal_motion_client as client


class DummyEngine:
    def __init__(self, audio_root: Path, output: Path) -> None:
        self.allowed_audio_roots = (audio_root.resolve(),)
        self.output = output
        self.initialized = 0
        self.closed = 0
        self.requests: list[dict] = []
        self.clear_calls: list[str] = []

    def initialize(self):
        self.initialized += 1

    def generate(self, request):
        self.requests.append(dict(request))
        if request.get("prompt") == "fail":
            return {"ok": False, "error_type": "RuntimeError", "error": "injected"}
        return {
            "ok": True,
            "output_dir": str(self.output),
            "request_id": request.get("request_id"),
        }

    def clear_cache(self, target):
        self.clear_calls.append(target)
        return {"text": 1, "music": 2}

    def status(self):
        return {"initialized": bool(self.initialized), "gem_instances": 1}

    def close(self):
        self.closed += 1


class DummyVideoStack:
    def __init__(self):
        self.initialized = False
        self.load_count = 0
        self.endecoder = object()
        self.shared_endecoder = None

    def initialize(self):
        self.initialized = True
        self.load_count += 1


class DummyMux:
    def __init__(self):
        self.start_count = 0
        self.closed = 0
        self.submitted: list[str] = []
        self.video = False
        self.estopped = False
        self.on_video_resume_reset = None

    def start(self):
        self.start_count += 1

    def submit_video_frame(self, _frame):
        pass

    def submit_generated_motion(self, path):
        self.submitted.append(str(path))

    def start_video_mode(self):
        self.video = True

    def stop_video_mode(self):
        self.video = False

    def set_idle(self):
        self.video = False

    def estop(self):
        self.estopped = True

    def clear_estop(self):
        self.estopped = False

    def status(self):
        return {
            "state": "VIDEO_LIVE" if self.video else "IDLE",
            "estop": self.estopped,
            "gmr_sender_instances": 1,
        }

    def close(self):
        self.closed += 1


class DummyVideoSession:
    def __init__(self):
        self.active = False
        self.pauses = 0
        self.resumes = 0
        self.resets = 0
        self.starts: list[dict] = []
        self.stops = 0
        self.closed = 0

    def start_source(self, **kwargs):
        self.active = True
        self.starts.append(kwargs)

    def stop_source(self):
        self.active = False
        self.stops += 1

    def pause_inference(self):
        self.pauses += 1

    def resume_inference(self):
        self.resumes += 1

    def request_reset(self):
        self.resets += 1

    def status(self):
        return {"active": self.active, "paused": self.pauses > self.resumes}

    def close(self):
        self.active = False
        self.closed += 1


@pytest.fixture()
def bundle(tmp_path: Path):
    audio_root = tmp_path / "audio"
    video_root = tmp_path / "video"
    output = tmp_path / "generation"
    for path in (audio_root, video_root, output):
        path.mkdir()
    (output / "READY").write_text("ready\n", encoding="utf-8")
    audio = audio_root / "song.wav"
    video = video_root / "clip.mp4"
    audio.write_bytes(b"audio")
    video.write_bytes(b"video")
    engine = DummyEngine(audio_root, output)
    stack = DummyVideoStack()
    mux = DummyMux()
    video_session = DummyVideoSession()
    service = server.MultimodalService(
        engine=engine,
        video_stack=stack,
        video_init="eager",
        allowed_video_roots=(video_root.resolve(),),
        mux=mux,
        video_session=video_session,
    )
    service.initialize()
    return service, engine, stack, mux, video_session, audio, video


@pytest.mark.parametrize("mode", ["video_text", "video_music", "video_text_music"])
def test_video_fusion_modes_return_explicit_unsupported_error(bundle, mode) -> None:
    service, *_ = bundle
    response, stop = service.handle({"op": "generate", "mode": mode, "prompt": "walk"})
    assert not stop and not response["ok"]
    assert response["error_type"] == "UnsupportedModeError"
    assert response["error"] == server.VIDEO_FUSION_ERROR


@pytest.mark.parametrize("mode", ["text", "music", "text_music"])
def test_generation_protocol_fields_and_mux_submission(bundle, mode) -> None:
    service, engine, _, mux, _, audio, _ = bundle
    payload = {"op": "generate", "mode": mode, "request_id": mode, "seed": 5}
    if mode in {"text", "text_music"}:
        payload["prompt"] = "walk"
    if mode in {"music", "text_music"}:
        payload["audio_path"] = str(audio)
        payload["start_sec"] = 1.5
    response, stop = service.handle(payload)
    assert response["ok"] and not stop
    assert mux.submitted[-1] == response["output_dir"]
    request = engine.requests[-1]
    assert request["mode"] == mode and request["seed"] == 5


def test_generation_pauses_and_resumes_active_video(bundle) -> None:
    service, _, _, _, video, _, _ = bundle
    video.active = True
    response, _ = service.handle({"op": "generate", "mode": "text", "prompt": "walk"})
    assert response["ok"]
    assert video.pauses == 1 and video.resumes == 1


def test_single_request_failure_does_not_stop_service(bundle) -> None:
    service, *_ = bundle
    failed, stop = service.handle({"op": "generate", "mode": "text", "prompt": "fail"})
    good, stop2 = service.handle({"op": "generate", "mode": "text", "prompt": "walk"})
    assert not failed["ok"] and not stop
    assert good["ok"] and not stop2


def test_video_start_stop_and_path_allowlist(bundle, tmp_path: Path) -> None:
    service, _, stack, mux, session, _, video = bundle
    response, _ = service.handle({"op": "video_start", "video_path": str(video)})
    assert response["ok"] and session.starts[-1]["video_path"] == video.resolve()
    assert mux.video and stack.load_count == 1
    stopped, _ = service.handle({"op": "video_stop"})
    assert stopped["ok"] and not mux.video

    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"video")
    rejected, _ = service.handle({"op": "video_start", "video_path": str(outside)})
    assert not rejected["ok"] and rejected["error_type"] == "PermissionError"


def test_symlink_and_dotdot_path_escape_are_rejected(bundle, tmp_path: Path) -> None:
    service, _, _, _, _, audio, _ = bundle
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"audio")
    link = audio.parent / "link.wav"
    link.symlink_to(outside)
    linked, _ = service.handle({"op": "generate", "mode": "music", "audio_path": str(link)})
    dotted, _ = service.handle(
        {
            "op": "generate",
            "mode": "music",
            "audio_path": str(audio.parent / ".." / "audio" / "song.wav"),
        }
    )
    assert linked["error_type"] == "PermissionError"
    assert dotted["error_type"] == "PermissionError"


def test_status_cache_estop_and_shutdown_protocol(bundle) -> None:
    service, engine, _, mux, _, _, _ = bundle
    status, _ = service.handle({"op": "status"})
    assert status["status"]["engine"]["gem_instances"] == 1
    assert status["status"]["mux"]["gmr_sender_instances"] == 1
    cleared, _ = service.handle({"op": "clear_cache", "target": "music"})
    assert cleared["removed"]["music"] == 2 and engine.clear_calls == ["music"]
    service.handle({"op": "estop"})
    assert mux.estopped
    service.handle({"op": "clear_estop"})
    assert not mux.estopped
    shutdown, stop = service.handle({"op": "shutdown"})
    assert shutdown["ok"] and stop


def test_fixed_clip_fields_cannot_be_overridden(bundle) -> None:
    service, *_ = bundle
    response, _ = service.handle(
        {
            "op": "generate",
            "mode": "text",
            "prompt": "walk",
            "clip_frames": 999,
        }
    )
    assert not response["ok"] and "unsupported request fields" in response["error"]


def _client_args(**values):
    defaults = {
        "timeout_ms": 1000,
        "mode": None,
        "video_start": False,
        "video_stop": False,
        "idle": False,
        "estop": False,
        "clear_estop": False,
        "status": False,
        "clear_cache": False,
        "shutdown": False,
        "request_id": None,
        "prompt": None,
        "audio": None,
        "start_sec": 0.0,
        "seed": 42,
        "camera_id": None,
        "video_path": None,
        "cache_target": "all",
    }
    defaults.update(values)
    return Namespace(**defaults)


def test_client_builds_joint_and_video_requests_strictly() -> None:
    joint = client.build_request(
        _client_args(
            mode="text_music",
            prompt="dance",
            audio="/server/song.wav",
            start_sec=2,
        )
    )
    assert joint == {
        "op": "generate",
        "mode": "text_music",
        "seed": 42,
        "prompt": "dance",
        "audio_path": "/server/song.wav",
        "start_sec": 2,
    }
    video = client.build_request(_client_args(video_start=True, camera_id=2))
    assert video == {"op": "video_start", "camera_id": 2}
    with pytest.raises(ValueError, match="requires --audio"):
        client.build_request(_client_args(mode="music"))
