"""CPU protocol tests for the resident music-to-motion service."""

from __future__ import annotations

import socket
import threading
import time
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.demo.demo_music_server import (
    RequestDefaults,
    apply_request_defaults,
    build_parser,
    handle_command,
    handle_json_message,
    parse_stdin_line,
    serve_stdin,
    serve_zmq,
)
from scripts.demo.music_motion_client import (
    build_request,
    send_request,
)


class ProtocolEngine:
    def __init__(self) -> None:
        self.requests = []
        self.cache = ["cached"]

    def generate(self, request):
        self.requests.append(request)
        if request.get("audio_path") == "bad.wav":
            return {
                "ok": False,
                "request_id": request.get("request_id"),
                "error_type": "ValueError",
                "error": "bad audio",
            }
        return {
            "ok": True,
            "request_id": request.get("request_id"),
            "audio_path": request["audio_path"],
            "duration_sec": request["duration_sec"],
        }

    def status(self):
        return {"initialized": True, "request_count": len(self.requests)}

    def clear_cache(self):
        removed = len(self.cache)
        self.cache.clear()
        return removed


DEFAULTS = RequestDefaults(start_sec=0.0, duration_sec=10.0, seed=42)


def test_stdin_plain_and_space_paths_use_defaults() -> None:
    kind, payload = parse_stdin_line("/data/song.wav", DEFAULTS)
    assert kind == "request"
    assert payload == {
        "audio_path": "/data/song.wav",
        "start_sec": 0.0,
        "duration_sec": 10.0,
        "seed": 42,
    }
    _, spaced = parse_stdin_line("/data/My Song 01.wav", DEFAULTS)
    assert spaced["audio_path"] == "/data/My Song 01.wav"


@pytest.mark.parametrize(
    "line",
    ['"/data/My Song.wav"', "'/data/My Song.wav'"],
)
def test_stdin_matching_outer_quotes_are_removed(line: str) -> None:
    _, payload = parse_stdin_line(line, DEFAULTS)
    assert payload["audio_path"] == "/data/My Song.wav"


def test_stdin_json_defaults_and_explicit_values() -> None:
    _, defaulted = parse_stdin_line('{"audio_path":"/a.wav"}', DEFAULTS)
    assert defaulted["duration_sec"] == 10.0
    _, explicit = parse_stdin_line(
        '{"request_id":"m1","audio_path":"/a.wav","start_sec":2,"duration_sec":null,"seed":7}',
        DEFAULTS,
    )
    assert explicit["request_id"] == "m1"
    assert explicit["start_sec"] == 2
    assert explicit["duration_sec"] is None
    assert explicit["seed"] == 7


def test_invalid_json_does_not_fall_back_to_path() -> None:
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_stdin_line('{"audio_path":', DEFAULTS)


def test_empty_line_and_unknown_command() -> None:
    assert parse_stdin_line(" \n", DEFAULTS) is None
    with pytest.raises(ValueError, match="unknown command"):
        parse_stdin_line("/unknown", DEFAULTS)


def test_status_help_clear_cache_and_quit() -> None:
    engine = ProtocolEngine()
    status, stop = handle_command(engine, "/status")
    assert status["ok"] and status["status"]["initialized"] and not stop
    help_response, stop = handle_command(engine, "/help")
    assert help_response["ok"] and "help" in help_response and not stop
    cleared, stop = handle_command(engine, "/clear-cache")
    assert cleared == {"ok": True, "removed_features": 1}
    assert not stop and engine.cache == []
    quit_response, stop = handle_command(engine, "/quit")
    assert quit_response["ok"] and stop


def test_stdin_recovers_after_error_then_quits() -> None:
    engine = ProtocolEngine()
    lines = iter(['{"audio_path":', "bad.wav", "good.wav", "/quit"])
    outputs = []
    stop = threading.Event()
    serve_stdin(
        engine,
        DEFAULTS,
        stop,
        input_fn=lambda _prompt: next(lines),
        output_fn=outputs.append,
    )
    assert stop.is_set()
    assert len(engine.requests) == 2
    assert any('"ok": false' in output for output in outputs)
    assert any('"audio_path": "good.wav"' in output for output in outputs)


def test_json_message_error_does_not_stop_service() -> None:
    response, stop = handle_json_message(ProtocolEngine(), ["bad"], DEFAULTS)
    assert response["ok"] is False
    assert response["error_type"] == "TypeError"
    assert stop is False


def _free_endpoint() -> str:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def test_zmq_success_error_recovery_and_graceful_quit() -> None:
    pytest.importorskip("zmq")
    endpoint = _free_endpoint()
    engine = ProtocolEngine()
    stop = threading.Event()
    thread = threading.Thread(
        target=serve_zmq,
        args=(engine, DEFAULTS, endpoint, stop),
        daemon=True,
    )
    thread.start()
    time.sleep(0.05)
    success = send_request(endpoint, {"audio_path": "song.wav"}, 2.0)
    failed = send_request(endpoint, ["invalid"], 2.0)
    recovered = send_request(endpoint, {"audio_path": "next.wav"}, 2.0)
    assert success["ok"]
    assert failed["ok"] is False
    assert recovered["ok"]
    quit_response = send_request(endpoint, {"command": "quit"}, 2.0)
    assert quit_response["ok"]
    thread.join(timeout=2)
    assert not thread.is_alive() and stop.is_set()


def test_client_payload_full_range_and_metadata(tmp_path: Path) -> None:
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text('{"operator":"alice"}', encoding="utf-8")
    args = Namespace(
        audio=Path("/shared/My Song.wav"),
        start_sec=3.0,
        duration_sec=10.0,
        full=True,
        seed=9,
        request_id="client-1",
        timeout_seconds=60.0,
        metadata_json=str(metadata_file),
    )
    payload = build_request(args)
    assert payload == {
        "request_id": "client-1",
        "audio_path": "/shared/My Song.wav",
        "start_sec": 3.0,
        "duration_sec": None,
        "seed": 9,
        "metadata": {"operator": "alice"},
    }


def test_client_timeout() -> None:
    pytest.importorskip("zmq")
    with pytest.raises(TimeoutError, match="did not reply"):
        send_request(_free_endpoint(), {"audio_path": "song.wav"}, 0.05)


def test_server_cli_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.transport == "stdin"
    assert args.bind == "tcp://127.0.0.1:7011"
    assert args.device == "cuda:0"
    assert args.duration_sec == 10.0
    assert args.ddim_steps == 20
    assert args.warmup is True
    assert args.shape_mode == "zero"
    assert args.feature_cache_size == 32


def test_request_defaults_preserve_explicit_null_duration() -> None:
    payload = apply_request_defaults(
        {
            "audio_path": "song.wav",
            "start_sec": 4,
            "duration_sec": None,
            "seed": 7,
        },
        DEFAULTS,
    )
    assert payload["start_sec"] == 4
    assert payload["duration_sec"] is None
    assert payload["seed"] == 7
