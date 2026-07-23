"""CPU protocol tests for stdin/ZMQ resident text-motion service."""

from __future__ import annotations

import socket
import threading
import time
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.demo.demo_smpl_text_server import (
    RequestDefaults,
    apply_request_defaults,
    build_parser,
    handle_command,
    handle_json_message,
    parse_stdin_line,
    serve_stdin,
    serve_zmq,
)
from scripts.demo.text_motion_client import build_request, resolve_prompt, send_request


class ProtocolEngine:
    def __init__(self) -> None:
        self.requests = []
        self.cache = ["cached"]
        self.closed = False

    def generate(self, request):
        self.requests.append(request)
        if not request.get("prompt"):
            return {
                "ok": False,
                "request_id": request.get("request_id"),
                "error_type": "ValueError",
                "error": "prompt must not be empty",
            }
        return {
            "ok": True,
            "request_id": request.get("request_id"),
            "num_frames": request["num_frames"],
        }

    def status(self):
        return {"initialized": True, "request_count": len(self.requests)}

    def clear_cache(self):
        count = len(self.cache)
        self.cache.clear()
        return count


DEFAULTS = RequestDefaults(num_frames=120, fps=30.0, seed=42)


def test_stdin_plain_text_uses_defaults() -> None:
    kind, payload = parse_stdin_line("  Walk forward.  ", DEFAULTS)
    assert kind == "request"
    assert payload == {
        "prompt": "Walk forward.",
        "num_frames": 120,
        "fps": 30.0,
        "seed": 42,
    }


def test_stdin_json_preserves_explicit_values() -> None:
    kind, payload = parse_stdin_line(
        '{"request_id":"x","prompt":"wave","num_frames":7,"seed":9}', DEFAULTS
    )
    assert kind == "request"
    assert payload["request_id"] == "x"
    assert payload["num_frames"] == 7 and payload["seed"] == 9
    assert payload["fps"] == 30.0


def test_stdin_empty_line_is_ignored() -> None:
    assert parse_stdin_line(" \n", DEFAULTS) is None


def test_stdin_invalid_json_is_not_treated_as_prompt() -> None:
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_stdin_line('{"prompt":', DEFAULTS)


def test_status_command() -> None:
    response, stop = handle_command(ProtocolEngine(), "/status")
    assert response["ok"] and response["status"]["initialized"]
    assert stop is False


def test_clear_cache_command() -> None:
    engine = ProtocolEngine()
    response, stop = handle_command(engine, "/clear-cache")
    assert response == {"ok": True, "removed_embeddings": 1}
    assert engine.cache == [] and stop is False


def test_quit_command_requests_graceful_stop() -> None:
    response, stop = handle_command(ProtocolEngine(), "/quit")
    assert response["ok"] and stop is True


def test_json_request_success() -> None:
    engine = ProtocolEngine()
    response, stop = handle_json_message(engine, {"prompt": "walk"}, DEFAULTS)
    assert response["ok"] and response["num_frames"] == 120
    assert stop is False


def test_json_error_reply_does_not_stop_service() -> None:
    response, stop = handle_json_message(ProtocolEngine(), ["not", "object"], DEFAULTS)
    assert response["ok"] is False
    assert response["error_type"] == "TypeError"
    assert stop is False


def test_json_management_command() -> None:
    response, stop = handle_json_message(
        ProtocolEngine(), {"command": "status"}, DEFAULTS
    )
    assert response["ok"] and stop is False


def test_stdin_loop_recovers_after_invalid_line_and_quits() -> None:
    engine = ProtocolEngine()
    lines = iter(['{"prompt":', "walk", "/quit"])
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
    assert len(engine.requests) == 1
    assert any('"ok": false' in output for output in outputs)
    assert any('"ok": true' in output for output in outputs)


def _free_endpoint() -> str:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    return f"tcp://127.0.0.1:{port}"


def test_zmq_request_success_and_graceful_quit() -> None:
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
    response = send_request(endpoint, {"prompt": "walk"}, 2.0)
    assert response["ok"] and response["num_frames"] == 120
    quit_response = send_request(endpoint, {"command": "quit"}, 2.0)
    assert quit_response["ok"]
    thread.join(timeout=2)
    assert not thread.is_alive() and stop.is_set()


def test_zmq_error_reply_keeps_server_alive() -> None:
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
    failed = send_request(endpoint, ["invalid"], 2.0)
    succeeded = send_request(endpoint, {"prompt": "wave"}, 2.0)
    assert failed["ok"] is False
    assert succeeded["ok"] is True
    send_request(endpoint, {"command": "quit"}, 2.0)
    thread.join(timeout=2)


def test_server_cli_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.transport == "stdin"
    assert args.device == "cuda:0"
    assert args.ddim_steps == 20
    assert args.num_frames == 120
    assert args.warmup is True
    assert args.shape_mode == "zero"


def test_request_defaults_do_not_override_explicit_values() -> None:
    payload = apply_request_defaults(
        {"prompt": "walk", "num_frames": 300, "fps": 50, "seed": 7}, DEFAULTS
    )
    assert payload["num_frames"] == 300
    assert payload["fps"] == 50 and payload["seed"] == 7


def test_client_prompt_file_and_payload(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("  turn left  ", encoding="utf-8")
    assert resolve_prompt(None, prompt_file) == "turn left"
    args = Namespace(
        prompt=None,
        prompt_file=prompt_file,
        num_frames=20,
        fps=25.0,
        seed=3,
        request_id="client-1",
        timeout_seconds=10.0,
    )
    request = build_request(args)
    assert request == {
        "request_id": "client-1",
        "prompt": "turn left",
        "num_frames": 20,
        "fps": 25.0,
        "seed": 3,
    }
