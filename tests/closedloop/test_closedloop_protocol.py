"""Stage8 本机 RPC 协议的线程／socket CPU 验收。

测试 numpy dtype、形状和 Unicode 计划来源能完整往返，不使用 pickle，不加载
GENMO 模型、Isaac 或 MuJoCo。通过 socketpair 检查帧边界和客户端身份校验，
通过 pytest 私有临时 Unix socket 检查服务端序号、远端异常、close 退出和清理。
重复 advance_id 的业务幂等由后端负责；传输层只拒绝重复传输序号，不能假装
物理执行已经幂等。所有服务线程均有超时和显式退出，不留下后台服务。
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import io
import socket
import struct
import threading
import time

import numpy as np
import pytest

from gem.runtime import closedloop_protocol as protocol


@contextmanager
def socket_pair():
    first, second = socket.socketpair()
    first.settimeout(2)
    second.settimeout(2)
    try:
        yield first, second
    finally:
        first.close()
        second.close()


class Handler:
    def __init__(self):
        self.calls = []
        self.closed = False

    def echo(self, **payload):
        self.calls.append(("echo", payload))
        return payload

    def advance(self, advance_id):
        self.calls.append(("advance", advance_id))
        return {"advance_id": advance_id, "handler_calls": len(self.calls)}

    def fail(self):
        class ReferenceFailure(ValueError):
            code = "late_plan"
        raise ReferenceFailure("arrival exceeded deadline")

    def close(self):
        self.closed = True
        return {"closed": True}


@contextmanager
def serving(tmp_path, handler=None):
    handler = handler or Handler()
    path = tmp_path / "rpc.sock"
    failures = []
    def serve():
        try:
            protocol.RpcServer(path, handler).serve()
        except BaseException as exc:
            failures.append(exc)
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    deadline = time.monotonic() + 2
    while not path.exists() and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.001)
    if failures:
        raise failures[0]
    assert path.exists(), "RPC server did not bind in time"
    # Unix 路径在 bind 后已存在，但 listen 可能尚未执行；用一次空连接确认服务就绪。
    while True:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as ready:
            try:
                ready.connect(str(path))
                break
            except ConnectionRefusedError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.001)
    try:
        yield path, handler, thread
    finally:
        if thread.is_alive():
            client = protocol.RpcClient(path, timeout_s=2)
            try:
                client.call("close")
            finally:
                client.close()
        thread.join(timeout=2)
        assert not thread.is_alive(), "RPC server leaked a background thread"
        assert not path.exists()
        if failures:
            raise failures[0]


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint64, np.int8, np.bool_])
def test_numpy_array_dtype_shape_and_noncontiguous_roundtrip(dtype):
    original = np.arange(12).reshape(3, 4).astype(dtype)[:, ::2]
    with socket_pair() as (sender, receiver):
        protocol.send_message(sender, {"array": original})
        value = protocol.receive_message(receiver)["array"]
    assert value.dtype == original.dtype and value.shape == original.shape
    np.testing.assert_array_equal(value, original)
    assert not np.shares_memory(value, original)


@pytest.mark.parametrize("shape", [(), (0,), (1,), (0, 3)])
def test_zero_dim_and_empty_numpy_shapes_preserved(shape):
    original = np.zeros(shape, dtype=np.float32)
    metadata, payload = protocol._pack(original)
    value = protocol._unpack(metadata, payload)
    assert value.shape == shape and value.dtype == original.dtype


def test_nested_feedback_unicode_origins_and_numpy_scalars_roundtrip():
    @dataclass
    class Nested:
        tick: object
        valid: object
    value = {"trace": [{"origins": np.array(["bootstrap:0", "计划:音乐春天"], dtype="U")},
                       {"opaque": np.array([b"abc"], dtype="S3")}],
             "nested": Nested(np.int64(612), np.bool_(True)),
             "tuples": (np.float32(0.02), None)}
    metadata, payload = protocol._pack(value)
    restored = protocol._unpack(metadata, payload)
    np.testing.assert_array_equal(restored["trace"][0]["origins"], value["trace"][0]["origins"])
    np.testing.assert_array_equal(restored["trace"][1]["opaque"], value["trace"][1]["opaque"])
    assert restored["nested"] == {"tick": 612, "valid": True}
    assert restored["tuples"][1] is None


@pytest.mark.parametrize("invalid", [np.array([object()], dtype=object), np.array([1+2j]),
                                     {1: "numeric key"}, {"bad": float("nan")}])
def test_disallowed_or_nonjson_data_rejected(invalid):
    with pytest.raises((TypeError, ValueError)):
        protocol._pack(invalid)


def test_incoming_complex_array_rejected():
    stream = io.BytesIO()
    np.savez(stream, a0=np.array([1+2j]))
    with pytest.raises(TypeError, match="received RPC array dtype"):
        protocol._unpack(b'{"__array__":"a0"}', stream.getvalue())


@pytest.mark.parametrize("header", [(0, 0), (protocol.MAX_PACKET_BYTES, 1)])
def test_bad_packet_lengths_rejected_before_payload_read(header):
    with socket_pair() as (sender, receiver):
        sender.sendall(struct.pack("!QQ", *header))
        with pytest.raises(ValueError, match="packet length"):
            protocol.receive_message(receiver)


def test_truncated_packet_reports_eof():
    with socket_pair() as (sender, receiver):
        sender.sendall(struct.pack("!QQ", 10, 0) + b"{}")
        sender.shutdown(socket.SHUT_WR)
        with pytest.raises(EOFError):
            protocol.receive_message(receiver)


@pytest.mark.parametrize("bad_field,bad_value", [("sequence", 0), ("version", "old-v0")])
def test_client_rejects_response_identity_mismatch(bad_field, bad_value):
    with socket_pair() as (client_socket, server_socket):
        client = protocol.RpcClient.__new__(protocol.RpcClient)
        client.connection, client.sequence = client_socket, 0
        reply = {"version": protocol.PROTOCOL_VERSION, "sequence": 1,
                 "ok": True, "result": {"unexpected": True}}
        reply[bad_field] = bad_value
        protocol.send_message(server_socket, reply)
        with pytest.raises(RuntimeError, match="identity/version mismatch"):
            client.call("snapshot")
        sent = protocol.receive_message(server_socket)
        assert sent["method"] == "snapshot" and sent["sequence"] == 1


def test_server_echo_remote_error_and_normal_close(tmp_path):
    with serving(tmp_path) as (path, handler, thread):
        client = protocol.RpcClient(path, timeout_s=2)
        try:
            result = client.call("echo", origins=np.array(["计划一", "bootstrap"]), tick=np.int64(300))
            np.testing.assert_array_equal(result["origins"], ["计划一", "bootstrap"])
            assert result["tick"] == 300
            with pytest.raises(protocol.RemoteError) as exc:
                client.call("fail")
            assert exc.value.code == "late_plan"
            assert exc.value.remote_error["type"] == "ReferenceFailure"
            assert client.call("close") == {"closed": True}
        finally:
            client.close()
        thread.join(timeout=2)
        assert handler.closed and not thread.is_alive()


def test_server_rejects_wrong_version_without_handler_execution(tmp_path):
    with serving(tmp_path) as (path, handler, thread):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            connection.connect(str(path))
            protocol.send_message(connection, {"version": "wrong", "sequence": 1,
                                               "method": "advance", "payload": {"advance_id": "one"}})
            response = protocol.receive_message(connection)
            assert response["ok"] is False
            assert "version mismatch" in response["error"]["message"]
            assert handler.calls == []


@pytest.mark.parametrize("sequence", [0, 2, True, "1"])
def test_server_rejects_invalid_initial_sequence(tmp_path, sequence):
    with serving(tmp_path) as (path, handler, thread):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            connection.connect(str(path))
            protocol.send_message(connection, {"version": protocol.PROTOCOL_VERSION, "sequence": sequence,
                                               "method": "advance", "payload": {"advance_id": "one"}})
            response = protocol.receive_message(connection)
            assert response["ok"] is False
            assert "sequence mismatch" in response["error"]["message"]
            assert handler.calls == []


def test_transport_rejects_duplicate_sequence_but_does_not_fake_advance_idempotence(tmp_path):
    with serving(tmp_path) as (path, handler, thread):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            connection.connect(str(path))
            request = {"version": protocol.PROTOCOL_VERSION, "sequence": 1,
                       "method": "advance", "payload": {"advance_id": "same"}}
            protocol.send_message(connection, request)
            assert protocol.receive_message(connection)["result"]["handler_calls"] == 1
            protocol.send_message(connection, request)
            assert protocol.receive_message(connection)["ok"] is False
            assert len(handler.calls) == 1
            request["sequence"] = 2
            protocol.send_message(connection, request)
            assert protocol.receive_message(connection)["result"]["handler_calls"] == 2
            assert handler.calls == [("advance", "same"), ("advance", "same")]


def test_server_will_not_replace_existing_socket_path(tmp_path):
    path = tmp_path / "rpc.sock"
    path.write_text("owned by another process")
    with pytest.raises(FileExistsError):
        protocol.RpcServer(path, Handler()).serve()
    assert path.read_text() == "owned by another process"
