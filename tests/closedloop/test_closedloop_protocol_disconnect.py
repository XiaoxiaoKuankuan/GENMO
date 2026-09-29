"""验证冻结闭环 RPC 在客户端断开后的服务存活和执行结果重放。

本文件只使用 CPU 上的 Unix socket 与有界线程，不导入模型或仿真。测试后端使用
与真实 advance_id 缓存相同的约束：同一 episode、步数、advance_id 重放返回原结果，
不能再次推进模拟控制时钟；传输序号则随新连接重新从 1 开始。

通过同步事件，在后端已提交推进结果、服务端准备回复时断开客户端，并定向注入
BrokenPipeError/ConnectionResetError，避免依赖不同内核上不稳定的 TCP/Unix 断开
表现。另外验证接收侧 reset、非法长度/JSON/根类型只关闭当前连接，不执行 handler，
且后续合法连接仍可使用；close 已执行但回复丢失时必须退出并清理自有 socket。
测试不通过 xfail 隐藏回归，所有线程均有超时、显式关闭和泄漏断言。
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import struct
import threading
import time

import pytest

from gem.runtime import closedloop_protocol as protocol


class CachedAdvanceHandler:
    """只模拟不可重复的推进和业务缓存，不宣称验证了真实动力学。"""

    def __init__(self):
        self.cache = {}
        self.tick = 600
        self.dispatch_count = 0
        self.execution_count = 0
        self.close_count = 0

    def advance(self, advance_id, control_steps, expected_episode_id):
        self.dispatch_count += 1
        assert expected_episode_id == "episode-1"
        key = str(advance_id)
        parameters = (expected_episode_id, control_steps)
        if key in self.cache:
            original_parameters, result = self.cache[key]
            if parameters != original_parameters:
                raise ValueError("Repeated advance_id has different parameters")
            return copy.deepcopy(result)
        self.execution_count += 1
        begin = self.tick
        self.tick += control_steps * protocol.CONTROL_TICKS
        result = {"advance_id": key, "episode_id": expected_episode_id,
                  "control_tick_begin": begin, "control_tick_end": self.tick,
                  "executed_control_steps": control_steps,
                  "executed_physics_steps": control_steps * 4}
        self.cache[key] = (parameters, copy.deepcopy(result))
        return result

    def close(self):
        self.close_count += 1
        return {"closed": True}


def connect(path):
    """bind 创建路径略早于 listen；有界重试只处理这一启动竞态。"""
    deadline = time.monotonic() + 2
    while True:
        try:
            return protocol.RpcClient(path, timeout_s=2)
        except ConnectionRefusedError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.001)


@contextmanager
def serving(tmp_path):
    path = tmp_path / "disconnect.sock"
    handler, failures = CachedAdvanceHandler(), []

    def run():
        try:
            protocol.RpcServer(path, handler).serve()
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=run, name="rpc-disconnect-server", daemon=True)
    thread.start()
    deadline = time.monotonic() + 2
    while not path.exists() and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert path.exists(), f"Server failed to bind: {failures!r}"
    try:
        yield path, handler, thread, failures
    finally:
        if thread.is_alive():
            try:
                client = connect(path)
                try:
                    client.call("close")
                finally:
                    client.close()
            except (OSError, EOFError):
                # 缺陷版本可能恰好在清理开始时退出；仍在下面保留真实服务端异常。
                pass
        thread.join(timeout=2)
        assert not thread.is_alive(), "RPC test leaked a server thread"
        assert not path.exists(), "RPC server did not remove its own socket"
        assert not failures, f"RPC server exited after a per-connection failure: {failures!r}"


def call_advance(client, *, advance_id="advance-1", control_steps=25):
    return client.call("advance", advance_id=advance_id, control_steps=control_steps,
                       expected_episode_id="episode-1")


def inject_lost_reply(monkeypatch, error_type, *, result_key):
    """服务已调用 handler 后通知测试关闭 peer，再在原发送位置触发确定的连接错误。"""
    original_send = protocol.send_message
    reply_ready, peer_closed = threading.Event(), threading.Event()
    injected = []

    def send(connection, value):
        if (threading.current_thread().name == "rpc-disconnect-server"
                and not injected and value.get("ok") and result_key in value.get("result", {})):
            injected.append(error_type.__name__)
            reply_ready.set()
            assert peer_closed.wait(timeout=2), "Client did not close before dropped reply"
            raise error_type("peer disconnected after handler committed")
        return original_send(connection, value)

    monkeypatch.setattr(protocol, "send_message", send)
    return reply_ready, peer_closed, injected


@pytest.mark.parametrize("error_type", [BrokenPipeError, ConnectionResetError])
def test_lost_advance_reply_reconnect_replays_cache_without_advancing_twice(tmp_path, monkeypatch, error_type):
    reply_ready, peer_closed, injected = inject_lost_reply(monkeypatch, error_type, result_key="advance_id")
    with serving(tmp_path) as (path, handler, thread, failures):
        first = connect(path)
        protocol.send_message(first.connection, {
            "version": protocol.PROTOCOL_VERSION, "sequence": 1, "method": "advance",
            "payload": {"advance_id": "advance-1", "control_steps": 25,
                        "expected_episode_id": "episode-1"},
        })
        assert reply_ready.wait(timeout=2)
        first.close()
        peer_closed.set()
        assert handler.execution_count == 1 and handler.tick == 900
        second = connect(path)
        try:
            result = call_advance(second)
            assert result == handler.cache["advance-1"][1]
            assert result["control_tick_begin"] == 600 and result["control_tick_end"] == 900
            assert result["executed_control_steps"] == 25 and result["executed_physics_steps"] == 100
            assert second.sequence == 1, "A reconnect has its own transport sequence"
            assert handler.dispatch_count == 2 and handler.execution_count == 1
            assert handler.tick == 900
            with pytest.raises(protocol.RemoteError, match="different parameters"):
                call_advance(second, control_steps=24)
            assert handler.execution_count == 1 and handler.tick == 900
            assert call_advance(second, advance_id="advance-2", control_steps=1)["control_tick_end"] == 912
        finally:
            second.close()
        assert injected == [error_type.__name__]
        assert thread.is_alive() and not failures


def test_receive_reset_only_closes_connection_and_accepts_new_client(tmp_path, monkeypatch):
    original_receive = protocol.receive_message
    reset_seen = threading.Event()

    def receive(connection):
        if threading.current_thread().name == "rpc-disconnect-server" and not reset_seen.is_set():
            reset_seen.set()
            raise ConnectionResetError("client reset while reading request")
        return original_receive(connection)

    monkeypatch.setattr(protocol, "receive_message", receive)
    with serving(tmp_path) as (path, handler, thread, failures):
        first = connect(path)
        assert reset_seen.wait(timeout=2)
        first.close()
        second = connect(path)
        try:
            assert call_advance(second, control_steps=1)["control_tick_end"] == 612
        finally:
            second.close()
        assert handler.dispatch_count == 1 and handler.execution_count == 1
        assert thread.is_alive() and not failures


@pytest.mark.parametrize("packet", [
    struct.pack("!QQ", 0, 0),
    struct.pack("!QQ", protocol.MAX_PACKET_BYTES, 1),
    struct.pack("!QQ", 1, 0) + b"{",
    struct.pack("!QQ", 2, 0) + b"[]",
    struct.pack("!QQ", 4, 0) + b"null",
])
def test_invalid_packet_closes_only_bad_connection_without_handler_dispatch(tmp_path, packet):
    with serving(tmp_path) as (path, handler, thread, failures):
        first = connect(path)
        try:
            first.connection.sendall(packet)
            try:
                assert first.connection.recv(1) == b"", "Malformed packet must not receive a business reply"
            except ConnectionResetError:
                pass
        finally:
            first.close()
        assert handler.dispatch_count == 0
        second = connect(path)
        try:
            assert call_advance(second, control_steps=1)["control_tick_end"] == 612
        finally:
            second.close()
        assert handler.dispatch_count == 1 and thread.is_alive() and not failures


def test_partial_request_eof_reconnects_without_dispatch(tmp_path):
    with serving(tmp_path) as (path, handler, thread, failures):
        first = connect(path)
        first.connection.sendall(struct.pack("!QQ", 99, 0) + b"{")
        first.close()
        second = connect(path)
        try:
            assert call_advance(second, control_steps=1)["control_tick_end"] == 612
        finally:
            second.close()
        assert handler.dispatch_count == 1 and thread.is_alive() and not failures


@pytest.mark.parametrize("error_type", [BrokenPipeError, ConnectionResetError])
def test_close_reply_loss_still_exits_and_removes_socket(tmp_path, monkeypatch, error_type):
    reply_ready, peer_closed, injected = inject_lost_reply(monkeypatch, error_type, result_key="closed")
    with serving(tmp_path) as (path, handler, thread, failures):
        first = connect(path)
        protocol.send_message(first.connection, {"version": protocol.PROTOCOL_VERSION,
            "sequence": 1, "method": "close", "payload": {}})
        assert reply_ready.wait(timeout=2)
        first.close()
        peer_closed.set()
        thread.join(timeout=2)
        assert handler.close_count == 1
        assert not thread.is_alive() and not path.exists() and not failures
        assert injected == [error_type.__name__]
