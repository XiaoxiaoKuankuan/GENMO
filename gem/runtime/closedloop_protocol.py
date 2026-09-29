"""冻结 GENMO—Isaac GMT 闭环的轻量执行协议和本机 RPC。

本模块仅依赖 Python 标准库和 NumPy，不导入 Actor、训练 Dataset、Isaac 或 MuJoCo。
两个解释器共享此协议；物理时间统一使用 600 Hz 整数 tick，而 wall time 只表示计算耗时。
状态、请求和反馈携带 episode/request/plan 身份；实际数组以禁止 pickle 的 NPZ 字节传递，
不会在每次规划时落盘。RPC 是单连接顺序请求；物理执行的幂等性由后端 advance_id 实现。
socket 只在指定私有目录创建，退出删除本进程创建的 socket，不触碰其他进程或实验。
"""
from __future__ import annotations

import dataclasses
import io
import json
import socket
import struct
import zipfile
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

PROTOCOL_VERSION = "genmo.gmt_frozen_isaac.v1"
CLOCK_HZ = 600
MOTION_TICKS = 20
CONTROL_TICKS = 12
PHYSICS_TICKS = 3
DECISION_TICKS = 300
MAX_PACKET_BYTES = 64 * 1024 * 1024


class StateSnapshot(TypedDict, total=False):
    env_id: int
    episode_id: str | int
    tick: int
    history_values: np.ndarray
    history_valid: np.ndarray
    history_ticks: np.ndarray
    robot_qpos: np.ndarray
    done: bool
    terminated: bool
    truncated: bool
    reason: str | None


class PlanRequest(TypedDict, total=False):
    env_id: int
    episode_id: str | int
    request_id: str
    decision_id: int
    decision_tick: int
    deadline_tick: int
    min_prefix: int


class GeneratedPlan(PlanRequest, total=False):
    plan_id: str
    parent_plan_id: str
    prefix_frames: int
    protected_end_tick: int
    qpos_world: np.ndarray
    qpos30: np.ndarray
    contact: np.ndarray


class ExecutionFeedback(TypedDict, total=False):
    snapshot: StateSnapshot
    trace: list[dict[str, Any]]
    executed_control_steps: int
    executed_physics_steps: int
    advance_id: str


class RemoteError(RuntimeError):
    def __init__(self, error: dict[str, Any]):
        self.code = error.get("code") or error.get("type", "RemoteError")
        self.remote_error = error
        super().__init__(f"{self.code}: {error.get('message', '')}")


def _pack(value: Any) -> tuple[bytes, bytes]:
    arrays: dict[str, np.ndarray] = {}

    def visit(item):
        if dataclasses.is_dataclass(item):
            item = dataclasses.asdict(item)
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject or item.dtype.kind not in "biufUS":
                raise TypeError(f"Unsupported array dtype: {item.dtype}")
            key = f"a{len(arrays)}"
            arrays[key] = np.array(item, copy=True, order="C")
            return {"__array__": key}
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise TypeError("RPC mapping keys must be strings")
            return {k: visit(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(v) for v in item]
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(f"Unsupported RPC value: {type(item).__name__}")

    metadata = json.dumps(visit(value), ensure_ascii=False, allow_nan=False).encode("utf-8")
    buffer = io.BytesIO()
    if arrays:
        np.savez(buffer, **arrays)
    return metadata, buffer.getvalue()


def _unpack(metadata: bytes, payload: bytes):
    values = json.loads(metadata)
    arrays = {}
    if payload:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
            if any(a.dtype.hasobject or a.dtype.kind not in "biufUS" for a in arrays.values()):
                raise TypeError("Unsupported received RPC array dtype")

    def visit(item):
        if isinstance(item, dict):
            if set(item) == {"__array__"}:
                return arrays[item["__array__"]]
            return {k: visit(v) for k, v in item.items()}
        if isinstance(item, list):
            return [visit(v) for v in item]
        return item

    return visit(values)


def _read_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        data = connection.recv(size - len(chunks))
        if not data:
            raise EOFError("RPC peer closed the connection")
        chunks.extend(data)
    return bytes(chunks)


def send_message(connection: socket.socket, value):
    metadata, payload = _pack(value)
    if len(metadata) + len(payload) > MAX_PACKET_BYTES:
        raise ValueError("RPC packet exceeds 64 MiB")
    connection.sendall(struct.pack("!QQ", len(metadata), len(payload)) + metadata + payload)


def receive_message(connection: socket.socket):
    meta_size, payload_size = struct.unpack("!QQ", _read_exact(connection, 16))
    if not meta_size or meta_size + payload_size > MAX_PACKET_BYTES:
        raise ValueError("Invalid RPC packet length")
    return _unpack(_read_exact(connection, meta_size), _read_exact(connection, payload_size))


class RpcClient:
    def __init__(self, socket_path, timeout_s: float = 120):
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.connection.settimeout(timeout_s)
        try:
            self.connection.connect(str(socket_path))
        except BaseException:
            self.connection.close()
            raise
        self.sequence = 0

    def call(self, method: str, **payload):
        self.sequence += 1
        send_message(self.connection, {"version": PROTOCOL_VERSION, "sequence": self.sequence,
                                       "method": method, "payload": payload})
        response = receive_message(self.connection)
        if response.get("sequence") != self.sequence or response.get("version") != PROTOCOL_VERSION:
            raise RuntimeError("RPC response identity/version mismatch")
        if not response.get("ok"):
            raise RemoteError(response["error"])
        return response["result"]

    def close(self):
        self.connection.close()


class RpcServer:
    def __init__(self, socket_path, handler):
        self.path = Path(socket_path)
        self.handler = handler

    def serve(self):
        if self.path.exists():
            raise FileExistsError(f"Refusing to replace socket: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound = False
        try:
            server.bind(str(self.path))
            bound = True
            self.path.chmod(0o600)
            server.listen(1)
            finished = False
            while not finished:
                connection, _ = server.accept()
                with connection:
                    last_sequence = 0
                    while not finished:
                        try:
                            request = receive_message(connection)
                            if not isinstance(request, dict):
                                raise ValueError("RPC request envelope must be a mapping")
                        except (EOFError, ConnectionError, ValueError, TypeError, KeyError, zipfile.BadZipFile):
                            # 断链或非法帧只终止该连接；未解码完成的请求绝不派发到后端。
                            break
                        response = {"version": PROTOCOL_VERSION, "sequence": request.get("sequence")}
                        try:
                            if request.get("version") != PROTOCOL_VERSION:
                                raise ValueError("RPC protocol version mismatch")
                            sequence = request.get("sequence")
                            if type(sequence) is not int or sequence != last_sequence + 1:
                                raise ValueError("RPC request sequence mismatch")
                            # 已接收的合法传输序号只消费一次，业务错误也不能重放该序号。
                            last_sequence = sequence
                            method = request["method"]
                            if not isinstance(method, str) or method.startswith("_"):
                                raise ValueError("Invalid RPC method")
                            result = getattr(self.handler, method)(**request.get("payload", {}))
                            response.update(ok=True, result=result)
                            finished = method == "close"
                        except Exception as exc:
                            response.update(ok=False, error={"type": type(exc).__name__,
                                            "message": str(exc), "code": getattr(exc, "code", None)})
                        try:
                            send_message(connection, response)
                        except ConnectionError:
                            # 后端可能已推进并缓存 advance_id；保留服务供新连接取回原结果，
                            # 这里不能重试 handler。close 已完成时 finished 保持 True。
                            break
        finally:
            server.close()
            if bound:
                self.path.unlink(missing_ok=True)
