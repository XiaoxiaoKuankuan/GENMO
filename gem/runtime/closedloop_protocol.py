"""冻结 GENMO—Isaac GMT 闭环的轻量执行协议和本机 RPC。

本模块仅依赖 Python 标准库和 NumPy，不导入 Actor、训练 Dataset、Isaac 或 MuJoCo。
两个解释器共享此协议；物理时间统一使用 600 Hz 整数 tick，而 wall time 只表示计算耗时。
状态、请求和反馈携带 episode/request/plan 身份；实际数组以禁止 pickle 的有界原始字节传递，
新传输格式显式标记v2并兼容读取旧NPZ帧，保持dtype/shape和独立可写结果，
不会在每次规划时落盘。RPC 是单连接顺序请求；物理执行的幂等性由后端 advance_id 实现。
socket 只在指定私有目录创建，退出删除本进程创建的 socket，不触碰其他进程或实验。
"""
from __future__ import annotations

import dataclasses
from contextvars import ContextVar
import io
import json
import socket
import struct
import time
import zipfile
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

_CALL_TIMING = ContextVar("rpc_call_timing", default=None)

PROTOCOL_VERSION = "genmo.gmt_frozen_isaac.v1"
WIRE_VERSION = "genmo.rpc.ndarray.raw.v2"
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


def _pack_legacy(value: Any) -> tuple[bytes, bytes]:
    arrays: dict[str, np.ndarray] = {}

    def visit(item):
        if dataclasses.is_dataclass(item):
            item = dataclasses.asdict(item)
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject or item.dtype.kind not in "biufUS":
                raise TypeError(f"Unsupported array dtype: {item.dtype}")
            key = f"a{len(arrays)}"
            # 单连接同步编码完成才返回；连续数组直接读取，np.savez写入独立字节快照。
            arrays[key] = item if item.flags.c_contiguous else np.ascontiguousarray(item)
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


def _raw_packet_reference(value: Any, *, size_only=False):
    """无ZIP成员循环：数组连续拼接，描述符显式给出dtype/shape/offset/size。"""
    parts, offsets, total = [], {}, 0
    def visit(item):
        nonlocal total
        kind = type(item)
        if kind in (type(None), str, int, float, bool):
            return item
        if kind is dict:
            if any(not isinstance(key, str) for key in item):
                raise TypeError('RPC mapping keys must be strings')
            return {key:visit(child) for key,child in item.items()}
        if kind in (list, tuple):
            return [visit(child) for child in item]
        if dataclasses.is_dataclass(item):
            item = dataclasses.asdict(item)
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject or item.dtype.kind not in "biufUS":
                raise TypeError(f"Unsupported array dtype: {item.dtype}")
            if id(item) not in offsets:
                size = item.nbytes
                if total+size > MAX_PACKET_BYTES:
                    raise ValueError('RPC arrays exceed packet limit')
                offsets[id(item)] = (item, dict(offset=total, size=size, dtype=item.dtype.str, shape=list(item.shape)))
                if not size_only:parts.append(item.tobytes(order='C'))
                total += size
            return {'__ndarray_raw__': offsets[id(item)][1]}
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError('RPC mapping keys must be strings')
            return {key: visit(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(child) for child in item]
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(f'Unsupported RPC value: {type(item).__name__}')
    metadata = json.dumps(dict(__rpc_wire__=WIRE_VERSION, value=visit(value)), ensure_ascii=False,
                          allow_nan=False, separators=(',', ':')).encode('utf-8')
    return metadata, total if size_only else b''.join(parts)


def _raw_packet(value: Any, *, size_only=False):
    """原v2字节合同的C层JSON快路；不再重建完整标量树，数组别名/上限不变。"""
    def native(item):
        kind=type(item)
        if kind in (type(None),str,int,float,bool,np.ndarray):return True
        if kind is dict:
            if any(not isinstance(k,str) for k in item):raise TypeError('RPC mapping keys must be strings')
            return all(native(v) for v in item.values())
        if kind in (list,tuple):return all(native(v) for v in item)
        return False
    if not native(value):return _raw_packet_reference(value,size_only=size_only)
    parts=[];offsets={};total=0
    def default(item):
        nonlocal total
        if type(item) is not np.ndarray or item.dtype.hasobject or item.dtype.kind not in 'biufUS':
            raise TypeError('Unsupported RPC array dtype/value')
        if id(item) not in offsets:
            size=item.nbytes
            if total+size>MAX_PACKET_BYTES:raise ValueError('RPC arrays exceed packet limit')
            offsets[id(item)]=(item,dict(offset=total,size=size,dtype=item.dtype.str,shape=list(item.shape)))
            if not size_only:parts.append(item.tobytes(order='C'))
            total+=size
        return {'__ndarray_raw__':offsets[id(item)][1]}
    metadata=json.dumps(dict(__rpc_wire__=WIRE_VERSION,value=value),ensure_ascii=False,
        allow_nan=False,separators=(',',':'),default=default).encode('utf-8')
    return metadata,total if size_only else b''.join(parts)


def _pack(value: Any) -> tuple[bytes, bytes]:
    return _raw_packet(value)


def raw_message_size(value: Any) -> int:
    """精确计算v2元数据及数组字节，不复制数组；供完整回复按现有包上限分页。

    元数据沿用真正编码器及同一数组别名规则，避免低估长尾证据。只计算每条已完成
    回复一次；跨回复求和是保守上界，不提高MAX_PACKET_BYTES或减少任何证据。
    """
    metadata,size=_raw_packet(value,size_only=True)
    return len(metadata)+size


def _decode_raw_array(spec, payload):
    """一次验证并恢复原始数组；同一描述符的多次引用仍各自独立可写。"""
    if not isinstance(spec, dict) or set(spec) != {'offset', 'size', 'dtype', 'shape'}:
        raise ValueError('Invalid raw array descriptor')
    offset, size, shape = spec['offset'], spec['size'], spec['shape']
    if (type(offset) is not int or type(size) is not int or offset < 0 or size < 0
            or offset+size > len(payload) or not isinstance(shape, list)
            or any(type(x) is not int or x < 0 for x in shape) or len(shape) > 32):
        raise ValueError('Raw array bounds/shape invalid')
    dtype = np.dtype(spec['dtype'])
    if dtype.hasobject or dtype.kind not in 'biufUS':
        raise TypeError('Unsupported received RPC array dtype')
    count = 1
    for dimension in shape:
        count *= dimension
    if count*dtype.itemsize != size or size > MAX_PACKET_BYTES:
        raise ValueError('Raw array byte count differs from shape/dtype')
    return np.frombuffer(payload, dtype=dtype, count=count, offset=offset).reshape(shape).copy()


def _unpack_raw(values, payload):
    def visit(item):
        if isinstance(item, dict):
            if set(item) == {'__ndarray_raw__'}:
                return _decode_raw_array(item['__ndarray_raw__'], payload)
            return {key: visit(child) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child) for child in item]
        return item
    return visit(values)


def _unpack(metadata: bytes, payload: bytes):
    # 本实现的标准 v2 帧拥有固定首字段。在 JSON 构建 dict 的同时恢复数组，
    # 避免解析后再递归重建整棵树。非标准排序 v2 和旧 NPZ 仍走原兼容路径。
    if metadata.startswith(b'{"__rpc_wire__":'):
        def decode_object(item):
            if len(item) == 1 and '__ndarray_raw__' in item:
                return _decode_raw_array(item['__ndarray_raw__'], payload)
            return item
        values = json.loads(metadata, object_hook=decode_object)
        if (not isinstance(values, dict) or values.get('__rpc_wire__') != WIRE_VERSION
                or set(values) != {'__rpc_wire__', 'value'}):
            raise ValueError('Unsupported RPC wire format')
        return values['value']
    values = json.loads(metadata)
    if isinstance(values, dict) and "__rpc_wire__" in values:
        if values.get("__rpc_wire__") != WIRE_VERSION or set(values) != {"__rpc_wire__", "value"}:
            raise ValueError("Unsupported RPC wire format")
        return _unpack_raw(values["value"], payload)
    arrays = {}
    if payload:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            # NPZ解码已分配独立数组；不再复制第二份，关闭archive不影响数组。
            arrays = {k: archive[k] for k in archive.files}
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


def send_message(connection: socket.socket, value, *, timing=None):
    timing = _CALL_TIMING.get() if timing is None else timing
    started = time.perf_counter()
    metadata, payload = _pack(value)
    encoded = time.perf_counter()
    if timing is not None:
        timing["encode_seconds"] = encoded-started
        timing["sent_bytes"] = 16+len(metadata)+len(payload)
    if len(metadata) + len(payload) > MAX_PACKET_BYTES:
        raise ValueError("RPC packet exceeds 64 MiB")
    connection.sendall(struct.pack("!QQ", len(metadata), len(payload)) + metadata + payload)
    if timing is not None:
        timing["send_seconds"] = time.perf_counter()-encoded


def receive_message(connection: socket.socket, *, timing=None):
    timing = _CALL_TIMING.get() if timing is None else timing
    started = time.perf_counter()
    meta_size, payload_size = struct.unpack("!QQ", _read_exact(connection, 16))
    if not meta_size or meta_size + payload_size > MAX_PACKET_BYTES:
        raise ValueError("Invalid RPC packet length")
    metadata, payload = _read_exact(connection, meta_size), _read_exact(connection, payload_size)
    received = time.perf_counter()
    result = _unpack(metadata, payload)
    if timing is not None:
        timing.update(receive_including_remote_seconds=received-started,
                      decode_seconds=time.perf_counter()-received, received_bytes=16+meta_size+payload_size)
    return result


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
        self.last_call_timing = {}

    def call(self, method: str, **payload):
        self.sequence += 1
        timing = self.last_call_timing = {}
        token = _CALL_TIMING.set(timing)
        try:
            send_message(self.connection, {"version": PROTOCOL_VERSION, "sequence": self.sequence,
                                           "method": method, "payload": payload})
            response = receive_message(self.connection)
        finally:
            _CALL_TIMING.reset(token)
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
        # 仅累计协议CPU/传输数据，不把等待下一次训练请求算作CPU计算。
        statistics=self.handler._rpc_transport_statistics={}
        def accumulate(prefix, timing):
            for key,value in timing.items():
                if key=='receive_including_remote_seconds':continue
                name=prefix+key;statistics[name]=statistics.get(name,0.)+value
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
                            receive_timing={}
                            request = receive_message(connection,timing=receive_timing)
                            accumulate('request_',receive_timing)
                            statistics['received_requests']=statistics.get('received_requests',0)+1
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
                            send_timing={}
                            send_message(connection, response,timing=send_timing)
                            accumulate('reply_',send_timing)
                        except ConnectionError:
                            # 后端可能已推进并缓存 advance_id；保留服务供新连接取回原结果，
                            # 这里不能重试 handler。close 已完成时 finished 保持 True。
                            break
        finally:
            server.close()
            if bound:
                self.path.unlink(missing_ok=True)
