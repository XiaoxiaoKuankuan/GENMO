"""AWR不可变数据的二进制编码、内容校验与原子发布。

输入为CPU Tensor、NumPy数组及小型Python元数据树。所有数组按dtype打包成少量
连续npy文件，JSON只保存shape/offset及标量；避免每个控制步或条件字段一个文件。
落盘先在同盘临时目录fsync，再rename发布。读取默认mmap并校验文件摘要；
本模块不读取任意pickle，不修改已发布内容，也不承担Replay淘汰策略。
v2保留公共物理回执的列共享关系：每块字段表仅存一次，行只保存表号和索引；
读取核验列结构并恢复只读视图，兼容已有v1密集Episode，不改任何奖励或条件数值。
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from gem.runtime.trajectory_blocks import _TraceRow, _validate_lazy_columns


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_tree(path, tree):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + path.name + ".", dir=path.parent))
    groups, sizes, column_fields, column_ids = {}, {}, [], {}

    def column_count(node):
        if node["kind"] in ("array", "scalar", "rows"):
            return len(node["values"])
        if node["kind"] in ("mapping", "list", "tuple"):
            fields = node["fields"].values() if node["kind"] == "mapping" else node["fields"]
            for child in fields:
                count = column_count(child)
                if count is not None:
                    return count
        return None

    def encode(value):
        if isinstance(value, _TraceRow):
            key = id(value._fields)
            if key not in column_ids:
                column_ids[key] = len(column_fields)
                count = column_count(dict(kind="mapping", fields=value._fields))
                entry = dict(fields=None, count=count, max_index=value._index)
                column_fields.append(entry)
                entry["fields"] = encode(value._fields)
            ordinal = column_ids[key]
            column_fields[ordinal]["max_index"] = max(value._index, column_fields[ordinal]["max_index"])
            return {"__column_row__": ordinal, "index": value._index}
        tensor = torch.is_tensor(value)
        if tensor:
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                if all(isinstance(v, str) for v in value.reshape(-1)):
                    return {"__object_strings__": encode(value.astype(str))}
                raise TypeError("Object arrays are not an AWR storage format")
            key = value.dtype.str
            offset = sizes.get(key, 0)
            array = np.ascontiguousarray(value).reshape(-1)
            groups.setdefault(key, []).append(array)
            sizes[key] = offset + array.size
            return {"__array__": key, "offset": offset, "shape": list(value.shape),
                    "tensor": tensor}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return {"__tuple__": [encode(v) for v in value]}
        if isinstance(value, list):
            return [encode(v) for v in value]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError(f"Unsupported AWR data type: {type(value)}")

    try:
        structure = encode(tree)
        files = {}
        for ordinal, (dtype, chunks) in enumerate(sorted(groups.items())):
            filename = f"arrays_{ordinal:02d}.npy"
            with (temporary / filename).open("wb") as stream:
                np.save(stream, np.concatenate(chunks), allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            files[dtype] = {"file": filename, "sha256": file_sha256(temporary / filename)}
        manifest = {"schema": "genmo.awr.array_tree.v1", "tree": structure, "files": files}
        if column_fields:
            for entry in column_fields:
                if entry["count"] is None:
                    entry["count"] = entry["max_index"]+1
            manifest["column_fields"] = column_fields
            manifest["schema"] = "genmo.awr.array_tree.v2"
        manifest["sha256"] = digest(manifest)
        atomic_json(temporary / "manifest.json", manifest)
        if path.exists():
            old = json.loads((path / "manifest.json").read_text())
            if old != manifest:
                raise ValueError("Immutable data identity already exists with different content")
        else:
            os.rename(temporary, path)
            sync_directory(path.parent)
        return manifest["sha256"]
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def read_tree(path, *, verify=True, episode_metadata_keys=None):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["schema"] not in ("genmo.awr.array_tree.v1", "genmo.awr.array_tree.v2"):
        raise ValueError("Unknown array tree schema")
    if (manifest["schema"] == "genmo.awr.array_tree.v2") != ("column_fields" in manifest):
        raise ValueError("Columnar storage requires its explicit v2 schema")
    if digest({k: v for k, v in manifest.items() if k != "sha256"}) != manifest["sha256"]:
        raise ValueError("Array manifest identity mismatch")
    arrays = {}
    for dtype, entry in manifest["files"].items():
        target = (path / entry["file"]).resolve()
        if not target.is_relative_to(path.resolve()):
            raise ValueError("Array path escaped immutable shard")
        if verify and file_sha256(target) != entry["sha256"]:
            raise ValueError("Replay array checksum mismatch")
        # copy-on-write映射提供可写Tensor视图，但任何使用方写入都不会修改磁盘。
        arrays[dtype] = np.load(target, mmap_mode="c", allow_pickle=False)

    restored_fields = {}

    def decode(value):
        if isinstance(value, dict) and "__column_row__" in value:
            ordinal, index = value["__column_row__"], value["index"]
            entry = manifest["column_fields"][ordinal]
            if not 0 <= index < entry["count"]:
                raise ValueError("Stored physical row is outside its authoritative column")
            if ordinal not in restored_fields:
                fields = decode(entry["fields"])
                _validate_lazy_columns(dict(kind="mapping", fields=fields), entry["count"])
                restored_fields[ordinal] = fields
            return _TraceRow(restored_fields[ordinal], index)
        if isinstance(value, dict) and "__object_strings__" in value:
            return decode(value["__object_strings__"]).astype(object)
        if isinstance(value, dict) and "__array__" in value:
            size = math.prod(value["shape"])
            array = arrays[value["__array__"]][value["offset"]:value["offset"] + size]
            array = array.reshape(value["shape"])
            return torch.from_numpy(array) if value["tensor"] else array
        if isinstance(value, dict) and "__tuple__" in value:
            return tuple(decode(v) for v in value["__tuple__"])
        if isinstance(value, dict):
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value
    tree = manifest["tree"]
    if episode_metadata_keys is not None:
        # 完整manifest及所有数组SHA已经验证；只减少训练常驻对象，不删改物理原始分片。
        if not isinstance(tree, dict) or not isinstance(tree.get("rows"), list):
            raise ValueError("Episode projection requires an actual complete Episode tree")
        for row in tree["rows"]:
            row["metadata"] = {k: v for k, v in row["metadata"].items() if k in episode_metadata_keys}
    return decode(tree)


def cpu_copy(value, memo=None):
    # 公共物理回执本来已按列存储；保留一块多行共享的列，避免每行重复展开所有物理字段。
    memo = {} if memo is None else memo
    if isinstance(value, _TraceRow):
        return value.snapshot_columns(cpu_copy, memo)
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {k: cpu_copy(v, memo) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(v, memo) for v in value)
    return value
