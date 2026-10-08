"""使用真实已持久化GMT回复对照旧NPZ与新原始数组传输的CPU成本。

只读SQLite内一条advance回复，重建原dtype/shape并逐字段比较两个传输格式的结果；
预热后重复测量编码、解码和字节数，报告均值/P50/P95及原回复JSON的SHA身份。
不执行RPC、不推进物理、不加载GPU模型，不把单条回复微基准当作完整训练加速比。
输出必须为新JSON文件；调用者应在训练计时结束后运行，避免CPU争用污染正式对照。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gem.runtime.closedloop_protocol import _pack, _pack_legacy, _unpack


def restore(value):
    if isinstance(value, dict):
        if set(value) == {'__array__', 'shape', 'values'}:
            return np.asarray(restore(value['values']), dtype=value['__array__']).reshape(value['shape'])
        if set(value) == {'__nonfinite__'}:
            return float(value['__nonfinite__'])
        return {key: restore(child) for key, child in value.items()}
    if isinstance(value, list):
        return [restore(child) for child in value]
    return value


def compare(left, right):
    if isinstance(left, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError('Array dtype/shape changed')
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError('Reply fields changed')
        for key in left:
            compare(left[key], right[key])
    elif isinstance(left, list):
        if len(left) != len(right):
            raise ValueError('Reply list length changed')
        for a, b in zip(left, right):
            compare(a, b)
    elif left != right and not (isinstance(left, float) and np.isnan(left) and np.isnan(right)):
        raise ValueError('Reply scalar changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or not 2 <= args.repeats <= 100:
        raise ValueError('New output and 2..100 bounded repeats required')
    connection = sqlite3.connect(f'{args.journal.resolve().as_uri()}?mode=ro', uri=True)
    try:
        selected = connection.execute("SELECT identity,payload FROM replies WHERE instr(payload, ?) > 0 LIMIT 1",
                                      ('"operation":"advance"',)).fetchone()
    finally:
        connection.close()
    if selected is None:
        raise ValueError('Journal has no complete advance reply')
    identity, text = selected
    value = restore(json.loads(text))
    results = {}
    for name, encode in [('npz_v1', _pack_legacy), ('raw_v2', _pack)]:
        times = []
        for index in range(args.repeats+2):
            start = time.perf_counter(); meta, data = encode(value); middle = time.perf_counter()
            decoded = _unpack(meta, data); end = time.perf_counter()
            compare(value, decoded)
            if index >= 2:
                times.append([middle-start, end-middle])
        times = np.asarray(times)
        results[name] = dict(bytes=len(meta)+len(data), repeats=args.repeats,
            **{key: dict(mean=float(times[:, i].mean()), p50=float(np.percentile(times[:, i], 50)),
                         p95=float(np.percentile(times[:, i], 95))) for i, key in enumerate(['encode_seconds', 'decode_seconds'])})
    report = dict(schema='genmo.rpc.cpu_benchmark.v1', reply_identity=identity,
        source_journal=str(args.journal), reply_json_sha256=hashlib.sha256(text.encode()).hexdigest(),
        exact_fields_dtype_shape_values=True, cases=results, scope='one_real_reply_cpu_microbenchmark')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
