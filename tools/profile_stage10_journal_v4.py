"""用一条真实已归档 advance 回复测量执行日志的各项 CPU 成本。

对同一完整回复分别执行旧 JSON 与二进制编码、SHA256、SQLite FULL 事务和解码。
每次逐字段比较 dtype/shape/值，输出均值、P50、P95、字节量及来源 SHA。事务使用
独立系统临时目录并自动清理，不修改输入 journal，也不连接 GMT 或启动 GPU。
这里的结果仅是单回复局部基准，不能写成整轮或八卡采样的加速比例。
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from gem.closedloop.dppo.buffer import StepJournal, _journal_value
from gem.closedloop.dppo.journal_codec import FORMAT, encode_binary, decode_payload
from tools.profile_closedloop_rpc import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reply', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or not 2 <= args.repeats <= 100:
        raise ValueError('Require new output and bounded repeats')
    raw = args.reply.read_bytes()
    value = decode_payload(raw)
    results = {}
    for format in ('json.v1', FORMAT):
        times = []
        with tempfile.TemporaryDirectory(prefix='genmo-journal-v4-') as folder:
            with StepJournal(Path(folder)/'journal.sqlite', format=format) as journal:
                for repeat in range(args.repeats+2):
                    value['mutation_seq'] = repeat+1
                    begin = time.perf_counter()
                    payload = (encode_binary(value) if format == FORMAT else
                        json.dumps(_journal_value(value), ensure_ascii=False, sort_keys=True,
                                   separators=(',',':'), allow_nan=False).encode())
                    encoded = time.perf_counter()
                    digest = hashlib.sha256(payload).hexdigest()
                    hashed = time.perf_counter()
                    from gem.closedloop.dppo.buffer import EncodedJournalReply
                    identity = json.dumps([value['backend_session_id'], value['mutation_seq']], separators=(',',':'))
                    journal.append_encoded(EncodedJournalReply(identity, payload, digest, format))
                    persisted = time.perf_counter()
                    decoded = decode_payload(payload)
                    ended = time.perf_counter()
                    compare(value, decoded)
                    if repeat >= 2:
                        times.append([encoded-begin, hashed-encoded, persisted-hashed, ended-persisted])
        values = np.asarray(times)
        results[format] = dict(payload_bytes=len(payload), full_synchronous=True,
            timings={name:dict(mean=float(values[:,i].mean()), p50=float(np.quantile(values[:,i],.5)),
                              p95=float(np.quantile(values[:,i],.95)))
                     for i,name in enumerate(('encode_seconds','sha256_seconds','transaction_seconds','decode_seconds'))})
    args.output.write_text(json.dumps(dict(scope='single_real_reply_cpu_microbenchmark_not_round',
        source_sha256=hashlib.sha256(raw).hexdigest(), repeats=args.repeats, field_parity=True,
        results=results), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
