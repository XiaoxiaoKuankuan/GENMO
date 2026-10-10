"""在服务器1八rank上用真实世界回复对照执行证据外围优化，不推进或伪造物理。

每个rank从自己已封存world journal只读选择最大一条完整回复，先校验SHA，再使用
固定Git版本的参考函数和当前实现分别测量列式展开、journal编码、RPC编码/解码
及GMT独立回复复制。逐项检查字节或完整字段完全一致，随后同步预热、报告P50/P95。
计时不包含值比较，输入不修改；该工具不把单条CPU数据通路结果当作8192训练轮提速。
输出只允许新目录，原SQLite、模型和执行证据均只读。所有八rank绑定各自GPU以确认
资源身份，实际被测工作明确是CPU内存处理，不冒称GPU算子或实际物理验收。
"""
from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import types

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gem.closedloop.dppo import journal_codec
from gem.runtime import closedloop_protocol, trajectory_blocks
from tools.profile_closedloop_rpc import compare


def old_module(revision, path):
    source = subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=ROOT, text=True)
    name = 'evidence_reference_' + Path(path).stem
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, f'{revision}:{path}', 'exec'), module.__dict__)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iteration', type=Path, required=True)
    parser.add_argument('--gmt-repo', type=Path, required=True)
    parser.add_argument('--reference', default='6a42693')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    if world != 8 or rank != int(os.environ['RANK']) or not 3 <= args.repeats <= 50:
        raise ValueError('Requires eight local ranks and bounded repeats')
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo', timeout=timedelta(minutes=10))
    output = args.output / f'rank{rank:02d}'
    output.mkdir(parents=True, exist_ok=False)
    journal = args.iteration / f'rank{rank:02d}' / 'world_journal.sqlite'
    with sqlite3.connect(journal.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        identity, digest, raw = connection.execute(
            'SELECT identity,sha256,payload FROM replies ORDER BY length(payload) DESC LIMIT 1').fetchone()
    value = journal_codec.decode_payload(raw, sha256=digest)
    old_journal = old_module(args.reference, 'gem/closedloop/dppo/journal_codec.py')
    old_rpc = old_module(args.reference, 'gem/runtime/closedloop_protocol.py')
    old_trace = old_module(args.reference, 'gem/runtime/trajectory_blocks.py')
    path = args.gmt_repo / 'source/NoetixRobot/NoetixRobot/tasks/mimic/mimic_noetix_bumi4340_mha_sonic/closedloop/execution_copy.py'
    spec = importlib.util.spec_from_file_location('execution_copy_benchmark', path)
    clone_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(clone_module)
    feedback = [row['result']['result'] for row in value['result']['replies']
                if row.get('ok') and row.get('result', {}).get('operation') == 'advance'
                and row['result'].get('ok')]
    if not feedback: raise ValueError('Largest world reply contains no real advance')
    metadata, payload = old_rpc._pack(value)
    cases = {
        'journal_encode':(lambda:old_journal.encode_binary(value), lambda:journal_codec.encode_binary(value), True),
        'rpc_encode':(lambda:old_rpc._pack(value), lambda:closedloop_protocol._pack(value), True),
        'rpc_decode':(lambda:old_rpc._unpack(metadata, payload), lambda:closedloop_protocol._unpack(metadata, payload), False),
        'trace_expand':(lambda:[old_trace.expand_feedback(v, readonly_views=True) for v in feedback],
                        lambda:[trajectory_blocks.expand_feedback(v, readonly_views=True,lazy=True) for v in feedback], False),
        'reply_copy':(lambda:copy.deepcopy(value), lambda:clone_module.execution_copy(value), False),
    }
    # 同时测完整消费，避免只延后展开却把未计时成本藏到奖励阶段。
    cases['trace_expand_and_full_consume']=(
        lambda:copy.deepcopy([old_trace.expand_feedback(v,readonly_views=True) for v in feedback]),
        lambda:copy.deepcopy([trajectory_blocks.expand_feedback(v,readonly_views=True,lazy=True) for v in feedback]),False)
    result = {}
    for name, (reference, candidate, byte_exact) in cases.items():
        first, second = reference(), candidate()
        if byte_exact:
            if first != second: raise ValueError(f'{name} changed encoded bytes')
        else: compare(first, second)
        del first, second
        timings = {}
        for label, function in [('reference', reference), ('candidate', candidate)]:
            samples = []
            for repeat in range(args.repeats + 2):
                dist.barrier()
                start = time.perf_counter(); answer = function(); elapsed = time.perf_counter() - start
                del answer
                if repeat >= 2: samples.append(elapsed)
            timings[label] = dict(p50=float(np.percentile(samples, 50)), p95=float(np.percentile(samples, 95)), samples=samples)
        result[name] = dict(exact=True, **timings)
    report = dict(scope='real_world_reply_cpu_only', rank=rank, reference=args.reference,
                  source_identity=identity, source_sha256=digest, bytes=len(raw), feedback_lanes=len(feedback), cases=result)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    ranks = [None]*8; dist.all_gather_object(ranks, report)
    if rank == 0:
        (args.output/'report.json').write_text(json.dumps(dict(ranks=ranks), indent=2)+'\n')
    dist.destroy_process_group()


if __name__ == '__main__': main()
