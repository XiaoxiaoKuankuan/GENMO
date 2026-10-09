"""只读统计完整世界journal的真实执行与后端阶段时间，支持失败采样证据。

直接解析规范二进制帧的元数据，不展开大型数组、不加载Actor、不重跑物理或奖励。
每个rank读取自己的权威世界journal，逐条验证完整SHA，累计GMT提供的原始计时键，
保留嵌套口径，不把它们简单相加。报告物理/控制计数、已持久化字节、回复分布、
生成批块数与时间间隔；失败任务的计数不冒充已接受训练数据或8192轮吞吐。

入口在服务器1八rank启动，输入路径只读，输出必须不存在；不覆盖原模型或证据。
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iteration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    directory = args.iteration/f'rank{rank:02d}'
    timing, operations, controls, physics = Counter(), Counter(), Counter(), Counter()
    payload_bytes = records = 0
    with sqlite3.connect(f'file:{directory}/world_journal.sqlite?mode=ro', uri=True) as connection:
        for digest, payload in connection.execute('SELECT sha256,payload FROM replies ORDER BY rowid'):
            if hashlib.sha256(payload).hexdigest() != digest: raise ValueError('Changed journal evidence')
            if payload[:8] != b'GENMOJ2\0': raise ValueError('Requires binary authoritative world journal')
            metadata, arrays = struct.unpack('!QQ', payload[8:24])
            if len(payload) != 24+metadata+arrays: raise ValueError('Truncated journal')
            record = json.loads(payload[24:24+metadata])['value']
            records += 1; payload_bytes += len(payload)
            result = record.get('result', {})
            timing.update(result.get('timing', {}))
            for reply in result.get('replies', []):
                envelope = reply.get('result', {})
                operation = envelope.get('operation', 'readonly_or_ack')
                operations[operation] += 1
                if operation == 'advance' and envelope.get('ok'):
                    feedback = envelope['result']; key = str(feedback['env_id'])
                    controls[key] += feedback['executed_control_steps']
                    physics[key] += feedback['executed_physics_steps']
    raw = sorted((directory/'raw_samples').glob('world_batch_*.pt'))
    times = [path.stat().st_mtime for path in raw]
    output = dict(rank=rank, journal_records=records, authoritative_payload_bytes=payload_bytes,
        raw_chain_bytes=sum(path.stat().st_size for path in raw), generation_blocks=len(raw),
        generation_block_intervals_seconds=[b-a for a,b in zip(times,times[1:])],
        operation_reply_counts=dict(operations), controls_by_env=dict(controls), physics_by_env=dict(physics),
        backend_timing=dict(timing), timing_scope='raw_backend_keys_may_be_nested_do_not_sum',
        accepted_rollout=(directory/'collection.json').exists())
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/f'rank{rank:02d}.json').open('x') as stream:json.dump(output, stream, indent=2)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':main()
