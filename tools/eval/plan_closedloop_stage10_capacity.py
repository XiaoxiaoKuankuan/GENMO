"""依据已完成训练的真实文件占用，为第十步下一段训练做只读容量规划。

工具读取参考运行中已接受轮次的摘要、完整checkpoint大小及该轮执行证据大小，取
实测最大值估算目标总轮数；另外保留失败重试轮数、比例余量、checkpoint暂存空间
和文件系统最小空闲。它不删除旧产物、不扩预算、不加载模型、不启动GPU或训练。
输出把“已测占用”和“估计未来容量”分开，估计失败时退出非零；运行期间仍必须由
DiskGuard逐次约束实际写入。target目录已有数据全部计入占用，不能靠旧账本归零。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import yaml


def _bytes(root):
    root = Path(root)
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Capacity evidence cannot contain symlinks')
        if path.is_file():
            total += path.stat().st_size
    return total


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Accepted checkpoint must be a regular file inside the reference run')
    return path


def _published_iteration(root):
    """只抵扣latest已完成发布的进度；仅写出accepted摘要的中断尝试仍需重做。"""
    latest_path = root/'latest.json'
    if not latest_path.exists():
        return 0
    latest = json.loads(latest_path.read_text())
    if (latest.get('schema') != 'genmo.closedloop.stage10.checkpoint_publication.v1'
            or type(latest.get('iteration')) is not int or latest['iteration'] < 1):
        raise ValueError('Capacity target has invalid latest publication')
    publication = json.loads(_inside(root, latest['publication']).read_text())
    if publication != {key:value for key,value in latest.items() if key != 'publication'}:
        raise ValueError('Capacity target latest disagrees with immutable publication')
    checkpoint = _inside(root, latest['path'])
    digest = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    if checkpoint.stat().st_size != latest['size_bytes'] or digest.hexdigest() != latest['sha256']:
        raise ValueError('Capacity target latest checkpoint size or SHA differs')
    return latest['iteration']


def plan_capacity(reference_run, target_run, configuration, *, safety_factor=1.25,
                  retry_iterations=2, free_bytes=None):
    reference_run, target_run = Path(reference_run).resolve(), Path(target_run).resolve()
    if not math.isfinite(safety_factor) or safety_factor < 1:
        raise ValueError('safety_factor must be finite and >=1')
    if type(retry_iterations) is not int or retry_iterations < 0:
        raise ValueError('retry_iterations must be a nonnegative integer')
    stage = configuration['stage10']
    target_iterations = stage['limits']['accepted_iterations']
    if type(target_iterations) is not int or target_iterations < 1:
        raise ValueError('Target accepted iterations must be positive')
    observed = []
    for path in sorted(reference_run.glob('sessions/*/iterations/*/summary.json')):
        raw = path.read_bytes()
        summary = json.loads(raw)
        if summary.get('status') != 'accepted':
            continue
        checkpoint = _inside(reference_run, summary['checkpoint'])
        observed.append(dict(iteration=summary['iteration'], summary=str(path.relative_to(reference_run)),
            summary_sha256=hashlib.sha256(raw).hexdigest(), checkpoint_bytes=checkpoint.stat().st_size,
            execution_evidence_bytes=_bytes(path.parent)))
    if len(observed) < 2:
        raise ValueError('Capacity planning requires at least two completed iteration measurements')
    per_iteration = max(row['checkpoint_bytes'] for row in observed) + max(row['execution_evidence_bytes'] for row in observed)
    observed_total = _bytes(reference_run)
    measured_iterations = sum(row['checkpoint_bytes']+row['execution_evidence_bytes'] for row in observed)
    fixed_overhead = max(0, observed_total-measured_iterations)
    target_used = _bytes(target_run)
    existing_iterations = _published_iteration(target_run)
    # 已有目录的所有字节原样计入，再保守计算剩余轮次；未接受的尝试不会抵扣目标轮数。
    if existing_iterations > target_iterations:
        raise ValueError('Capacity target iteration limit cannot be below its published progress')
    remaining = max(0, target_iterations-existing_iterations)
    startup_overhead = fixed_overhead if existing_iterations == 0 else 0
    projected_new = math.ceil((startup_overhead+(remaining+retry_iterations)*per_iteration)*safety_factor)
    checkpoint_reserve = int(stage['storage']['checkpoint_reserve_bytes'])
    projected_new += checkpoint_reserve
    projected_total = target_used+projected_new
    filesystem = target_run
    while not filesystem.exists():
        filesystem = filesystem.parent
    free_bytes = shutil.disk_usage(filesystem).free if free_bytes is None else int(free_bytes)
    quota, reserve = int(stage['storage']['max_run_bytes']), int(stage['storage']['min_free_bytes'])
    checks = dict(within_run_quota=projected_total <= quota,
                  filesystem_keeps_free_reserve=free_bytes-projected_new >= reserve)
    return dict(schema='genmo.closedloop.stage10.capacity_plan.v1',
        status='passed' if all(checks.values()) else 'failed', checks=checks,
        reference_run=str(reference_run), target_run=str(target_run), measurements=observed,
        observed_run_bytes=observed_total, observed_fixed_overhead_bytes=fixed_overhead,
        max_iteration_bytes=per_iteration, target_existing_bytes=target_used,
        existing_accepted_iterations=existing_iterations, target_total_iterations=target_iterations,
        remaining_iterations=remaining, retry_iterations=retry_iterations, safety_factor=safety_factor,
        checkpoint_reserve_bytes=checkpoint_reserve, projected_additional_bytes=projected_new,
        projected_total_bytes=projected_total, max_run_bytes=quota,
        filesystem_free_bytes=free_bytes, min_free_bytes=reserve,
        semantics='Storage estimate from measured maxima; actual writes remain guarded; no deletion, budget extension or training performed.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, required=True)
    parser.add_argument('--target-run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--safety-factor', type=float, default=1.25)
    parser.add_argument('--retry-iterations', type=int, default=2)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    result = plan_capacity(args.reference_run, args.target_run, config,
        safety_factor=args.safety_factor, retry_iterations=args.retry_iterations)
    result['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(f"Stage10 capacity plan {result['status']}: projected={result['projected_total_bytes']/2**30:.2f} GiB, quota={result['max_run_bytes']/2**30:.2f} GiB")
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
