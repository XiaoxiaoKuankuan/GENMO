"""把 --calibrate-only 的单机器人部署标定结果写回 Stage10 配置的 timing.deployment_profile。

用途：数值核或随机核变化后，训练入口要求 deployment_profile 的 behavior_kernel_sha256、
numerical_execution 与 measured_assets 对应新的实际前向；此前需要手工复制 64 个延迟样本
和多处身份字段，容易遗漏。本工具只读取标定报告，核验八个rank的kernel、数值执行和
资产完全一致后，按rank顺序拼接延迟样本，计算与训练入口相同的kernel摘要
（sorted keys、紧凑分隔符的JSON SHA256），并写入provenance（报告路径与SHA）。

边界：只替换YAML中 timing.deployment_profile 这一段文本，其余配置与注释逐字保留；
写入后重新解析并确认除该段外的配置完全相同。不运行标定、不修改checkpoint或训练身份
之外的任何文件；schema/target/prefix_budget_seconds 沿用原配置。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def profile_from_calibration(report, previous, provenance_path):
    if report.get('status') != 'passed' or report.get('scope') != 'single_request_deployment_calibration_not_training':
        raise ValueError('A passed single-request deployment calibration report is required')
    calibrations = report['calibrations']
    if len(calibrations) != 8:
        raise ValueError('Eight-rank calibration required')
    first = calibrations[0]
    for item in calibrations[1:]:
        for key in ('kernel', 'numerical_execution', 'assets'):
            if item[key] != first[key]:
                raise ValueError(f'Calibration ranks disagree on {key}')
    samples = [float(x) for item in calibrations for row in item['durations'] for x in row]
    if len(samples) != 64 or not all(0 < x < previous['prefix_budget_seconds'] for x in samples):
        raise ValueError('Expected 64 positive latency samples within the prefix budget')
    kernel_sha = hashlib.sha256(json.dumps(first['kernel'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(schema=previous['schema'], behavior_kernel_sha256=kernel_sha, target=previous['target'],
        prefix_budget_seconds=previous['prefix_budget_seconds'], samples_seconds=samples,
        provenance=[dict(path=str(Path(provenance_path).resolve()), sha256=file_sha256(provenance_path))],
        measured_assets=dict(first['assets']), numerical_execution=dict(first['numerical_execution']))


def replace_profile_block(text, profile):
    lines = text.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.rstrip('\n') == '  deployment_profile:')
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or lines[end].startswith('    ')):
        end += 1
    body = yaml.safe_dump(profile, sort_keys=False, allow_unicode=True, default_flow_style=False)
    block = ['  deployment_profile:\n'] + ['    ' + line + '\n' for line in body.splitlines()]
    return ''.join(lines[:start] + block + lines[end:])


def update(config_path, calibration_path):
    config_path, calibration_path = Path(config_path), Path(calibration_path)
    text = config_path.read_text()
    before = yaml.safe_load(text)
    report = json.loads(calibration_path.read_text())
    profile = profile_from_calibration(report, before['timing']['deployment_profile'], calibration_path)
    updated = replace_profile_block(text, profile)
    after = yaml.safe_load(updated)
    if after['timing']['deployment_profile'] != profile:
        raise ValueError('Rewritten deployment profile does not parse back to the calibrated values')
    after['timing']['deployment_profile'] = before['timing']['deployment_profile']
    if after != before:
        raise ValueError('Profile rewrite changed configuration outside timing.deployment_profile')
    config_path.write_text(updated)
    return profile


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True, help='--calibrate-only 输出的 deployment_calibration.json')
    args = parser.parse_args()
    result = update(args.config, args.calibration)
    print(json.dumps(dict(behavior_kernel_sha256=result['behavior_kernel_sha256'],
        samples=len(result['samples_seconds']), maximum_seconds=max(result['samples_seconds'])), ensure_ascii=False))
