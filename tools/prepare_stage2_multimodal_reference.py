#!/usr/bin/env python3
"""离线生成联合 Stage1 的音乐/文本名义缓存，供同一个 Stage2 Adapter训练与配对评价。

只读显式配置、任务和冻结Stage1/Actor资产，加载RobotSpec取得关节顺序；生成所有
任务及未来十帧加导数halo的真实采样点。初态和条件摘要写入v2缓存合同，不把源动作
标签复制成生成结果，不外推末帧。该入口不启动物理、WM、优化器或正式训练；每次
仅写新输出目录，失败保留对应任务与异常，不伪称失败任务已生成。
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--tasks')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    sys.path.insert(0, cfg['supermimic']['repo'])
    from supermimic.bridge.bundle import load_bundle
    from gem.closedloop.reference_adapter.application import read_tasks, raw_task, seed_all
    from gem.closedloop.reference_adapter.nominal import prepare_cache
    tasks = read_tasks(args.tasks or cfg['tasks'], preparing=True)
    if any('condition_kind' not in task for task in tasks):
        raise ValueError('离线联合缓存入口需要显式condition_kind和任务时长')
    bundle = load_bundle(cfg['supermimic']['checkpoint'], 'cpu', robot_asset_dir=cfg['supermimic']['robot_asset_dir'])
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepared = []
    for index, task in enumerate(tasks):
        seed_all(task['seed'])
        music, initial = raw_task(task, args.device)
        target = output/f'nominal_{index:04d}.pt'
        contract = prepare_cache(None, bundle, cfg['genmo'], task, music, initial, target, bootstrap_latency_ticks=0)
        prepared.append(task | dict(nominal_reference=str(target), generation_contract=contract))
        (output/'tasks_partial.json').write_text(json.dumps(prepared, ensure_ascii=False, indent=2))
        print(json.dumps(dict(task=task['id'], condition_kind=task['condition_kind'], completed=len(prepared))), flush=True)
    (output/'tasks.json').write_text(json.dumps(prepared, ensure_ascii=False, indent=2))
    cfg['tasks'] = str(output/'tasks.json')
    OmegaConf.save(OmegaConf.create(cfg), output/'runtime.yaml')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
