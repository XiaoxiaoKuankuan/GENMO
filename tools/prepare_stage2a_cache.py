"""Stage2-A冻结GENMO缓存的有界逐歌曲补充生成，不覆写既有成功或失败资产。

消费已按音乐组隔离的任务清单，沿用仓库原真实GENMO预约、保护、Actor和物理生成
流程。每首歌以预先声明的延迟余量有限重试，保留每次真实轨迹和超时原因；一首失败
不会阻断其余歌曲。只把prepare_cache确认真实halo完整的产物加入成功清单，禁止
外推、拼接数据集未来动作或把短片段冒充完整歌曲。此命令不更新Stage1/Actor/Adapter。
"""

from pathlib import Path
import argparse
import json
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--margins', type=int, nargs='+', default=[120, 180])
    preliminary, _ = parser.parse_known_args()
    cfg = OmegaConf.to_container(OmegaConf.load(preliminary.config), resolve=True)
    sys.path.insert(0, cfg['supermimic']['repo'])
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(parser)
    args = parser.parse_args()
    import torch
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from gem.closedloop.reference_adapter.application import build_engine, read_tasks, raw_task, seed_all
    from gem.closedloop.reference_adapter.nominal import prepare_cache
    from gem.closedloop.reference_adapter.vector_entry import json_write
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    engine = build_engine(cfg, 'cuda:0')
    tasks = read_tasks(args.tasks, preparing=True)
    passed, failures = [], []
    sim_cfg = make_sim('cuda:0', args)
    json_write(output/'manifest.json', dict(config=cfg, tasks=tasks, margins=args.margins,
        command=[sys.executable, *sys.argv], method='unchanged_frozen_GENMO_real_accepted_plan_generation'))
    with launch_simulation(sim_cfg, args):
        env = StreamingPhysicalEnv(sim_cfg, 1, engine.step.bundle)
        for index, task in enumerate(tasks):
            for trial, margin in enumerate(args.margins):
                destination = output/f'task_{index:03d}_attempt_{trial}'
                destination.mkdir()
                try:
                    seed_all(task['seed'])
                    music, initial = raw_task(task, 'cuda:0')
                    generation = dict(cfg['genmo'], latency_margin_ticks=margin)
                    contract = prepare_cache(env, engine.step.bundle, generation, task, music, initial,
                        destination/'nominal.pt', bootstrap_latency_ticks=generation['bootstrap_latency_ticks'])
                    passed.append(task | dict(nominal_reference=str(destination/'nominal.pt'), generation_contract=contract))
                    json_write(output/'tasks_partial.json', passed, replace=True)
                    print(json.dumps(dict(task=task['id'], passed=True, margin=margin, completed=len(passed)), ensure_ascii=False), flush=True)
                    break
                except Exception:
                    failure = dict(task=task['id'], trial=trial, margin=margin, traceback=traceback.format_exc())
                    failures.append(failure)
                    json_write(destination/'failure.json', failure)
                    json_write(output/'failures.json', failures, replace=True)
                    print(json.dumps(dict(task=task['id'], passed=False, trial=trial, margin=margin), ensure_ascii=False), flush=True)
    json_write(output/'tasks.json', passed)
    json_write(output/'result.json', dict(requested=len(tasks), successful=len(passed), failures=len(failures),
        missing=[t['id'] for t in tasks if t['id'] not in {p['id'] for p in passed}], adapter_updates=0))
    return 0 if len(passed) == len(tasks) else 2


if __name__ == '__main__':
    raise SystemExit(main())
