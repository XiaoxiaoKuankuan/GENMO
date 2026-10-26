"""服务器上真实独立物理采集进程的并行交接验收，不执行GENMO或价值优化。

父进程在CPU装配真实资产和数据合同，指定GPU上的子进程各自执行Newton、冻结SM和
完整GENMO。核对同一Actor版本、完整真实anchor、独立collector标识、冻结参数及
采集时间重叠；保存原始响应和报告。它只证明并行采集工程可用，不能证明梯度方向、
Critic质量、正式训练收益或多步优化成功。
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", type=int, nargs=2, required=True)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2:
        raise ValueError("真实并行验收需要两张独立GPU")
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    sys.path.insert(0, cfg["supermimic"]["repo"])
    from gem.closedloop.mb_control.application import build_components, read_tasks, seed_all
    from gem.closedloop.mb_control.parallel_collection import RemotePhysicalCollector
    from supermimic.bridge.bundle import atomic_save
    seed_all(cfg["training"]["seed"])
    actor, adapter, _, _, _, _, _ = build_components(cfg, "cpu")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = output/"config.yaml"
    OmegaConf.save(OmegaConf.create(cfg), config)
    weights = output/"actor_for_collection.pt"
    atomic_save({"schema": "genmo.collection.actor.v1", "actor": actor.state_dict(), "actor_version": 0,
                 "frozen_sha256": adapter.bundle.metadata["sha256"]}, weights)
    workers, rows = [], []
    try:
        for index, gpu in enumerate(args.gpus, 1):
            workers.append(RemotePhysicalCollector(cfg, config, output/f"worker_{index}", gpu, index))
        for worker in workers:
            worker.wait_ready(adapter.bundle.metadata["sha256"])
        tasks = read_tasks(cfg["tasks"])
        for round_index in range(2):
            for index, worker in enumerate(workers):
                worker.submit(weights, 0, 3, tasks[index] if worker.needs_task else None)
            responses = [worker.receive(adapter, "cpu") for worker in workers]
            for response in responses:
                if response["actor_version"] != 0 or not response["frozen_unchanged"]:
                    raise RuntimeError("采集版本或冻结合同失败")
                for sample in response["anchors"]:
                    sample.validate(cfg["imagination"]["horizon_steps"])
                    if sample.collector_id != response["collector_id"]:
                        raise RuntimeError("不同机器人的anchor标识混淆")
            overlap = min(x["ended_monotonic"] for x in responses)-max(x["started_monotonic"] for x in responses)
            row = {"round": round_index, "parallel_overlap_seconds": overlap,
                   "controls_per_worker": [r["physical_steps"] for r in responses],
                   "anchors_per_worker": [len(r["anchors"]) for r in responses],
                   "worker_ticks": [r["tick"] for r in responses],
                   "frozen_unchanged": all(r["frozen_unchanged"] for r in responses)}
            rows.append(row)
            print(json.dumps(row), flush=True)
        passed = all(r["parallel_overlap_seconds"] > 0 and min(r["controls_per_worker"]) > 0 for r in rows) and all(
            sum(r["anchors_per_worker"][i] for r in rows) > 0 for i in range(2))
        result = {"passed": passed, "rows": rows, "accepted_actor_updates": 0,
                  "evidence": "real_parallel_newton_collection_not_training_gain"}
        (output/"parallel_collection_audit.json").write_text(json.dumps(result, indent=2))
        if not passed:
            raise RuntimeError("真实两路采集验收失败")
    finally:
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    main()
