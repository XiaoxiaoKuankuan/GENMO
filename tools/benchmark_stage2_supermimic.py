"""实际单GPU与多GPU独立闭环采集吞吐基准，使用固定真实Stage1/SM资产。

每个worker是独立Newton物理进程，有独立音乐、缓存、生成请求和种子；比较一张卡与
多张空闲卡完成同一任务集合的墙钟和有效参考接管。该工具不优化Actor，不把并行
采集基准宣称为已实现分布式训练；部署延迟仍取各机器人异步请求，不能用批墙钟替代。
测试输出独占目录，已有进程占用指定GPU时拒绝运行，不停止其他任务。
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=3.)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    cfg = OmegaConf.load(args.config)
    tasks = json.loads(Path(cfg.tasks).read_text())
    gpus = [int(x) for x in args.gpus.split(",")]
    if len(gpus)<2 or len(set(gpus)) != len(gpus) or len(tasks)<len(gpus):
        raise ValueError("需要至少两张独立空闲GPU及相应数量真实音乐")
    inventory = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
    memory = {int(row.split(",")[0]):int(row.split(",")[1]) for row in inventory.splitlines()}
    if any(memory[g] >= 1024 for g in gpus):
        raise RuntimeError("基准GPU已有任务，不抢占")
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    configs=[]
    for index,task in enumerate(tasks[:len(gpus)]):
        task=dict(task,seeds=[9400+index])
        task_path=output/f"task_{index}.json"
        task_path.write_text(json.dumps([task],ensure_ascii=False))
        worker=OmegaConf.merge(cfg,{"tasks":str(task_path)})
        path=output/f"worker_{index}.yaml"
        OmegaConf.save(worker,path)
        configs.append(path)
    def launch(index,gpu,regime):
        target=output/f"{regime}_{index}"
        command=[sys.executable,"-u","tools/evaluate_stage2_supermimic.py","--mode","calibration", "--config",str(configs[index]),
                 "--max-seconds",str(args.seconds),"--output",str(target)]
        log=(output/f"{regime}_{index}.log").open("w")
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
        return subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT), log, target
    results={}
    for regime in ("single_gpu_sequential","multi_gpu_independent"):
        start=time.perf_counter()
        handles=[]
        if regime=="single_gpu_sequential":
            for i in range(len(configs)):
                proc,log,target=launch(i,gpus[0],regime)
                code=proc.wait()
                log.close()
                if code:
                    raise RuntimeError(f"{regime}/{i}退出{code}；原始日志已保留")
                handles.append((proc,log,target))
        else:
            handles=[launch(i,g,regime) for i,g in enumerate(gpus)]
            codes=[p.wait() for p,_,_ in handles]
            for _,log,_ in handles:
                log.close()
            if any(codes):
                raise RuntimeError(f"{regime}退出码{codes}")
        rows=[]
        for _,_,target in handles:
            rows.extend(json.loads((target/"physical_summary.json").read_text())["physical_results"])
        wall=time.perf_counter()-start
        results[regime]={"wall_seconds_including_startup":wall,"physical_control_steps":sum(round(r["physical_seconds"]*50) for r in rows),
                         "accepted_commits":sum(r["commits"] for r in rows),"rows":rows,
                         "accepted_commits_per_wall_second":sum(r["commits"] for r in rows)/wall}
        (output/"benchmark.json").write_text(json.dumps(results,ensure_ascii=False,indent=2))
        print(json.dumps({"event":regime,"wall_seconds":wall,"accepted_commits":results[regime]["accepted_commits"]}),flush=True)


if __name__ == "__main__":
    main()
