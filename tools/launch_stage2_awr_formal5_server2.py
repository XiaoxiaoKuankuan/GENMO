"""Server-2正式五轮AWR的有限预算启动与退出回执。

读取已提交的formal5配置，确认512/128/512数据规模、最多五轮与全运行26次补采上限，
然后调用原八卡启动器；训练、物理检查和正常停止语义仍由原入口负责。每次新建独立
运行目录，stdout、启动PID/源码/配置摘要和真实退出码写在目录旁，失败也完整留存。
本脚本不自动重试未知异常、不停止其他作业、不更改源码或旧运行身份。它是前台父进程，
需要在持久终端或明确的新会话运行；停止训练使用原stop命令，让Episode与优化步排空。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_formal_budget(config):
    """拒绝把本次五轮授权隐式扩大；默认单轮配置不受影响。"""
    awr = config["awr"]
    expected = dict(initial_episodes=512, new_episodes=128, replay_capacity=512,
                    max_macro_iterations=5, run_ablation=True, final_all_unique=True)
    if any(awr.get(k) != v for k, v in expected.items()) or awr["profile"] != "pilot":
        raise ValueError("Formal launch requires the approved 512/128/512 five-macro comparison")
    completion = awr.get("collection_completion", {})
    if completion.get("max_extra_attempts_total") != 26 or not completion.get("authorization", "").strip():
        raise ValueError("Formal run requires explicit 26 attempts shared across all five macros")
    return dict(max_macro_iterations=5, target_complete_episodes=1024,
                max_extra_attempts_total=26, max_training_collection_attempts=1050,
                evaluation_tasks_are_separate=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/closedloop/stage2_awr_formal5.yaml")
    args = parser.parse_args()
    from gem.closedloop.awr.config import load_config, git
    from gem.closedloop.awr.io import atomic_json, digest, file_sha256
    config = load_config(args.config)
    config["paths"]["genmo_repo"] = str(ROOT)
    budget = validate_formal_budget(config)
    root = args.run_dir.resolve()
    approved = Path("/data1/user/liwei/GENMO_outputs/stage2_awr").resolve()
    if root.parent != approved or root.exists():
        parser.error("Use a new run directly under the approved Server-2 AWR output directory")
    sources = {}
    for name in ("genmo", "gmt"):
        repo = config["paths"][name+"_repo"]
        if git(repo, "branch", "--show-current") != "feature/stage2-awr" or git(repo, "status", "--porcelain"):
            parser.error("Both AWR source directories must be on the reviewed clean branch")
        sources[name] = git(repo, "rev-parse", "HEAD")
    launch = root.with_name(root.name+".launch.json")
    ending = root.with_name(root.name+".exit.json")
    log = root.with_name(root.name+".log")
    if any(p.exists() for p in (launch, ending, log)):
        parser.error("Launch evidence already exists; never overwrite an earlier attempt")
    command = ["bash", str(ROOT/"tools/run_stage2_awr_server2.sh"), "run",
               "--config", str(args.config.resolve()), "--run-dir", str(root)]
    started = time.time()
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        receipt = dict(pid=process.pid, parent_pid=os.getpid(), started_unix=started,
                       command=command, source_commits=sources, config_sha256=digest(config),
                       config_file_sha256=file_sha256(args.config), budget=budget,
                       authorization=config["awr"]["collection_completion"]["authorization"])
        atomic_json(launch, receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        code = process.wait()
    exit_receipt = dict(pid=process.pid, parent_pid=os.getpid(), exit_code=code,
                        finished_unix=time.time(), source_commits=sources)
    atomic_json(ending, exit_receipt)
    print(json.dumps(exit_receipt, ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
