"""Server-2新学习信号的有限真实闭环启动及完整退出回执。

只接受独立signals.v2配置、最多64次任务尝试和一个宏迭代；教师约束与保留门槛必须
关闭。运行完整Episode、双Critic、自蒸馏诊断和AWR，不启动旧五轮作业、不自动补采。
stdout及来源、配置、预算、PID、真实退出码写入全新运行旁路文件，失败证据也保留。
更大训练预算应通过常规入口显式指定，这个验收启动器不隐式扩容。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from gem.closedloop.awr.config import load_config, git
    from gem.closedloop.awr.io import atomic_json, digest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gmt-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/closedloop/stage2_awr_rewardstd_nobc.yaml")
    parser.add_argument("--finish-before-utc", help="本次墙钟边界；提前10分钟请求排空，不恢复旧任务")
    args = parser.parse_args()
    deadline = (datetime.fromisoformat(args.finish_before_utc.replace("Z", "+00:00")).timestamp()
                if args.finish_before_utc else None)
    if deadline is not None and deadline-time.time() < 20*60:
        parser.error("A bounded physical run requires at least twenty minutes")
    config = load_config(args.config)
    if (config["version"] != "genmo.stage2.awr.signals.v2" or config["awr"]["max_macro_iterations"] != 1
            or config["awr"]["initial_episodes"] > 64 or config["awr"].get("collection_completion")):
        parser.error("Use one finite signals.v2 macro with <=64 planned attempts and no supplementation")
    config["paths"].update(genmo_repo=str(ROOT), gmt_repo=str(args.gmt_repo.resolve()))
    root = args.run_dir.resolve()
    if root.parent != Path("/data1/user/liwei/GENMO_outputs/stage2_awr") or root.exists():
        parser.error("Use a new run under the Server-2 AWR output directory")
    sources = {}
    for name in ("genmo", "gmt"):
        repo = config["paths"][name+"_repo"]
        if git(repo, "branch", "--show-current") != "feature/stage2-awr" or git(repo, "status", "--porcelain"):
            parser.error("Both independent sources must be clean feature/stage2-awr")
        sources[name] = git(repo, "rev-parse", "HEAD")
    paths = {suffix: root.with_name(root.name+"."+suffix) for suffix in ("log", "launch.json", "exit.json")}
    if any(p.exists() for p in paths.values()):
        parser.error("Never overwrite an earlier launch")
    command = ["bash", str(ROOT/"tools/run_stage2_awr_server2.sh"), "run", "--config", str(args.config.resolve()),
               "--run-dir", str(root), "--gmt-repo", str(args.gmt_repo.resolve())]
    started = time.time()
    with paths["log"].open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        atomic_json(paths["launch.json"], dict(pid=process.pid, parent_pid=os.getpid(), started_unix=started,
            command=command, source_commits=sources, config_sha256=digest(config),
            budget=dict(planned_attempts=config["awr"]["initial_episodes"], macro_iterations=1,
                        actor_epochs=config["awr"]["actor"]["epochs"], supplemental_attempts=0)))
        requested = False
        while True:
            try:
                code = process.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                if deadline is not None and time.time() >= deadline-600 and not requested and root.exists():
                    atomic_json(root/"STOP", dict(reason="authorized_two_hour_boundary_drain", deadline_unix=deadline))
                    requested = True
    atomic_json(paths["exit.json"], dict(exit_code=code, pid=process.pid, finished_unix=time.time(),
        seconds=time.time()-started, source_commits=sources))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
