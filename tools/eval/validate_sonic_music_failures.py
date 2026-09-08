#!/usr/bin/env python3
"""音乐部署的三进程故障注入与用户停止验收工具。

该工具只启动本机仿真会话，默认关闭音频和可视窗口。心跳用例暂停本次启动的
GENMO 进程，检查 SONIC 与 MuJoCo 自行锁存故障且物理时间停止，再恢复协调器
让其保存报告。延迟用例只在本次子进程内延缓第三个生成窗口，不修改生产代码。
停止用例通过统一入口 --stop-after 执行与 Ctrl+C 相同的淡出和站立收尾。
所有输出保存在调用者指定的新目录；不会查找或终止其他用户进程。
"""

from __future__ import annotations

# 直接运行脚本时先建立仓库导入路径。
# ruff: noqa: E402
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gem.runtime.sonic_music import SessionClient


def validate(args):
    """顺序运行三个用例，输出实际故障状态和冻结前后的物理时间。"""
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for case in ("heartbeat", "generation_delay", "user_stop"):
        destination = args.output_dir / case
        common = [
            "--audio",
            str(args.audio.resolve()),
            "--duration-sec",
            "8",
            "--launch-local",
            "--headless",
            "--audio-output",
            "off",
            "--exit-on-finish",
            "--output-dir",
            str(destination),
        ]
        command = [sys.executable, "-B", str(ROOT / "scripts/demo/demo_music_sonic.py"), *common]
        if case != "generation_delay":
            command += ["--smpl-npz", str(args.smpl_npz.resolve())]
        if case == "user_stop":
            command += ["--stop-after", "2"]
        if case == "generation_delay":
            # 第一次调用用于预热，第二、三次预缓冲；第四次是播放中的第三窗。
            code = """import time
from scripts.demo.demo_music_sonic import parser, run, SlidingDDIMGenerator
original=SlidingDDIMGenerator.generate_window
count=0
def delayed(self,*args,**kwargs):
    global count
    count+=1
    if count==4: time.sleep(6)
    return original(self,*args,**kwargs)
SlidingDDIMGenerator.generate_window=delayed
raise SystemExit(run(parser().parse_args()))
"""
            command = [sys.executable, "-B", "-c", code, *common]
        row = dict(case=case, command=command)
        with (args.output_dir / f"{case}.log").open("x") as log:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                start_new_session=True,
            )
            paused = False
            try:
                if case == "heartbeat":
                    sonic, sim = (
                        SessionClient("tcp://127.0.0.1:5560", timeout_ms=100),
                        SessionClient("tcp://127.0.0.1:5561", timeout_ms=100),
                    )
                    try:
                        deadline = time.monotonic() + 120
                        while True:
                            if child.poll() is not None:
                                raise RuntimeError("注入之前协调器已经退出")
                            try:
                                status = sonic.call("status")
                                if status["state"] == "playing" and status["used_frame"] >= 110:
                                    break
                            except TimeoutError:
                                pass
                            if time.monotonic() > deadline:
                                raise TimeoutError("等待故障注入时刻超时")
                            time.sleep(0.05)
                        os.kill(child.pid, signal.SIGSTOP)
                        paused = True
                        time.sleep(1.9)
                        a, b = sonic.call("status"), sim.call("status")
                        time.sleep(0.12)
                        c = sim.call("status")
                        row.update(
                            sonic_fault=a,
                            sim_fault=b,
                            sim_time_after=c["sim_time"],
                            frozen=a["state"] == b["state"] == "fault"
                            and b["sim_time"] == c["sim_time"],
                        )
                    finally:
                        if paused:
                            os.kill(child.pid, signal.SIGCONT)
                            paused = False
                        sonic.close()
                        sim.close()
                child.wait(timeout=120)
                report = json.loads((destination / "report.json").read_text())
                row.update(returncode=child.returncode, report=report)
                if case == "heartbeat":
                    row["passed"] = (
                        row["frozen"] and child.returncode == 1 and "心跳" in report["error"]
                    )
                elif case == "generation_delay":
                    row["passed"] = child.returncode == 1 and "未来参考缓冲不足" in report["error"]
                else:
                    row["passed"] = (
                        child.returncode == 0 and report["user_stopped"] and not report["error"]
                    )
            finally:
                if paused:
                    os.kill(child.pid, signal.SIGCONT)
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                    child.wait(timeout=15)
        results.append(row)
        (args.output_dir / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n"
        )
        print(f"case={case} passed={row['passed']}", flush=True)
    return all(r["passed"] for r in results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--smpl-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    raise SystemExit(0 if validate(parser.parse_args()) else 1)
