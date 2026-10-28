"""将联合训练的追加式JSONL指标持续转换为TensorBoard，并可启动本机网页服务。

数据流：只读指定run/metrics.jsonl，展开有限数值指标，向独立tensorboard目录写事件；
不加载GENMO、SM或WM，不占用训练GPU，不改变训练配置、采样或checkpoint。
完整JSON行才会消费，写入一半的末行等待下一次轮询。输出目录持有独占锁，重启时从
已持久化的joint/global_step标记去重；源日志截断或完整行损坏时明确退出，不篡改原文件。
默认启动仅监听127.0.0.1的TensorBoard，远程访问使用SSH转发；--once只导出当前日志，
用于离线检查。退出时只清理本工具自己创建的网页子进程，不影响其它TensorBoard或训练。
"""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def scalars(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from scalars(child, f"{prefix}/{key}" if prefix else str(key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        yield prefix, float(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--port", type=int, default=6013)
    parser.add_argument("--poll-seconds", type=float, default=5.)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.poll_seconds <= 0:
        parser.error("端口或轮询间隔非法")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    run = args.run.expanduser().resolve(strict=True)
    source = run / "metrics.jsonl"
    if not source.is_file():
        parser.error(f"不存在训练指标文件：{source}")
    output = run / "tensorboard"
    output.mkdir(exist_ok=True)
    server = writer = None
    with (output / ".writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = EventAccumulator(str(output), size_guidance={"scalars": 0}).Reload()
        marks = previous.Scalars("joint/global_step") if "joint/global_step" in previous.Tags()["scalars"] else []
        last_step = max((item.step for item in marks), default=-1)
        writer = EventFileWriter(str(output))
        try:
            if not args.once:
                server = subprocess.Popen([sys.executable, "-m", "tensorboard.main", "--logdir", str(output),
                    "--host", "127.0.0.1", "--port", str(args.port)])
            with source.open("rb") as stream:
                while True:
                    count = 0
                    if source.stat().st_size < stream.tell():
                        raise RuntimeError("原指标日志发生截断，停止转换并保留已有事件")
                    while True:
                        offset = stream.tell()
                        line = stream.readline()
                        if not line or not line.endswith(b"\n"):
                            stream.seek(offset)
                            break
                        row = json.loads(line)
                        step = int(row["step"])
                        if step <= last_step:
                            continue
                        values = [Summary.Value(tag=name, simple_value=value)
                                  for name, value in scalars(row) if name != "step"]
                        values.append(Summary.Value(tag="joint/global_step", simple_value=step))
                        writer.add_event(Event(wall_time=time.time(), step=step, summary=Summary(value=values)))
                        last_step = step
                        count += 1
                    writer.flush()
                    if count or args.once:
                        print(json.dumps({"exported_rows": count, "latest_logged_step": last_step,
                                          "logdir": str(output)}, ensure_ascii=False), flush=True)
                    if args.once:
                        break
                    if server.poll() is not None:
                        raise RuntimeError(f"TensorBoard服务已退出：{server.returncode}")
                    time.sleep(args.poll_seconds)
        finally:
            if writer is not None:
                writer.close()
            if server is not None and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("TensorBoard子进程尚未退出，PID=" + str(server.pid), file=sys.stderr)


if __name__ == "__main__":
    main()
