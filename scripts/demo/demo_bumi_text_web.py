"""GENMO 本地文本动作工作台启动入口。

此入口在仓库根目录启动仅监听回环地址的 Flask 服务。网页提供文本模型选择、
动作生成参数、手动视频播放和本地历史；实际 GPU 推理由独立常驻进程执行，
不会启动远端训练。默认读取本机 T5 缓存，所有正式结果保存在 outputs 下。
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--output_root", type=Path, default=ROOT / "outputs/text_motion_web")
    args = parser.parse_args()
    os.chdir(ROOT)
    from gem.runtime.text_motion_web.app import create_app
    from gem.runtime.text_motion_web.service import JobService

    service = JobService(args.output_root)
    app = create_app(service)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        print(f"GENMO 本地工作台：http://127.0.0.1:{args.port}", flush=True)
        print(f"生成结果：{service.root}", flush=True)
        app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False)
    finally:
        service.close()


if __name__ == "__main__":
    main()
