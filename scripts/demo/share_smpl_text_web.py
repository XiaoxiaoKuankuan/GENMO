"""为已运行的 GENMO 文本动作网页提供匿名公网分享入口。

此脚本用 Waitress 在本机回环端口启动轻量网关，复用 8766 服务及其单一 GPU worker。
通过 --public-origin 指定隧道或自有域名的公开地址，访客可选已注册模型、生成动作、
查看共享历史和播放视频；本地模型路径注册仍只在 8766 管理入口提供。网关本身不
创建模型、训练任务或历史副本，关闭它不会停止本机推理服务。可用 Cloudflare Tunnel
将 --port 对应的回环端口映射到公网 HTTPS；域名、启动和停止方式见 docs/text_motion_web.md。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:8766")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--public-origin", required=True)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port 必须为 1–65535")
    from waitress import serve

    from gem.runtime.text_motion_web.share import create_share_app

    app = create_share_app(args.upstream, args.public_origin)
    print(f"公开文本动作网页：{args.public_origin}", flush=True)
    print(f"公网入口监听 127.0.0.1:{args.port}，复用 {args.upstream}", flush=True)
    serve(app, host="127.0.0.1", port=args.port, threads=12, max_request_body_size=16 * 1024)


if __name__ == "__main__":
    main()
