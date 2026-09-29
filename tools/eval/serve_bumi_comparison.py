#!/usr/bin/env python3
"""在本机或局域网提供BUMI对比报告与支持拖动定位的MP4字节范围服务。

标准库SimpleHTTPRequestHandler在当前Python版本不处理Range，长视频拖动可能重复
读取整文件。本工具只为已存在的本地文件增加单区间bytes请求、206/416响应和有界流式
传输；普通页面和目录继续复用标准处理器。默认仅绑定127.0.0.1；需要局域网分享时，
可用--bind指定本机IPv4地址，或指定0.0.0.0同时接受本机和局域网访问。服务只读取
指定报告目录及其已有链接资源，不上传数据、不修改网页、视频或模型。服务器可由
systemd用户服务托管，退出Codex后继续提供报告；局域网访问链接须使用本机实际IP。
"""

from __future__ import annotations

import argparse
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class VideoRangeHandler(SimpleHTTPRequestHandler):
    """以MP4播放器需要的单字节区间读取文件，HEAD与GET共享正确长度信息。"""

    def send_head(self):
        self.remaining = None
        value = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not value or not path.is_file():
            return super().send_head()
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
        size = path.stat().st_size
        if match is None or not any(match.groups()):
            self.send_error(400, "Invalid byte range")
            return None
        first, last = match.groups()
        if first:
            start = int(first)
            end = min(int(last), size - 1) if last else size - 1
        else:
            count = int(last)
            start, end = max(0, size - count), size - 1
        if size == 0 or start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        stream = path.open("rb")
        stream.seek(start)
        self.remaining = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(self.remaining))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        return stream

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def copyfile(self, source, outputfile):
        try:
            if self.remaining is None:
                return super().copyfile(source, outputfile)
            while self.remaining > 0:
                chunk = source.read(min(self.remaining, 1024 * 1024))
                if not chunk:
                    break
                outputfile.write(chunk)
                self.remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # 用户切换样本或拖动进度时，浏览器会正常取消旧媒体请求。
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--port", type=int, default=38927)
    parser.add_argument(
        "--bind", default="127.0.0.1", help="监听IPv4地址；局域网分享可设为0.0.0.0"
    )
    args = parser.parse_args()
    root = args.directory.resolve(strict=True)
    server = ThreadingHTTPServer(
        (args.bind, args.port), partial(VideoRangeHandler, directory=str(root))
    )
    host = "127.0.0.1" if args.bind == "0.0.0.0" else args.bind
    print(f"BUMI对比报告：http://{host}:{server.server_port}/index.html", flush=True)
    if args.bind == "0.0.0.0":
        print(f"局域网访问：将链接地址换成本机局域网IP，端口保持{server.server_port}。", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
