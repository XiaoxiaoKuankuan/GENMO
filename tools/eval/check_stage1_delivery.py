#!/usr/bin/env python3
"""验收BUMI本地网页交付的真实媒体、相对链接和HTTP范围播放。

逐个读取正式/补充样本报告，以ffprobe核对H.264、AAC、30Hz、帧数和视频真实时长；
检查每个视频及报告HTTP可达，并实际验证首段、尾段和越界Range响应。对HTML脚本使用
系统JavaScript引擎做语法验证，不替代浏览器视觉质量判断。结果保存到用户要求保留的验收JSON，
脚本语法临时文件由TemporaryDirectory自动回收，不修改模型、视频和训练数据。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    files, videos, results = [], [], []
    for suffix in ("", "current_mine_reference/"):
        local = root / suffix
        records = json.loads((local / "results.json").read_text())
        for name in (
            "index.html",
            "training_analysis.json",
            "training_curves.png",
            "selection.json",
            "summary.json",
            "onnx_parity.json",
            "protocol.json",
            "checkpoint_capture.json",
        ):
            files.append(suffix + name)
        for row in records:
            stem = suffix + f"samples/{row['number']:02d}/"
            files.append(stem + "report.json")
            for kind in ("original", "generated"):
                path = stem + kind + ".mp4"
                files.append(path)
                videos.append((path, row["rollout"]["num_frames"]))
        script = re.search(
            r"<script>(.*?)</script>", (local / "index.html").read_text(), re.S
        ).group(1)
        with tempfile.TemporaryDirectory(prefix="bumi_site_syntax_") as folder:
            path = Path(folder) / "page.js"
            path.write_text(script)
            if shutil.which("node"):
                command = ["node", "--check", str(path)]
            elif shutil.which("gjs"):
                command = [
                    "gjs",
                    "-c",
                    "const bytes=imports.gi.GLib.file_get_contents(ARGV[0])[1]; "
                    "new Function(imports.byteArray.toString(bytes));",
                    str(path),
                ]
            else:
                raise RuntimeError("需要node或gjs执行真实JavaScript语法校验")
            subprocess.run(command, check=True, capture_output=True)

    def check_video(item):
        path, frames = item
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_format",
                    "-of",
                    "json",
                    str(root / path),
                ]
            )
        )
        video = next(s for s in probe["streams"] if s["codec_type"] == "video")
        audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
        assert video["codec_name"] == "h264" and audio["codec_name"] == "aac", path
        assert int(video["nb_frames"]) == frames, (path, video["nb_frames"], frames)
        assert video["r_frame_rate"] == "30/1", path
        assert abs(float(video["duration"]) - frames / 30) < 0.001, path
        assert (root / path).stat().st_size > 1024, path
        return {
            "path": path,
            "frames": frames,
            "video_seconds": float(video["duration"]),
            "width": video["width"],
            "height": video["height"],
            "video_codec": "h264",
            "audio_codec": "aac",
        }

    def check_http(relative):
        request = urllib.request.Request(args.url.rstrip("/") + "/" + relative, method="HEAD")
        with opener.open(request, timeout=10) as response:
            assert response.status == 200 and int(response.headers["Content-Length"]) > 0, relative
        return relative

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(check_video, videos))
        links = list(pool.map(check_http, files))
    video_path = videos[0][0]
    url = args.url.rstrip("/") + "/" + video_path
    for requested, expected in (
        ("bytes=0-31", (root / video_path).read_bytes()[:32]),
        ("bytes=-32", (root / video_path).read_bytes()[-32:]),
    ):
        with opener.open(urllib.request.Request(url, headers={"Range": requested})) as response:
            assert response.status == 206 and response.read() == expected
    invalid = str((root / video_path).stat().st_size + 100)
    try:
        opener.open(urllib.request.Request(url, headers={"Range": f"bytes={invalid}-"}))
        raise AssertionError("越界Range应该返回416")
    except urllib.error.HTTPError as error:
        assert error.code == 416
    leftovers = [
        str(p)
        for p in root.rglob("*")
        if p.is_file() and any(x in p.name for x in (".silent.", ".mux.", ".tmp", ".part"))
    ]
    assert not leftovers, leftovers
    report = {
        "status": "passed",
        "videos_checked": len(results),
        "http_links_checked": len(links),
        "javascript_syntax": "passed",
        "range_checks": ["prefix206", "suffix206", "out_of_bounds416"],
        "all_video_frames_match_source": True,
        "temporary_media_leftovers": leftovers,
        "videos": results,
    }
    (root / "delivery_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "videos"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
