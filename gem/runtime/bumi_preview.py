"""BUMI 运动学预览的资源校验、发布快照和非阻塞查看器客户端。

渲染资源具有独立的完整文件指纹，与训练时导出的 kinematics/MJCF 绑定；模型资产
清单保持原格式。PoseSnapshot 保存已经发布的原生 qpos，而不是下一播放游标。
查看器由当前 Python 启动，使用匿名管道接收小于 PIPE_BUF 的 JSON 帧。慢窗口时丢弃
显示帧，绝不等待窗口，也不会影响生成/控制。GMT 轮询使用自己的 ZMQ 连接和线程；
本地播放器直接返回快照，不使用网络。此模块不导入 MuJoCo 或创建图形上下文。
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np


def validate_robot_assets(manifest_path, kinematics_path):
    manifest = Path(manifest_path).resolve(strict=True)
    root = manifest.parent
    payload = json.loads(manifest.read_text())
    spec_path = Path(kinematics_path).resolve(strict=True)
    spec = json.loads(spec_path.read_text())
    if payload.get("contract_version") != "genmo.bumi_viewer.v1":
        raise ValueError("不支持的 MuJoCo 资源清单")
    if payload.get("kinematics_sha256") != hashlib.sha256(spec_path.read_bytes()).hexdigest():
        raise ValueError("查看器资源与当前 kinematics 不匹配")
    if (spec.get("qpos_dim"), spec.get("joint_dim"), spec.get("quaternion_convention")) != (
        28,
        21,
        "wxyz",
    ):
        raise ValueError("查看器需要 qpos28/21 关节/wxyz 契约")
    files = payload["files"]
    for name, record in files.items():
        relative = Path(name)
        actual = (root / relative).resolve(strict=True)
        if relative.is_absolute() or ".." in relative.parts or not actual.is_relative_to(root):
            raise ValueError("查看器资源路径越界")
        if (
            actual.stat().st_size != record["size_bytes"]
            or hashlib.sha256(actual.read_bytes()).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"查看器资源指纹不匹配: {name}")
    mjcf = payload["mjcf"]
    if mjcf not in files or files[mjcf]["sha256"] != spec["source_mjcf_sha256"]:
        raise ValueError("查看器 XML 不是当前运动学的源 MJCF")
    # 所有外部 mesh 必须属于已校验集合，不能让 XML 引用未携带或未校验的文件。
    import xml.etree.ElementTree as ET

    xml = ET.parse(root / mjcf).getroot()
    if xml.findall(".//include"):
        raise ValueError("查看器清单要求自包含 MJCF，不允许未展开的 include")
    compiler = xml.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    for element in xml.iter():
        if "file" not in element.attrib:
            continue
        directory = meshdir if element.tag == "mesh" else ""
        asset = ((root / mjcf).parent / directory / element.get("file")).resolve()
        if not asset.is_relative_to(root) or asset.relative_to(root).as_posix() not in files:
            raise ValueError("XML 引用了未校验的资源")
    return root / mjcf, spec


class PoseSnapshot:
    """线程安全、只保存最后一次成功发布姿态；读取不会推进任何播放状态。"""

    def __init__(self, kinematics_sha256):
        self.lock = threading.Lock()
        self.kinematics_sha256 = kinematics_sha256
        self.sequence = 0
        self.frame = None

    def record(self, qpos, *, frame_index, revision, request_id, state, now=None, packet=None):
        values = np.asarray(qpos, dtype=np.float32)
        if (
            values.shape != (28,)
            or not np.isfinite(values).all()
            or abs(np.linalg.norm(values[3:7]) - 1) > 1e-3
        ):
            raise ValueError("预览姿态必须为有限 qpos28，四元数为单位 wxyz")
        with self.lock:
            self.sequence += 1
            self.frame = dict(
                qpos=values.tolist(),
                frame_index=int(frame_index),
                revision=int(revision),
                request_id=request_id,
                state=state,
                sequence=self.sequence,
                published_monotonic=time.monotonic() if now is None else now,
                kinematics_sha256=self.kinematics_sha256,
            )
            self.frame.update(
                stream_id=getattr(packet, "stream_id", None),
                packet_sequence=getattr(packet, "sequence", None),
            )

    def read(self, now=None):
        with self.lock:
            if self.frame is None:
                return {"ok": True, "available": False}
            result = dict(self.frame)
            result["qpos"] = list(result["qpos"])
        result.update(
            ok=True,
            available=True,
            age_seconds=max(
                0.0, (time.monotonic() if now is None else now) - result["published_monotonic"]
            ),
        )
        return result


class MujocoPreview:
    """查看器和只读姿态源的生命周期管理；退出窗口不向控制器发送命令。"""

    def __init__(self, robot_manifest, kinematics_path, source_factory):
        validate_robot_assets(robot_manifest, kinematics_path)
        self.kinematics_sha256 = hashlib.sha256(Path(kinematics_path).read_bytes()).hexdigest()
        if not os.environ.get("DISPLAY"):
            raise RuntimeError("MuJoCo 预览需要桌面 DISPLAY；请在图形桌面终端启动")
        self.stop = threading.Event()
        self.last_error = None
        self.dropped_frames = 0
        self.sent_frames = 0
        self.source_factory = source_factory
        root = Path(__file__).resolve().parents[2]
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-u",
                str(root / "scripts/demo/bumi_mujoco_viewer.py"),
                "--robot-manifest",
                str(robot_manifest),
                "--kinematics",
                str(kinematics_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
            start_new_session=True,
        )
        try:
            ready, _, _ = select.select([self.process.stdout], [], [], 15.0)
            if not ready or self.process.stdout.read(1) != b"V":
                raise RuntimeError("MuJoCo 窗口启动失败，检查上方图形或资源错误")
            os.set_blocking(self.process.stdin.fileno(), False)
        except BaseException:
            self.close()
            raise
        self.thread = threading.Thread(target=self._loop, name="bumi-preview-reader", daemon=True)
        self.thread.start()

    def _loop(self):
        source = None
        last_sequence = -1
        stale = False
        try:
            source = self.source_factory()
            while not self.stop.is_set():
                if self.process.poll() is not None:
                    if self.process.returncode != 0:
                        self.last_error = f"查看器退出码 {self.process.returncode}"
                    print(
                        "[MuJoCo] 窗口已关闭；控制台与播放继续，输入 stand 可停止动作", flush=True
                    )
                    break
                started = time.monotonic()
                try:
                    frame = source.request({"command": "preview_frame"})
                    if not frame.get("ok"):
                        raise RuntimeError(
                            frame.get("error", "Bridge 缺少 preview_frame，请更新部署代码")
                        )
                    fresh = frame.get("available") and frame["age_seconds"] <= 0.2
                    if fresh and frame["kinematics_sha256"] != self.kinematics_sha256:
                        raise ValueError("参考姿态与查看器的运动学指纹不匹配")
                    if fresh and frame["sequence"] != last_sequence:
                        frame["observed_monotonic"] = time.monotonic() - frame["age_seconds"]
                        payload = (json.dumps(frame, separators=(",", ":")) + "\n").encode()
                        if len(payload) > os.fpathconf(self.process.stdin.fileno(), "PC_PIPE_BUF"):
                            raise ValueError("预览消息超过管道原子写入上限")
                        try:
                            os.write(self.process.stdin.fileno(), payload)
                            self.sent_frames += 1
                            last_sequence = frame["sequence"]
                        except BlockingIOError:
                            self.dropped_frames += 1
                    if stale and fresh:
                        print("[MuJoCo] 参考姿态已恢复", flush=True)
                    stale = not fresh
                except (ValueError, RuntimeError) as exc:
                    self.last_error = str(exc)
                    print(f"[MuJoCo ERROR] {exc}；仅停止预览", flush=True)
                    break
                except Exception as exc:
                    if not stale:
                        print(f"[MuJoCo] 参考连接中断，冻结画面：{exc}", flush=True)
                    stale = True
                self.stop.wait(max(0.0, 0.02 - (time.monotonic() - started)))
        finally:
            if source is not None:
                source.close()

    def status(self):
        return {
            "alive": self.process.poll() is None,
            "sent_frames": self.sent_frames,
            "dropped_frames": self.dropped_frames,
            "last_error": self.last_error,
        }

    def close(self):
        self.stop.set()
        thread = getattr(self, "thread", None)
        if thread is not None:
            thread.join(timeout=1.0)
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()
        if self.process.returncode != 0:
            self.last_error = f"查看器退出码 {self.process.returncode}"


class LocalPreviewReader:
    """本地姿态只读适配器，关闭查看器不能关闭播放引擎。"""

    def __init__(self, player):
        self.player = player

    def request(self, payload):
        return self.player.request(payload)

    def close(self):
        pass
