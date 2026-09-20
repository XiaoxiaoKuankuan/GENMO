"""BUMI 文本动作的无控制器预览与离线视频渲染。

复用部署分支的资源清单和passive viewer，当前模块只按单调时钟提供最新qpos快照。
无Redis/ZMQ/ROS、无音频、无Bridge站姿安全门；坐卧/跳跃均原样显示。视频与桌面
窗口都只写qpos并执行mj_forward，不调用mj_step，不作为动力学或实机跟踪验收。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from gem.runtime.bumi_preview import (
    MujocoPreview,
    LocalPreviewReader,
    PoseSnapshot,
    validate_robot_assets,
)


def check_bumi_motion(path, frames):
    with np.load(path, allow_pickle=False) as motion:
        for name, width in [("qpos", 28), ("qpos_raw", 28), ("foot_contact_logits", 2)]:
            value = motion[name]
            if value.shape != (frames, width) or not np.isfinite(value).all():
                raise ValueError(f"BUMI产物{name}形状/有限性错误")
        if float(motion["fps"]) != 30 or str(motion["quaternion_convention"]) != "wxyz":
            raise ValueError("BUMI产物需要30FPS/wxyz")
        if len(motion["joint_names"]) != 21 or len(set(motion["joint_names"].tolist())) != 21:
            raise ValueError("BUMI关节顺序错误")
        if not np.allclose(np.linalg.norm(motion["qpos"][:, 3:7], axis=1), 1, atol=1e-3):
            raise ValueError("非单位根四元数")


def render_bumi_video(output, frames):
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import av

    output = Path(output)
    check_bumi_motion(output / "motion.npz", frames)
    metadata = json.loads((output / "metadata.json").read_text())
    xml, spec = validate_robot_assets(metadata["robot_manifest"], metadata["kinematics_path"])
    with np.load(output / "motion.npz", allow_pickle=False) as payload:
        qpos = payload["qpos"].copy()
        if payload["joint_names"].tolist() != spec["joint_order"]:
            raise ValueError("视频资产与动作关节顺序不匹配")
    model = mujoco.MjModel.from_xml_path(str(xml))
    if model.nq != 28:
        raise ValueError("MJCF qpos不是28维")
    for i, name in enumerate(spec["joint_order"]):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0 or model.jnt_qposadr[jid] != i + 7:
            raise ValueError("MJCF关节排列与qpos不符")
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 2.6
    camera.azimuth = 130
    camera.elevation = -15
    try:
        with av.open(str(output / "global.mp4"), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width = 1280
            stream.height = 720
            stream.pix_fmt = "yuv420p"
            for pose in qpos:
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                camera.lookat[:] = pose[:3] + np.array([0.0, 0.0, 0.12])
                renderer.update_scene(data, camera=camera)
                frame = av.VideoFrame.from_ndarray(renderer.render(), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    finally:
        renderer.close()


class TextPreviewPlayer:
    """本地最新帧源：窗口关闭不停止引擎，慢渲染不阻塞播放或生成。"""

    def __init__(self, robot_manifest, kinematics):
        validate_robot_assets(robot_manifest, kinematics)
        from gem.robots.bumi.kinematics import BumiKinematics
        from gem.runtime.bumi_text_contract import sha256_file

        self.pose = PoseSnapshot(sha256_file(kinematics))
        self.standing = BumiKinematics(kinematics).make_standing_qpos().cpu().numpy().copy()
        self.frames = None
        self.cursor = 0
        self.state = "STAND"
        self.started = 0.0
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="bumi-text-playback", daemon=True)
        self.thread.start()
        try:
            self.viewer = MujocoPreview(
                robot_manifest, kinematics, lambda: LocalPreviewReader(self)
            )
        except Exception:
            self.stop_event.set()
            self.thread.join(timeout=2)
            raise

    def request(self, payload):
        if payload["command"] == "preview_frame":
            return self.pose.read()
        raise ValueError("预览只提供当前姿态")

    def play(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != 28 or not len(qpos) or not np.isfinite(qpos).all():
            raise ValueError("预览要求有限qpos[F,28]")
        with self.lock:
            self.frames = qpos.copy()
            self.cursor = 0
            self.started = time.monotonic()
            self.state = "PLAYING"

    def pause(self):
        with self.lock:
            self.state = "PAUSED"

    def resume(self):
        with self.lock:
            if self.frames is not None:
                self.started = time.monotonic() - self.cursor / 30
                self.state = "PLAYING"

    def stand(self):
        with self.lock:
            self.state = "STAND"
            self.frames = None
            self.cursor = 0

    def _loop(self):
        while not self.stop_event.wait(0.02):
            with self.lock:
                if self.state == "PLAYING":
                    self.cursor = min(
                        int((time.monotonic() - self.started) * 30), len(self.frames) - 1
                    )
                    if self.cursor == len(self.frames) - 1:
                        self.state = "FINISHED"
                value = self.standing if self.frames is None else self.frames[self.cursor]
                self.pose.record(
                    value,
                    frame_index=self.cursor,
                    revision=0,
                    request_id="local-text",
                    state=self.state,
                )

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.viewer.close()
