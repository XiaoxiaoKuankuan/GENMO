#!/usr/bin/env python3
"""BUMI 纯运动学动画窗口，作为常驻 GENMO 的独立子进程使用。

从匿名管道读取带版本/帧号的原生 qpos28，只显示最新一帧；没有新姿态时保持画面，
超过 0.2 秒提示参考暂停。检查机器人指纹和 MuJoCo 关节地址后逐帧调用 mj_forward，
不调用 mj_step、不运行策略、不连接 ROS/Redis。鼠标仅用于改变相机视角；关闭窗口
不向控制器发送任何指令。管道 EOF 时退出，避免控制台退出后留下孤立窗口。
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import select
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gem.runtime.bumi_preview import validate_robot_assets


@contextmanager
def managed_viewer(model, data):
    """等待本窗口创建的渲染线程退出，避免 GLFW 在解释器退出时提前释放 GLX。"""
    import mujoco.viewer

    before = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    workers = set(threading.enumerate()) - before
    try:
        yield viewer
    finally:
        # MuJoCo 3.2.3 的 Handle.close() 只设置退出请求，不等待渲染线程。
        # 本查看器是独立子进程，只回收此次 launch 创建的 Python 线程。
        viewer.close()
        deadline = time.monotonic() + 2.0
        for thread in workers:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in workers):
            raise RuntimeError("MuJoCo 渲染线程未在关闭期限内退出")
        # 独立子进程只有一个窗口；在 Python 模块销毁前完成 GLFW 清理。
        # 与上面的 join 一起避免渲染线程与解释器清理发生竞争。
        import glfw

        glfw.terminate()
        atexit.unregister(glfw.terminate)


def load_model(manifest, kinematics):
    import mujoco

    path, spec = validate_robot_assets(manifest, kinematics)
    model = mujoco.MjModel.from_xml_path(str(path))
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
        if model.jnt_qposadr[i] >= 7
    ]
    if model.nq != 28 or model.nv != 27 or names != spec["joint_order"]:
        raise ValueError("MuJoCo 关节排列与模型原生 qpos 契约不一致")
    if int(model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("MuJoCo 根关节必须为自由关节")
    return model, spec


def main():
    import mujoco
    import mujoco.viewer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-manifest", type=Path, required=True)
    parser.add_argument("--kinematics", type=Path, required=True)
    args = parser.parse_args()
    model, spec = load_model(args.robot_manifest, args.kinematics)
    data = mujoco.MjData(model)
    data.qpos[:] = spec["default_qpos"]
    mujoco.mj_forward(model, data)
    buffer = b""
    last_frame_time = time.monotonic()
    stale = False
    with managed_viewer(model, data) as viewer:
        with viewer.lock():
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -15
            viewer.cam.lookat[:] = data.qpos[:3]
        sys.stdout.buffer.write(b"V")
        sys.stdout.buffer.flush()
        while viewer.is_running():
            started = time.monotonic()
            latest = None
            while select.select([sys.stdin.buffer], [], [], 0)[0]:
                block = os.read(sys.stdin.fileno(), 65536)
                if not block:
                    return
                buffer += block
                if len(buffer) > 131072:
                    raise ValueError("预览管道出现无效长消息")
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    latest = json.loads(line)
            if latest is not None and time.monotonic() - latest["observed_monotonic"] <= 0.2:
                qpos = np.asarray(latest["qpos"], dtype=float)
                if (
                    qpos.shape != (28,)
                    or not np.isfinite(qpos).all()
                    or abs(np.linalg.norm(qpos[3:7]) - 1) > 1e-3
                ):
                    raise ValueError("无效预览 qpos")
                with viewer.lock():
                    data.qpos[:] = qpos
                    data.time = latest["frame_index"] / 50.0
                    mujoco.mj_forward(model, data)
                    viewer.cam.lookat[:] = qpos[:3]
                last_frame_time = time.monotonic()
                stale = False
            if time.monotonic() - last_frame_time > 0.2 and not stale:
                print("[MuJoCo] 超过 0.2 秒无新参考，画面保持最后姿态", file=sys.stderr, flush=True)
                stale = True
            viewer.sync()
            time.sleep(max(0.0, 0.02 - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
