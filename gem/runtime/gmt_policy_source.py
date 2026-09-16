"""只读获取 GMT 自己选择的策略路径，避免 GENMO 再维护一份控制器权重选择。

正常在线启动通过 ROS 1 参数服务读取 /gmtPolicyFile，该参数正是当前 GMT AcController
加载策略时读取的来源。本模块只执行 getParam，不启动 ROS、不设置参数、不修改 launch，
也不替换控制器模型。Bridge 读取该 ONNX 仅用于确认输入契约、关节顺序和默认站姿。

GMT 在 Docker 内运行时，参数可能是容器路径；宿主机不可直接访问该路径时，通过
docker inspect 的实际 bind mount 映射回宿主机文件，使用最长匹配挂载，不猜测工作区。
找不到 ROS 参数、容器映射或本地文件就明确失败，不回退到部署包里的旧 policy。
显式 --gmt-policy 仅为历史命令兼容保留，新部署命令使用 ROS 参数自动发现。
"""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import xmlrpc.client
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse


class _TimeoutTransport(xmlrpc.client.Transport):
    def make_connection(self, host):
        return http.client.HTTPConnection(host, timeout=2.0)


def map_container_policy(path: str, mounts: list[dict[str, Any]]) -> Path:
    """按 Docker 实际 bind mount 映射文件；拒绝路径穿越、缺失和非 bind 资产。"""
    remote = PurePosixPath(path)
    if not remote.is_absolute() or ".." in remote.parts:
        raise ValueError("GMT policy parameter must be an absolute path without '..'")
    candidates = []
    for mount in mounts:
        destination = PurePosixPath(mount.get("Destination", ""))
        if destination.is_absolute() and remote.is_relative_to(destination):
            candidates.append((len(destination.parts), destination, mount))
    if not candidates:
        raise FileNotFoundError(f"GMT policy is not covered by a Docker bind mount: {path}")
    _, destination, mount = max(candidates, key=lambda item: item[0])
    if mount.get("Type") != "bind":
        raise ValueError("GMT policy must be accessible through a bind mount on the GENMO host")
    source = Path(mount["Source"]).resolve(strict=True)
    resolved = (source / str(remote.relative_to(destination))).resolve(strict=True)
    if not resolved.is_relative_to(source) or not resolved.is_file():
        raise ValueError("GMT policy is not a regular file within the mapped bind mount")
    return resolved


def discover_gmt_policy(
    *, master_uri: str | None = None, container: str = "noetic"
) -> tuple[Path, str]:
    """读取 GMT 当前 ROS 配置；不会选择、下载或运行控制策略。"""
    uri = master_uri or os.environ.get("ROS_MASTER_URI", "http://127.0.0.1:11311")
    parsed = urlparse(uri)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("ROS_MASTER_URI must be an HTTP ROS 1 master address without credentials")
    try:
        with xmlrpc.client.ServerProxy(uri, transport=_TimeoutTransport()) as master:
            code, message, value = master.getParam("/genmo_bumi_bridge", "/gmtPolicyFile")
            if code != 1 or not isinstance(value, str) or not value.strip():
                raise ValueError(f"missing /gmtPolicyFile: {message}")
            robot_code, _, robot_type = master.getParam("/genmo_bumi_bridge", "/robot_type")
            if robot_code == 1 and robot_type != "bumi":
                raise ValueError(f"BUMI Bridge requires robot_type=bumi, got {robot_type!r}")
    except (OSError, xmlrpc.client.Error, ValueError) as exc:
        raise RuntimeError(
            f"无法从 GMT 的 ROS 参数读取策略：{uri} /gmtPolicyFile；"
            "请先启动 GMT，并核对 deployment.ini 的 [gmt] ros_master_uri"
            "（旧 demo 命令读取 ROS_MASTER_URI）。GENMO 不会回退到固定 policy。"
        ) from exc
    remote = PurePosixPath(value)
    if not remote.is_absolute() or ".." in remote.parts:
        raise ValueError("GMT /gmtPolicyFile must be an absolute path without '..'")
    local = Path(value)
    if local.is_file():
        return local.resolve(strict=True), f"ROS {uri} /gmtPolicyFile={value}"
    if not container:
        raise FileNotFoundError(f"GMT policy is not readable on the GENMO host: {value}")
    try:
        inspected = subprocess.run(
            ["docker", "inspect", "--type", "container", container],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        entries = json.loads(inspected.stdout)
        if not isinstance(entries, list) or len(entries) != 1:
            raise ValueError("docker inspect did not return one container")
        local = map_container_policy(value, entries[0]["Mounts"])
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        raise RuntimeError(
            f"GMT policy 位于容器路径 {value}，无法通过容器 {container!r} 的实际 bind mount 读取；"
            "请核对 deployment.ini 的 [gmt] container 和工作区挂载"
            "（旧 demo 命令使用 --gmt-container）。"
        ) from exc
    return local, f"ROS {uri} /gmtPolicyFile={value}; Docker {container} bind mount"


def resolve_bridge_policy(args: Any) -> tuple[Path, str]:
    legacy = getattr(args, "gmt_policy", None)
    if legacy is not None:
        return Path(legacy).expanduser().resolve(strict=True), "legacy explicit --gmt-policy"
    return discover_gmt_policy(
        master_uri=getattr(args, "ros_master_uri", None),
        container=getattr(args, "gmt_container", "noetic"),
    )
