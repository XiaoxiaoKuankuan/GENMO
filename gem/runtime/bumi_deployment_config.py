"""读取 BUMI 便携部署的单一配置，并为现有入口生成一致的启动参数。

用户只需在项目根目录 deployment.ini 中修改模型清单、GPU、DDIM、ZeroMQ、Redis、
ROS 参数服务和容器名。Console 和 Bridge 共用同一 ZeroMQ 配置，Bridge 的运动学文件
从已通过完整指纹校验的模型清单读取，避免分别编辑路径后混用模型。GMT policy 不属于
此配置，仍由 GMT 自己选择。显式 ROS URI 防止终端环境变量意外覆盖文件中的配置。

本模块保留原 demo 的命令行接口，仅供 run.sh 的统一入口使用。相对路径以配置文件
所在目录为基准，允许部署包整体移动；解析时拒绝拼错的字段、越界端口和非法采样参数。
读取配置不启动推理或网络；只有构造 Bridge 命令时才校验模型清单。
"""

from __future__ import annotations

import configparser
import math
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class DeploymentConfig:
    path: Path
    manifest: Path
    device: str
    ddim_steps: int
    guidance_scale: float
    bridge_endpoint: str
    audio_playback: str
    redis_host: str
    redis_port: int
    redis_db: int
    redis_key: str
    ros_master_uri: str
    container: str


def load_deployment_config(path: str | Path) -> DeploymentConfig:
    path = Path(path).expanduser().resolve(strict=True)
    parser = configparser.ConfigParser(interpolation=None)
    with path.open(encoding="utf-8") as stream:
        parser.read_file(stream)
    expected = {
        "model": {"manifest", "device", "ddim_steps", "guidance_scale"},
        "bridge": {"host", "port", "audio_playback"},
        "redis": {"host", "port", "db", "key"},
        "gmt": {"ros_master_uri", "container"},
    }
    if parser.defaults() or set(parser.sections()) != set(expected):
        raise ValueError("deployment.ini 必须包含且仅包含 model/bridge/redis/gmt 四个配置段")
    for section, keys in expected.items():
        if set(parser[section]) != keys:
            raise ValueError(f"{section} 字段不匹配，必须是 {sorted(keys)}")
        for key in keys - {"container"}:
            if not parser[section][key].strip():
                raise ValueError(f"{section}.{key} 不能为空")

    def port(section: str) -> int:
        value = parser.getint(section, "port")
        if not 1 <= value <= 65535:
            raise ValueError(f"{section}.port 必须为 1–65535")
        return value

    def host(section: str) -> str:
        value = parser[section]["host"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value == "0.0.0.0":
            raise ValueError(f"{section}.host 必须是可连接的 IPv4 地址或主机名")
        return value

    uri = parser["gmt"]["ros_master_uri"]
    parsed = urlparse(uri)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.port is None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        raise ValueError("gmt.ros_master_uri 必须是含端口的 HTTP ROS master 地址")
    if not 1 <= parsed.port <= 65535:
        raise ValueError("ROS master 端口必须为 1–65535")
    steps = parser.getint("model", "ddim_steps")
    guidance = parser.getfloat("model", "guidance_scale")
    db = parser.getint("redis", "db")
    if not 2 <= steps <= 1000 or not math.isfinite(guidance) or guidance <= 0 or db < 0:
        raise ValueError("DDIM 必须为 2–1000，CFG 必须为有限正数，Redis db 必须非负")
    device = parser["model"]["device"]
    if not re.fullmatch(r"cuda:[0-9]+", device):
        raise ValueError("model.device 必须为 cuda:0 等 CUDA 设备")
    audio = parser["bridge"]["audio_playback"]
    if audio not in {"off", "ffplay"}:
        raise ValueError("bridge.audio_playback 必须为 ffplay 或 off")
    manifest = Path(parser["model"]["manifest"]).expanduser()
    if not manifest.is_absolute():
        manifest = path.parent / manifest
    return DeploymentConfig(
        path,
        manifest.resolve(),
        device,
        steps,
        guidance,
        f"tcp://{host('bridge')}:{port('bridge')}",
        audio,
        host("redis"),
        port("redis"),
        db,
        parser["redis"]["key"],
        uri,
        parser["gmt"]["container"],
    )


def deployment_command(config: DeploymentConfig, role: str) -> tuple[str, list[str]]:
    """返回既有脚本和参数；不启动进程，不修改任何控制器配置。"""
    model = ["--deployment-manifest", str(config.manifest)]
    gmt = ["--ros-master-uri", config.ros_master_uri, "--gmt-container", config.container]
    if role in {"check", "check-gmt"}:
        options = ["--check-gmt", *gmt] if role == "check-gmt" else ["--inference"]
        return "check_bumi_deployment.py", [*model, "--device", config.device, *options]
    if role == "genmo":
        return "demo_music_bumi_console.py", [
            *model,
            "--backend",
            "tensorrt",
            "--device",
            config.device,
            "--bridge",
            config.bridge_endpoint,
            "--ddim-steps",
            str(config.ddim_steps),
            "--guidance-scale",
            str(config.guidance_scale),
        ]
    if role == "bridge":
        from gem.runtime.bumi_deployment_bundle import load_bumi_deployment_manifest

        bundle = load_bumi_deployment_manifest(config.manifest)
        return "demo_bumi_gmt_bridge.py", [
            "--kinematics",
            str(bundle.paths["kinematics"]),
            "--bind",
            config.bridge_endpoint,
            *gmt,
            "--redis-host",
            config.redis_host,
            "--redis-port",
            str(config.redis_port),
            "--redis-db",
            str(config.redis_db),
            "--redis-key",
            config.redis_key,
            "--audio-playback",
            config.audio_playback,
            "--verbose",
        ]
    raise ValueError(f"未知运行角色：{role}")
