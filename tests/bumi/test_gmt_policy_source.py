"""GMT策略由控制器自主选择时的发现和路径解析回归。

测试只在系统临时目录创建假资产，并在随机回环端口启动只实现getParam的XML-RPC服务，
验证读取GMT当前配置、缺参/机器人不匹配失败、配置改变后的重新解析、Docker最长挂载
映射以及旧参数兼容。不会启动ROS、Redis、仿真或实机，也不会连接生产ROS master。
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from xmlrpc.server import SimpleXMLRPCServer

import pytest

from gem.runtime.gmt_policy_source import (
    discover_gmt_policy,
    map_container_policy,
    resolve_bridge_policy,
)


@pytest.fixture
def ros_parameters():
    parameters = {"/robot_type": "bumi"}
    calls = []
    server = SimpleXMLRPCServer(("127.0.0.1", 0), logRequests=False)

    def get_param(caller, name):
        calls.append((caller, name))
        return [1, "ok", parameters[name]] if name in parameters else [-1, "missing", ""]

    server.register_function(get_param, "getParam")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield parameters, f"http://127.0.0.1:{server.server_address[1]}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_follows_gmt_choice_without_pinned_filename(tmp_path, ros_parameters):
    parameters, uri, calls = ros_parameters
    for name in ("controller_A.onnx", "用户自己的 新策略.onnx"):
        policy = tmp_path / name
        policy.write_bytes(b"test metadata source")
        parameters["/gmtPolicyFile"] = str(policy)
        path, source = discover_gmt_policy(master_uri=uri)
        assert path == policy
        assert "/gmtPolicyFile" in source
    assert [name for _, name in calls] == ["/gmtPolicyFile", "/robot_type"] * 2


def test_missing_parameter_does_not_fall_back(ros_parameters):
    _, uri, _ = ros_parameters
    with pytest.raises(RuntimeError, match="不会回退"):
        discover_gmt_policy(master_uri=uri)


def test_wrong_robot_is_rejected(tmp_path, ros_parameters):
    parameters, uri, _ = ros_parameters
    policy = tmp_path / "policy.onnx"
    policy.touch()
    parameters.update({"/gmtPolicyFile": str(policy), "/robot_type": "e1"})
    with pytest.raises(RuntimeError):
        discover_gmt_policy(master_uri=uri)


def test_actual_docker_bind_mapping(tmp_path, ros_parameters, monkeypatch):
    parameters, uri, _ = ros_parameters
    policy = tmp_path / "policy.onnx"
    policy.touch()
    parameters["/gmtPolicyFile"] = "/container/gmt/policy.onnx"
    mounts = [{"Type": "bind", "Source": str(tmp_path), "Destination": "/container/gmt"}]

    def inspect(command, **kwargs):
        assert command == ["docker", "inspect", "--type", "container", "noetic-test"]
        assert kwargs["timeout"] == 5
        return SimpleNamespace(stdout=json.dumps([{"Mounts": mounts}]))

    monkeypatch.setattr("gem.runtime.gmt_policy_source.subprocess.run", inspect)
    path, source = discover_gmt_policy(master_uri=uri, container="noetic-test")
    assert path == policy
    assert "bind mount" in source


def test_nested_mount_and_escape_are_not_guessed(tmp_path):
    outer, inner = tmp_path / "outer", tmp_path / "inner"
    outer.mkdir()
    inner.mkdir()
    policy = inner / "policy.onnx"
    policy.touch()
    mounts = [
        {"Type": "bind", "Source": str(outer), "Destination": "/workspace"},
        {"Type": "bind", "Source": str(inner), "Destination": "/workspace/models"},
    ]
    assert map_container_policy("/workspace/models/policy.onnx", mounts) == policy
    with pytest.raises(ValueError):
        map_container_policy("/workspace/models/../policy.onnx", mounts)
    external = tmp_path / "outside.onnx"
    external.touch()
    (inner / "escape.onnx").symlink_to(external)
    with pytest.raises(ValueError):
        map_container_policy("/workspace/models/escape.onnx", mounts)
    mounts[1]["Type"] = "volume"
    with pytest.raises(ValueError, match="bind mount"):
        map_container_policy("/workspace/models/policy.onnx", mounts)


def test_legacy_path_needs_no_ros(tmp_path):
    path = tmp_path / "old.onnx"
    path.touch()
    assert resolve_bridge_policy(SimpleNamespace(gmt_policy=path))[0] == path


def test_bridge_default_does_not_specify_policy():
    from scripts.demo.demo_bumi_gmt_bridge import build_parser

    args = build_parser().parse_args(["--kinematics", "robot.json"])
    assert args.gmt_policy is None
