"""验证统一配置、可搬迁启动及 TensorRT 库来源检查。

所有配置与模型夹具写入 pytest 临时目录，不连接 Redis/ROS、不创建 GPU 上下文。
覆盖两端端口共用、终端环境不能覆盖显式 ROS URI、模型清单导出正确运动学路径、
非法配置提前拒绝，以及 wheel/系统 libnvinfer 的实际加载路径识别。安装器另外在隔离
目录验证完整训练目录保护；真实环境安装与 engine 检查由独立部署验收执行。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from gem.runtime import tensorrt_environment as trt_env
from gem.runtime.bumi_deployment_config import deployment_command, load_deployment_config
from scripts.demo.demo_bumi_gmt_bridge import build_parser as bridge_parser
from scripts.demo.demo_music_bumi_console import build_parser as console_parser
from tests.bumi import test_bumi_deployment_bundle as bundle_tests

ROOT = Path(__file__).resolve().parents[2]
deployment_manifest = bundle_tests.deployment_manifest


@pytest.fixture
def config_path(tmp_path):
    directory = tmp_path / "部署 空格目录"
    directory.mkdir()
    target = directory / "deployment.ini"
    shutil.copyfile(ROOT / "deployment.ini", target)
    return target


def test_shared_endpoint_and_manifest_resolve_independent_of_cwd(
    config_path, deployment_manifest, monkeypatch
):
    text = config_path.read_text().replace("7022", "27022")
    text = text.replace("models/bumi_v5_s350000/deployment.json", str(deployment_manifest))
    config_path.write_text(text)
    monkeypatch.chdir(config_path.parent.parent)
    monkeypatch.setenv("ROS_MASTER_URI", "http://wrong-host:9999")
    config = load_deployment_config(config_path)
    _, bridge_argv = deployment_command(config, "bridge")
    _, console_argv = deployment_command(config, "genmo")
    bridge = bridge_parser().parse_args(bridge_argv)
    console = console_parser().parse_args(console_argv)
    assert bridge.bind == console.bridge == "tcp://127.0.0.1:27022"
    assert bridge.ros_master_uri == "http://127.0.0.1:11311"
    assert bridge.gmt_policy is None and bridge.gmt_container == "noetic"
    assert bridge.kinematics == deployment_manifest.parent / "kinematics.json"
    assert console.deployment_manifest == deployment_manifest
    assert console.checkpoint is None and console.ddim_steps == 20
    assert bridge.redis_port == 6379 and bridge.redis_key == "gmt_online_frame_bumi"


def test_config_relative_model_is_relative_to_file(config_path):
    config = load_deployment_config(config_path)
    assert config.manifest == config_path.parent / "models/bumi_v5_s350000/deployment.json"
    script, argv = deployment_command(config, "check")
    assert script == "check_bumi_deployment.py" and "--inference" in argv
    assert "--check-gmt" not in argv
    _, argv = deployment_command(config, "check-gmt")
    assert "--check-gmt" in argv and "--inference" not in argv


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("port = 7022", "port = 0"),
        ("port = 6379", "port = 65536"),
        ("port = 7022", "poort = 7022"),
        ("ddim_steps = 20", "ddim_steps = 1"),
        ("ddim_steps = 20", "ddim_steps = 1001"),
        ("guidance_scale = 2.5", "guidance_scale = nan"),
        ("device = cuda:0", "device = cpu"),
        ("host = 127.0.0.1", "host = 0.0.0.0"),
        ("db = 0", "db = -1"),
        ("audio_playback = ffplay", "audio_playback = unknown"),
        ("http://127.0.0.1:11311", "http://127.0.0.1"),
        ("http://127.0.0.1:11311", "http://user:pass@localhost:11311"),
        ("http://127.0.0.1:11311", "http://localhost:70000"),
        ("container = noetic", "container = noetic\npolicy = /tmp/old.onnx"),
    ],
)
def test_invalid_config_rejected_before_network(config_path, old, new):
    config_path.write_text(config_path.read_text().replace(old, new))
    with pytest.raises(ValueError):
        load_deployment_config(config_path)


def test_library_maps_use_real_path_with_spaces_and_ignore_plugin(tmp_path):
    library = tmp_path / "部署 目录" / "libnvinfer.so.10"
    maps = f"a b c d e {library}\nf g h i j {library}\na b c d e /usr/lib/libnvinfer_plugin.so.10"
    assert trt_env.loaded_nvinfer_paths(maps) == [library]


def test_actual_mapped_library_wins_over_system_ldconfig(monkeypatch, tmp_path):
    library = tmp_path / "libnvinfer.so.10"
    monkeypatch.setattr(trt_env, "loaded_nvinfer_paths", lambda _: [library])
    monkeypatch.setattr(
        trt_env.ctypes.util, "find_library", lambda _: pytest.fail("不应查询另一套系统库")
    )
    calls = []

    class Version:
        def __call__(self):
            return 101303

    monkeypatch.setattr(
        trt_env.ctypes,
        "CDLL",
        lambda path: calls.append(path) or SimpleNamespace(getInferLibVersion=Version()),
    )
    assert trt_env.linked_tensorrt_version() == "10.13.3"
    assert calls == [str(library)]


def test_ambiguous_runtime_rejected(monkeypatch):
    monkeypatch.setattr(
        trt_env,
        "loaded_nvinfer_paths",
        lambda _: [Path("/a/libnvinfer.so.10"), Path("/b/libnvinfer.so.10")],
    )
    with pytest.raises(RuntimeError, match="多份"):
        trt_env.linked_tensorrt_version()


def test_installer_preserves_training_directory(tmp_path):
    shutil.copyfile(ROOT / "install.sh", tmp_path / "install.sh")
    (tmp_path / "setup.cfg").write_text("[metadata]\n")
    result = subprocess.run(["bash", str(tmp_path / "install.sh")], capture_output=True, text=True)
    assert result.returncode == 1 and "训练环境" in result.stderr
    assert not (tmp_path / ".venv").exists()


def test_installer_creates_environment_fills_missing_libraries_and_reuses_it(tmp_path):
    """用隔离可执行夹具覆盖无环境/无系统TensorRT路径，不触碰APT、网络或现有环境。"""
    project = tmp_path / "便携 目录"
    binaries = tmp_path / "bin"
    project.mkdir()
    binaries.mkdir()
    for name in ("install.sh", "run.sh"):
        shutil.copyfile(ROOT / name, project / name)
    for name in ("nvidia-smi", "ffmpeg", "ffplay", "redis-server", "redis-cli"):
        file = binaries / name
        file.write_text("#!/bin/bash\nexit 0\n")
        file.chmod(0o755)
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/bin/bash\n"
        'printf "python %s\\n" "$*" >> "$BUMI_TEST_LOG"\n'
        'if [[ "$*" == *tensorrt_environment.py* ]]; then\n'
        '  test -f "$BUMI_TEST_ROOT/.trt_ready"\n'
        "else exit 0; fi\n"
    )
    fake_python.chmod(0o755)
    uv = binaries / "uv"
    uv.write_text(
        "#!/bin/bash\nset -eu\n"
        'printf "uv %s\\n" "$*" >> "$BUMI_TEST_LOG"\n'
        'if [[ "$*" == *" venv "* ]]; then\n'
        "  mkdir -p .venv/bin\n"
        '  cp "$BUMI_TEST_PYTHON" .venv/bin/python\n'
        "fi\n"
        'if [[ "$*" == *tensorrt-runtime.lock* ]]; then touch .trt_ready; fi\n'
    )
    uv.chmod(0o755)
    log = tmp_path / "calls.log"
    env = os.environ | {
        "PATH": f"{binaries}:/usr/bin:/bin",
        "BUMI_TEST_LOG": str(log),
        "BUMI_TEST_ROOT": str(project),
        "BUMI_TEST_PYTHON": str(fake_python),
    }
    subprocess.run(
        ["bash", str(project / "install.sh")],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )
    calls = log.read_text()
    assert "venv --python 3.10 .venv" in calls
    assert "tensorrt-bindings.lock" in calls and "tensorrt-runtime.lock" in calls
    assert "run_bumi_deployment.py check" in calls
    assert "apt-get" not in calls
    log.write_text("")
    subprocess.run(
        ["bash", str(project / "install.sh")],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )
    calls = log.read_text()
    assert "runtime.lock" in calls and "tensorrt-" not in calls
    assert "venv --python" not in calls
