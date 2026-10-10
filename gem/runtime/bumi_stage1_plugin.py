"""加载当前Stage1的CUDA掩码插件，并绑定引擎与共享库的完整身份。

本模块不编译插件、不运行模型，也不转换采样输入。构建器和TensorRT运行器共用
同一加载入口：先校验目标共享库SHA256，再用ctypes保留共享库句柄、显式注册
genmo_stage1命名空间下的IPluginV3 creator。注册必须发生在ONNX解析或引擎反序列化前。
同一进程只允许一份该creator实现，防止同时加载不同包时静默复用旧代码。
engine.json仅允许引用同目录插件文件，部署清单还会校验包内路径、大小与哈希。
原始ONNX Runtime后端不需要此插件；GMT、Redis协议和采样输出均不受本模块影响。
"""

from __future__ import annotations

import ctypes
import hashlib
import threading
from pathlib import Path

PLUGIN_NAME = "BumiStage1Mask"
PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "genmo_stage1"
PLUGIN_FILENAME = "libbumi_stage1_mask.so"
_handles = {}
_lock = threading.Lock()


def plugin_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def engine_plugin_path(engine_path, metadata):
    """严格解析同目录插件引用，不允许元数据跳到系统或其他模型的共享库。"""
    record = metadata.get("plugin_library")
    if not isinstance(record, dict) or record.get("path") != PLUGIN_FILENAME:
        raise ValueError("Stage1 engine缺少当前CUDA掩码插件记录")
    expected = metadata.get("build_options", {}).get("plugin_library_sha256")
    if record.get("sha256") != expected:
        raise ValueError("Stage1插件哈希与构建策略不一致")
    engine = Path(engine_path).resolve(strict=True)
    path = (engine.parent / PLUGIN_FILENAME).resolve(strict=True)
    if path.parent != engine.parent or not path.is_file() or plugin_sha256(path) != expected:
        raise ValueError("Stage1插件文件路径或SHA256不匹配")
    return path, expected


def load_stage1_plugin(path, expected_sha256, trt):
    """仅加载核验过的共享库并注册creator，不执行CUDA内核或模型预热。"""
    path = Path(path).expanduser().resolve(strict=True)
    if plugin_sha256(path) != expected_sha256:
        raise ValueError(f"Stage1插件SHA256不匹配: {path}")
    with _lock:
        if _handles:
            if expected_sha256 not in _handles:
                raise RuntimeError("同一进程不能混用不同Stage1插件；请重新启动控制台")
            return _handles[expected_sha256]
        registry = trt.get_plugin_registry()
        if registry.get_creator(PLUGIN_NAME, PLUGIN_VERSION, PLUGIN_NAMESPACE) is not None:
            raise RuntimeError("Stage1插件creator已被其他实现注册，拒绝混用")
        handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        register = handle.bumi_stage1_register_plugins
        register.argtypes = []
        register.restype = ctypes.c_bool
        if not register() or registry.get_creator(PLUGIN_NAME, PLUGIN_VERSION, PLUGIN_NAMESPACE) is None:
            raise RuntimeError("Stage1 CUDA掩码插件注册失败")
        _handles[expected_sha256] = handle
        return handle
