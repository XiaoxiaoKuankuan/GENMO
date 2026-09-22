# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""常驻推理接口的延迟导入入口。

保留原有from gem.runtime import接口；仅加载调用者实际使用的后端，避免纯BUMI文本
部署被SMPL、视频或其他训练依赖拖入。直接导入子模块同样不会启动无关模型。
"""

from importlib import import_module

_EXPORTS = {
    "ResidentBumiTextEngine": "bumi_text_runtime",
    "encode_prompt_with_loaded_t5": "text_encoding",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
