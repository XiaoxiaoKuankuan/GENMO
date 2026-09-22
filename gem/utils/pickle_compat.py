"""可信本地数据的 NumPy pickle 兼容读取。

只处理 NumPy 2 与 NumPy 1 的模块路径差异，保留原有反序列化行为；输入仍必须来自
可信数据源。本模块不附带旧 GMR 动作格式、旧机器人关节顺序或任何资产默认值。
"""

from __future__ import annotations

import pickle
from typing import Any


class _NumpyCompatibleUnpickler(pickle.Unpickler):
    """兼容 NumPy 2 ``numpy._core`` 路径的受信任本地 pickle reader。"""

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module == "numpy._core" or module.startswith("numpy._core."):
                legacy_module = "numpy.core" + module[len("numpy._core") :]
                return super().find_class(legacy_module, name)
            raise
