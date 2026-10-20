"""保留旧导入位置，实际实现位于算法公共层。

PPO与AWR共享同一asset_cache模块对象，不复制奖励或物理实现；既有测试的模块级补丁
也作用于唯一实现。此兼容导出不增加AWR分支或改变原数据契约。
"""
import sys
from gem.closedloop.common import asset_cache as _implementation
sys.modules[__name__] = _implementation
