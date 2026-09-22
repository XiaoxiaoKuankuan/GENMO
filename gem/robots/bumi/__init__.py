"""BUMI 资产与正向运动学公共接口。

本分支的旧93D训练实现已经退役；保留不依赖学习表示的运动学供资产核验使用。
现行 qpos30 训练分别由 BUMI 文本与音乐分支维护。
"""
from .kinematics import BumiKinematics

__all__ = ["BumiKinematics"]
