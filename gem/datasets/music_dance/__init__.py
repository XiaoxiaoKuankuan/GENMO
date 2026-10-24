"""人类与机器人音乐动作数据集的按需导出入口。

BUMI读取原manifest时只装载BUMI reader；显式请求SMPL类时才导入人体模型及图像
依赖。两个公开类名保持不变，不改动数据内容、采样或网络，避免机器人运行环境
因为无关的SMPL/OpenCV模块而无法加载原Stage1配对库。
"""

from .music_dance_bumi import BumiMusicDanceDataset

__all__ = ["BumiMusicDanceDataset", "MusicDanceSmplDataset"]


def __getattr__(name):
    if name == "MusicDanceSmplDataset":
        from .music_dance_smpl import MusicDanceSmplDataset
        return MusicDanceSmplDataset
    raise AttributeError(name)
