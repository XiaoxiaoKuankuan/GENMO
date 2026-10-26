"""十帧Adapter独立训练/音乐基线标定入口。

只在显式调用main后读配置并启动物理；默认最多四次有界更新，长训需显式--formal。
复用原SM环境和固定GENMO名义缓存，不使用旧GENMO Trainer，不覆盖已有输出目录。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gem.closedloop.reference_adapter.entry import main

if __name__ == "__main__":
    raise SystemExit(main("train"))
