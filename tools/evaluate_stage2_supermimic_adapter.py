"""零Adapter/无音乐Adapter/有音乐Adapter的真实配对评价入口。

显式读取音乐、初态和同一个名义缓存，再在原SM物理后端记录完整率、六项跟踪、
节拍/活动/jerk/接触脚滑与控制延迟；三组预算比较使用--compare，不自动运行训练。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gem.closedloop.reference_adapter.entry import main

if __name__ == "__main__":
    raise SystemExit(main("evaluate"))
