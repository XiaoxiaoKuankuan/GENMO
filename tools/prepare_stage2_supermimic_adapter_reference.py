"""通过冻结GENMO已有真实滚动生成流程准备固定名义缓存。

关闭ForecastContext，记录实际采用的原始世界姿态、音乐/初态/噪声来源，供后续三组
严格配对；不改变GENMO或Actor权重，不把实际执行状态当参考，不覆盖任何已有缓存。
本命令需要SM物理环境，只在用户显式执行时运行，不属于纯代码构建阶段的自动动作。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gem.closedloop.reference_adapter.entry import main

if __name__ == "__main__":
    raise SystemExit(main("prepare"))
