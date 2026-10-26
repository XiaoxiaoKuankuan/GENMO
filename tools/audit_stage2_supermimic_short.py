"""在匹配Newton后端运行真实接管短时物理方向审计的命令行入口。

实际资产、预声明阈值、任务与独立输出目录由命令行提供；真实采集后从初态重放前缀，
分别验证8/16/24/32控制步。导入无训练或GPU副作用，不允许用此入口自动更新GENMO。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gem.closedloop.mb_control.entry import main

if __name__ == "__main__":
    main("short")
