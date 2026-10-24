"""真实全曲SuperMimic评价入口，不在导入时加载模型或初始化物理。

读取独立Stage2配置，复用现有Stage1与同源SM组合包，通过专用entry装配所需流程。
输出写入新的独立目录；所有测试/推理/训练只在用户明确执行本命令时发生。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gem.closedloop.mb_control.entry import main


if __name__ == "__main__":
    main("evaluate")
