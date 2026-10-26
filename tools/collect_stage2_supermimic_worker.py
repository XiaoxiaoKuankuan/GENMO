"""正式训练管理的独立Newton物理采集子进程入口。

从显式配置加载单机器人及同源冻结SM，接收当前GENMO权重和采集命令，返回真实首次
接管、逐控制步数据与RNG。没有Actor/Critic/WM优化器，也不独立决定训练或自动重启；
只能在父训练进程管理的专用IPC目录工作，父进程离开后退出。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gem.closedloop.mb_control.entry import main


if __name__ == "__main__":
    main("collector")
