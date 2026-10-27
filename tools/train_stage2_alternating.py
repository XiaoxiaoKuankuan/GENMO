"""正式单GPU Stage2-A交替训练命令。

从显式配置读取冻结GENMO缓存和SuperMimic权重，进入真实采集/H8 WM/BPTT/可选
Critic循环；不调用旧Oracle、方向审计或PPO。进程只使用CUDA_VISIBLE_DEVICES
指定的一张卡；八卡由独立启动器分别管理，支持完整checkpoint恢复到新输出目录。
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gem.closedloop.reference_adapter.alternating_entry import main

if __name__ == '__main__':
    raise SystemExit(main())
