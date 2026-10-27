"""随机初始化联合训练和完整新schema恢复入口。

从静态配置构建原音乐Transformer；一个AdamW优化原监督及有效执行约束。默认有界smoke，
正式单卡/torchrun必须显式选择train；旧GENMO/Stage2权重和优化器不能作为初始化或恢复。
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gem.closedloop.joint_physics.entry import main

if __name__ == "__main__":
    main("train")
