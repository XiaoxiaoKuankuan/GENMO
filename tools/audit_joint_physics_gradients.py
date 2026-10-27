"""分离执行导数、有限差分和真实Newton方向的审计入口。

监督梯度不作为执行梯度证据。--interface-probe只验证GT邻域接口并明确标记；学生执行必须
使用真实新checkpoint的x0预测。--newton从episode初态重放旧控制前缀后做配对扰动。
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gem.closedloop.joint_physics.entry import main

if __name__ == "__main__":
    main("gradients")
