"""新方法三种互相独立的评价入口。

重建固定GT噪声身份；自由生成从纯噪声完整DDIM；闭环从真实历史连续接管并执行Newton。
只接受本run的checkpoint。结果区分局部重建、自由生成分布与完整连续物理，不混写收益。
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gem.closedloop.joint_physics.entry import main

if __name__ == "__main__":
    main("evaluate")
