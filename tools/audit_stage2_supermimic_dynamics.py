"""固定SM候选的真实动力学故障定位入口；所有执行由显式CLI触发。

读取现有Stage2资产与已知30Hz参考，在同一Newton/MuJoCo-Warp物理合同下，分别
比较记录的真实动作自由滚动、固定参考SM闭环及真实输入的一步WM预测。候选仍从
同一个编号checkpoint同时加载pi/wm并冻结；不训练、不修改normalizer、不读取
未发生的真实状态作为在线条件。此局部诊断不覆盖GENMO连续滚动或物理梯度方向，
因此输出不能用作正式启动许可，也不替换完整P3报告。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gem.closedloop.mb_control.entry import main


if __name__ == "__main__":
    main("dynamics")
