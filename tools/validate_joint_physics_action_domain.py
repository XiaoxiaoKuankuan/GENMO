"""补充冻结WM动作输入的真实Newton认证入口。

读取本方法已通过的H/运动接触组合方向报告，在其原固定anchor上实际执行WM反馈SM动作，
检查同动作动力学误差及安全终止后发布独立新证书。不会改原报告或用GT接口冒充学生更新，
所有真实动作和状态轨迹保存到新目录，供支持域与训练门禁审计。
"""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gem.closedloop.joint_physics.entry import main

if __name__=='__main__':main('action_domain')
