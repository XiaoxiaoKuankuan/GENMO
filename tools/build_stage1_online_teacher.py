"""在独立数据准备阶段构建原Stage1加冻结SM的真实在线音乐教师。

读取明确音乐、原始权重及同款Newton配置，完整运行异步请求和参考接管，保存逐步
真实参考、执行状态、原请求噪声和版本。不会更新Stage1/SM，不使用学生轨迹改写目标，
不把墙钟异步采集与固定延迟离线回放混称。输出目录独占创建，保留失败证据。
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gem.closedloop.mb_control.entry import main
if __name__=="__main__":main("online_teacher")
