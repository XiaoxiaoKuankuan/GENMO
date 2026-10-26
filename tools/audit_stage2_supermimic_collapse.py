"""服务器2动作坍缩复现入口：真实原Stage1请求上的多检查点同条件、同噪声比较。

配置显式列出完整1/100/200检查点和音乐任务；复用匹配Newton后端，逐分支重放
真实前史并保存原始参考、真实状态、身体轨迹和模型奖励。输出必须为独立目录。
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gem.closedloop.mb_control.entry import main
if __name__ == "__main__":main("collapse")
