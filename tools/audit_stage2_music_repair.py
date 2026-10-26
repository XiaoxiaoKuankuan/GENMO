"""用真实在线Stage1请求检查GENMO候选输出信任域及Newton前后执行。

入口读取独立在线教师的完整逐步证据，使用原监督数据和实际20步DDIM执行有限诊断
更新，保存候选、回滚、物理轨迹和耗时。不会自动启动正式训练或宣称单例质量收益。
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gem.closedloop.mb_control.entry import main
if __name__=="__main__":main("music_repair")
