"""独立收集冻结GENMO的完整真实物理MC，用于新音乐奖励Critic。

任务、划分、种子和全局episode偏移由配置明确给出；不同进程可以收集不相交的
完整歌曲。保留真实终止、每个控制步奖励和价值快照，行政截断禁止冒充MC。
本入口不优化Actor或Critic，合并与独立验证由单独拟合入口完成。
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gem.closedloop.mb_control.entry import main
if __name__ == "__main__":
    main("mc")
