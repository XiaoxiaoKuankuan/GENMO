"""在服务器2真实物理上独立预热Upper Critic，不更新GENMO或底层SM。

沿用正式训练的完整MC采集、独立音乐验证和质量阈值，覆盖任意50Hz/pending状态。
通过后保存可导入的价值参数/优化器与实际模型、奖励、调度合同，源码版本只记录。
此命令只进行价值预热，不表示GENMO正式执行反馈训练已经启动或物理性能提升。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gem.closedloop.mb_control.entry import main

if __name__ == "__main__":
    main("critic")
