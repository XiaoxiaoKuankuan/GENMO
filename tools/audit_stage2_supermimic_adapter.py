"""十帧Adapter合同/数值审计和显式真实8/16步物理审计入口。

--self-test只运行CPU数值测试；默认审计使用实际名义缓存和SM权重，不启动Isaac。
--physical才进行当前Adapter分布的reset配对、预测误差和方向检查，不导入旧H16结论。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gem.closedloop.reference_adapter.entry import main

if __name__ == "__main__":
    raise SystemExit(main("audit"))
