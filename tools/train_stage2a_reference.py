"""Stage2-A单GPU独立WM+BPTT训练命令，不导入或调用旧PPO入口。

具体配置明确冻结资产、原GENMO缓存、训练标尺、Oracle证据和有限修复预算；候选
通过真实物理参数信任域才计为有效更新。完整状态保存与失败恢复由stage2a_entry管理。
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gem.closedloop.reference_adapter.stage2a_entry import main

if __name__ == '__main__':
    raise SystemExit(main())
