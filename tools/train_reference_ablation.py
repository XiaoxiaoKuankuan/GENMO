"""启动从零初始化的A/B/C单GPU真实物理修复对照。

读取显式配置与全新输出目录，保留旧8组失败模型和日志；每次有效更新为一个完整
Adam参数步，真实评价固定为25/50/100。A/B审计完整参数方向且有修复预算，C使用
真实物理潜变量PPO；所有组冻结GENMO和SuperMimic Actor，不启动跨GPU通信。
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    from gem.closedloop.reference_adapter.repair_entry import main
    raise SystemExit(main())
