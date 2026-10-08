"""审计旧第二阶段断点的固定位置表，并可导出修复后的独立 Actor 权重。

默认只读取源文件并把差异报告输出到终端；明确传 --output 才写一个全新的 weights-only
checkpoint。必须同时提供该 Stage2 初始化时使用的 Stage1 架构断点以及相同内容的
stats/FK，工具核验 SHA 和 Actor 条件接口，拒绝覆盖任何源或已有输出。修复会恢复所有
共享位置表别名，丢弃全部训练／优化器状态；所得文件只能新建训练，不能完整 resume。

示例：python tools/repair_stage2_position_encoding.py --source stage2.pt
      --architecture-checkpoint stage1.pt --stats stats.npz --kinematics fk.json
      [--output repaired_actor_weights.pt]
此工具不启动训练、GMT 或物理仿真，修复后策略仍需按新的基线执行验证。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gem.closedloop.dppo.position_repair import repair_stage2_weights


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--architecture-checkpoint', type=Path, required=True)
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--kinematics', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    arguments = parser.parse_args(argv)
    report = repair_stage2_weights(arguments.source, arguments.architecture_checkpoint,
                                  stats=arguments.stats, kinematics=arguments.kinematics, output=arguments.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
