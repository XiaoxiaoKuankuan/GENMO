"""在独立克隆运行中采集原DPPO的完整固定数据验证样本。

数据流：原训练器从完整checkpoint恢复Actor、Critic、两个Adam、LR控制器与八卡
随机状态，在新建PhysX世界采集一个真实rollout，随后退出，不执行Actor/Critic更新。
本入口只把原诊断写入器的full_execution选项设为True，保留真实GMT控制、参考消费
与奖励证据；采样、条件、GAE、物理时间和训练数学均继续使用原实现。

边界：只接受有origin.json来源证明的独立capture克隆目录，禁止把正式训练目录作为
输出。修改仅存在于此进程的诊断序列化入口，不修改生产源码、原checkpoint或账本。
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_capture_arguments(arguments):
    if '--collect-only' not in arguments or '--resume' not in arguments:
        raise ValueError('固定数据采集必须显式collect-only并恢复完整状态')
    output = Path(arguments[arguments.index('--output-dir') + 1]).resolve()
    origin = json.loads((output.parent / 'origin.json').read_text())
    if output != Path(origin['clone']).resolve() or output == Path(origin['source']).resolve():
        raise ValueError('固定数据只能写入已登记的独立克隆目录')
    return output


def main():
    validate_capture_arguments(sys.argv[1:])
    from gem.closedloop.dppo import diagnostic_rollout
    original = diagnostic_rollout.persist_rollout

    def persist_full_execution(*args, **kwargs):
        kwargs['full_execution'] = True
        return original(*args, **kwargs)

    diagnostic_rollout.persist_rollout = persist_full_execution
    from tools.train_closedloop_stage10 import main as train_main
    return train_main()


if __name__ == '__main__':
    raise SystemExit(main())
