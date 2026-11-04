"""使用指定noetix源码和冻结40k底层Actor启动独立单卡GENMO参考Adapter训练。

先显式加入源码包src和当前GENMO目录，避免误导入旧supermimic.bridge；可单独准备
GENMO参考数据，或按JSON实验配置运行。每组独立模拟器、WM、Adapter和Replay。
不修改指定源码包、不停止已有任务，正式启动和数值核验由训练模块负责。
"""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--config')
    parser.add_argument('--prepare-tasks')
    parser.add_argument('--destination')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(args.source) / 'src'))
    if args.prepare_tasks:
        from gem.closedloop.noetix_adapter.data import prepare
        print(json.dumps(prepare(args.prepare_tasks, args.destination), ensure_ascii=False, indent=2))
    else:
        from gem.closedloop.noetix_adapter.training import run
        run(json.loads(Path(args.config).read_text()), args.source)


if __name__ == '__main__':
    main()
