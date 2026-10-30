"""验证现有全库真实sidecar上的冻结WM精度，允许与正式GENMO训练同时运行。

读取正式run配置或独立YAML，只加载冻结SM/WM组合包并调用已有连续缓存；CPU/FP32
执行全部一步预测以及固定锚点H1/8/16/32开环动作回放。不会启动Newton、GENMO、
优化器或GPU任务。输出必须是新目录，真实失败与缺失末状态单独记录，不覆盖旧证据。
"""

import argparse
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--anchors-per-source', type=int, default=8)
    parser.add_argument('--max-sources', type=int)
    args = parser.parse_args()
    if min(args.threads, args.batch_size, args.anchors_per_source) < 1 or (args.max_sources is not None and args.max_sources < 1):
        parser.error('线程、批量、锚点和可选来源上限必须为正数')
    from gem.closedloop.joint_physics.config import read_config
    cfg = read_config(args.config)
    sys.path.insert(0, str(Path(cfg['supermimic']['repo']).resolve()))
    from gem.closedloop.joint_physics.cached_wm_audit import run_audit
    run_audit(cfg, args.output, threads=args.threads, batch_size=args.batch_size,
              anchors=args.anchors_per_source, max_sources=args.max_sources)


if __name__ == '__main__':
    main()
