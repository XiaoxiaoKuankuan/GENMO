"""第二阶段音乐漂移的独立诊断入口。

offline模式只读取已有完整真实轨迹、参考版本与优化日志，重算三种对齐目标、
实际H16新运动覆盖和采样分布；不会初始化模拟器或改动正式训练。
physical模式沿用现有Isaac入口，在独立进程、目录和模型副本上做候选/小实验。
所有数值结果区分历史数据重算与新Newton执行，不把梯度非零当作收益证明。
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if __name__ == '__main__':
    if '--offline' in sys.argv:
        p=argparse.ArgumentParser(description=__doc__)
        p.add_argument('--offline',action='store_true');p.add_argument('--run',required=True)
        p.add_argument('--output',required=True);p.add_argument('--versions',nargs='+',type=int,default=[0,100,300,600,700])
        p.add_argument('--evaluation-folders',help='显式版本到完整真实评价目录的JSON映射，独立新实验不借用旧版本轨迹')
        p.add_argument('--skip-log-audit',action='store_true')
        args=p.parse_args()
        import yaml
        cfg=yaml.safe_load((Path(args.run)/'config.yaml').read_text())
        sys.path.insert(0,cfg['supermimic']['repo'])
        from gem.closedloop.mb_control.drift_diagnostics import offline_audit
        import json
        folders=json.loads(Path(args.evaluation_folders).read_text()) if args.evaluation_folders else None
        offline_audit(cfg,Path(args.run),Path(args.output),args.versions,
            evaluation_folders=folders,audit_training_logs=not args.skip_log_audit)
    else:
        from gem.closedloop.mb_control.entry import main
        main('drift')
