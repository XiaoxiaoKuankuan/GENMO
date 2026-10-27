"""独立漂移实验的完整八曲评价协调进程。

读取已原子保存的完整Actor检查点，调用原Newton多GPU评价器；不访问训练中的参数，
不写训练优化器，不根据质量结果向训练发停止信号。父进程可继续真实采集和更新，
本进程仅输出固定歌曲、初态、噪声与延迟事件下的完整物理评价或真实失败日志。
"""
import argparse
import os
from pathlib import Path
from omegaconf import OmegaConf


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();cfg=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    import sys
    sys.path.insert(0,cfg['supermimic']['repo'])
    from .formal import run_evaluation_processes
    run_evaluation_processes(cfg,args.config,Path(args.checkpoint),Path(args.output),'full',os.environ.copy())


if __name__=='__main__':main()
