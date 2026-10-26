"""从新保存的完整真实在线轨迹重算新音乐目标MC，不接受旧稀疏价值快照。

加载原Stage1和冻结SM接口仅用于一致的FK/奖励计算；不启动物理求解器、不优化网络。
每个价值状态固定一次anchor对齐并覆盖剩余真实episode，保存明确的新奖励合同。
输出可交给fit_stage2_supermimic_mc.py重新拟合独立Critic。
"""
import argparse,json,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cuda:0")
    a=p.parse_args();sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from omegaconf import OmegaConf
    cfg=OmegaConf.to_container(OmegaConf.load(a.config),resolve=True);sys.path.insert(0,cfg["supermimic"]["repo"])
    from gem.closedloop.mb_control.application import build_components,read_tasks,seed_all
    from gem.closedloop.mb_control.music_mc import relabel_online_mc
    seed_all(cfg["training"]["seed"])
    actor,adapter,scheduler,rewards,critic,engine,original=build_components(cfg,a.device)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    relabel_online_mc(cfg,actor,engine,read_tasks(cfg["tasks"]),output,a.device)

if __name__=="__main__":main()
