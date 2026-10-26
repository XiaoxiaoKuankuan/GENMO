"""统计真实回放候选的运动分布，以及在原状态支持边界下的模型滚动合法性。

输入完整训练检查点或真实物理审计保存的anchor，速度、幅度和接触均取真实记录。
使用各样本自己的实际行为计划重放WM，保持原H及原支持边界，不更新参数、不扩域。
旧记录缺失的来源按音乐张量精确匹配原任务清单，缺失接触标记unknown，绝不猜测。
这些是候选池的重新检查，不冒充历史优化器实际抽样/拒绝明细，也不是物理收益。
"""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--report",required=True)
    p.add_argument("--checkpoint");p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args()
    from omegaconf import OmegaConf
    cfg=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    sys.path[:0]=[str(Path(__file__).resolve().parents[1]),cfg["supermimic"]["repo"]]
    import torch
    from gem.closedloop.mb_control.application import build_components,read_tasks,read_music_features,seed_all
    from gem.closedloop.mb_control.checkpoint import unpack_value
    from gem.closedloop.mb_control.teacher import tensor_digest
    from gem.closedloop.mb_control.sample_distribution import describe_sample
    from gem.closedloop.mb_control.validation import ModelSupport,UnsupportedModelState
    report=json.loads(Path(args.report).read_text());seed_all(cfg["training"]["seed"])
    actor,adapter,_,_,_,engine,_=build_components(cfg,args.device)
    engine.support_guard=ModelSupport(report["validated_state_envelope"],device=args.device)
    if args.checkpoint:
        payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False,mmap=True)
        raw=payload["replay"]["items"];cursor=payload["runtime_state"]["task_cursor"]
        tasks=read_tasks(cfg["tasks"])[:cursor]
    else:
        raw=[torch.load(c["artifact"],map_location="cpu",weights_only=False)["sample"] for c in report["cases"]]
        tasks=[]
    music_ids={tensor_digest(read_music_features(t["music_features"])[None]):t for t in tasks}
    rows=[];memo={}
    for i,value in enumerate(raw):
        sample=unpack_value(value,adapter,args.device,memo)
        if not sample.state.task_id:
            task=music_ids.get(tensor_digest(sample.state.music))
            if task:sample=replace(sample,state=replace(sample.state,task_id=task["id"],task_source=task["source"]))
        row=describe_sample(sample,adapter)
        row["actor_version"]=sample.generation.request.actor_version
        try:
            with torch.no_grad():engine.rollout_anchor(sample,plan_override=sample.generation.plan)
            row.update(legal=True,rejection=None)
        except UnsupportedModelState as error:
            row.update(legal=False,rejection=str(error))
        rows.append(row)
        print(json.dumps({"sample":i,"source":row["source"],"speed":row["speed_bin"],"legal":row["legal"]}),flush=True)
    counts=Counter((r["legal"],r["source"],r["speed_bin"],r["amplitude_bin"],r["contact"],r["motion_class"]) for r in rows)
    result={"horizon":engine.horizon_steps,"source_checkpoint":args.checkpoint,"support_report":args.report,
        "rows":rows,"counts":[{"legal":k[0],"source":k[1],"speed":k[2],"amplitude":k[3],"contact":k[4],"class":k[5],"count":v} for k,v in counts.items()],
        "not_historical_optimizer_selection_log":True,"state_support_modified":False}
    with Path(args.output).open("x") as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)


if __name__=="__main__":main()
