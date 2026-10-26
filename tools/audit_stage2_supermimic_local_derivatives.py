"""对全部已保存真实接管点进行更局部的模型导数数值检查。

有限幅度物理扰动可能跨过神经网络分段边界，其弦斜率不必随每次缩小步长单调
接近零点导数。本工具另行预声明局部尺度，用原绝对/相对容差逐一检验实际
FP32计算图的autograd，要求连续两个局部尺度吻合；过小步长的舍入误差单独保留。
原始大尺度数值失败、所有物理方向与状态精度记录保留。
本工具不运行物理、不改变WM参数/状态支持、不单独授权正式训练。只有与同一批
全部真实物理配对合并时，局部数值检查才补充导数有效性证据，不能证明训练收益。
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--reports",nargs="+",required=True)
    parser.add_argument("--criteria",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    from omegaconf import OmegaConf
    cfg=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True)
    sys.path[:0]=[str(Path(__file__).resolve().parents[1]),cfg["supermimic"]["repo"]]
    import torch
    from gem.closedloop.mb_control.application import build_components,seed_all
    from gem.closedloop.mb_control.checkpoint import unpack_value
    from gem.closedloop.mb_control.short_audit import perturb_plan
    from gem.closedloop.mb_control.imagination import PhysicalAnchorImagination
    criteria=json.loads(Path(args.criteria).read_text())
    if (criteria["minimum_consecutive_scales"] < 2 or not all(0 < f <= 1 for f in criteria["scale_factors"]) or
            sorted(set(criteria["scale_factors"]),reverse=True)!=criteria["scale_factors"]):
        raise ValueError("必须预声明至少连续两个局部尺度，不能只挑一次匹配")
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    (output/"declared_criteria.json").write_text(json.dumps(criteria,indent=2))
    reports=[json.loads(Path(p).read_text()) for p in args.reports]
    seed_all(cfg["training"]["seed"])
    actor,adapter,scheduler,rewards,_,_,_=build_components(cfg,args.device)
    cases=[]
    for report in reports:
        if report["reward_contract"]!=cfg["rewards"] or report["frozen_sha256"]!=adapter.bundle.metadata["sha256"]:
            raise ValueError("局部数值测试必须使用同一真实物理奖励与冻结资产")
        if criteria["atol"]!=report["criteria"]["autograd_fd_atol"] or criteria["rtol"]!=report["criteria"]["autograd_fd_rtol"]:
            raise ValueError("保持原数值容差，不能通过改大容差绕过失败")
        for case in report["cases"]:
            sample=unpack_value(torch.load(case["artifact"],map_location="cpu",weights_only=False)["sample"],adapter,args.device)
            engine=PhysicalAnchorImagination(actor,adapter,scheduler,rewards,horizon_steps=32,gamma=cfg["critic"]["gamma_per_control"])
            results=[]
            source=case["horizons"]["32"]["directions"]
            for direction in ("root_x","root_z","leg","arm"):
                rows=[x for x in source if x["direction"]==direction]
                coordinate=rows[0]["coordinate"];origin=min(x["epsilon"] for x in rows)
                def objective(amplitude):
                    interval,_,_=engine.rollout_anchor(sample,plan_override=perturb_plan(sample.generation.plan,coordinate,amplitude))
                    return interval.reward.mean()
                amplitude=torch.zeros((),device=args.device,requires_grad=True)
                derivative,=torch.autograd.grad(objective(amplitude),amplitude)
                for factor in criteria["scale_factors"]:
                    eps=origin*factor
                    with torch.no_grad():minus=objective(-eps);plus=objective(eps)
                    fd=float((plus-minus)/(2*eps));error=abs(float(derivative)-fd)
                    results.append({"direction":direction,"epsilon":eps,"autograd":float(derivative),"centered_difference":fd,
                        "absolute_error":error,"passed":bool(torch.isfinite(derivative)) and error<=criteria["atol"]+criteria["rtol"]*abs(fd)})
            direction_passed={}
            for direction in ("root_x","root_z","leg","arm"):
                sequence=sorted([x for x in results if x["direction"]==direction],key=lambda x:-x["epsilon"])
                streak=longest=0
                for x in sequence:
                    streak=streak+1 if x["passed"] else 0;longest=max(longest,streak)
                direction_passed[direction]=longest>=criteria["minimum_consecutive_scales"]
            row={"task":case["task"],"anchor_tick":case["anchor_tick"],"artifact":case["artifact"],"rows":results,
                "direction_passed":direction_passed,"all_scales_passed":all(x["passed"] for x in results),
                "passed":all(direction_passed.values())}
            cases.append(row)
            with (output/"cases.jsonl").open("a") as f:f.write(json.dumps(row,allow_nan=False)+"\n")
            print(json.dumps({"task":row["task"],"anchor_tick":row["anchor_tick"],"passed":row["passed"]}),flush=True)
    result={"passed":bool(cases) and all(c["passed"] for c in cases),"cases":cases,"criteria":criteria,
        "source_reports":args.reports,"reward_contract":cfg["rewards"],"frozen_sha256":adapter.bundle.metadata["sha256"],
        "scope":"model_local_numerical_derivatives_only_not_physics_or_training_quality"}
    (output/"local_derivatives.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":main()
