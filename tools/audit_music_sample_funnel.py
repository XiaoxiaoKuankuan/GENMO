"""按真实在线请求逐级核对音乐舞蹈采样覆盖与原WM支持域拒绝。

读取冻结Stage1真实在线完整控制记录，包括提前失败episode；A统计所有实际请求，
B统计形成合法长度的真实anchor，再用原支持域检查H16/H32的预测状态并形成C。
速度、幅度、接触和类别只取真实状态。D/E读取实际候选更新日志，明确与正式长期
训练分布区分。脚本不推进Newton、不优化Actor、不扩大状态边界；输出具体拒绝分量
及首次预测时刻，未抽到或未成功更新的动作不冒充有效覆盖。
"""
import argparse,json,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True)
    p.add_argument("--output",required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--candidate-reports",nargs="*",default=[])
    args=p.parse_args();sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from omegaconf import OmegaConf
    cfg=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True);sys.path.insert(0,cfg["supermimic"]["repo"])
    import torch
    from gem.closedloop.mb_control.application import build_components,read_tasks
    from gem.closedloop.mb_control.music_repair_audit import online_episode,anchors_from_episode
    from gem.closedloop.mb_control.sample_distribution import describe_request,describe_sample,coverage_summary
    from gem.closedloop.mb_control.validation import ModelSupport,UnsupportedModelState,training_support_envelope
    actor,adapter,scheduler,rewards,_,engine,_=build_components(cfg,args.device)
    support_path=Path(cfg["imagination"]["physical_validation_report"])
    support=training_support_envelope(cfg,json.loads(support_path.read_text()))
    engine.support_guard=ModelSupport(support,device=args.device)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    all_rows=[];before=adapter.bundle.frozen_snapshot();parity=[];verified=set()
    for episode,task in enumerate(read_tasks(cfg["tasks"])):
        payload,_,controls,generations,_=online_episode(cfg,engine,task,args.device,allow_partial_for_real_mc=True)
        requests={g.request.ticket.request_tick:g.request for g in generations}
        for step in controls:
            pending=step.start.pending_request
            if pending is not None:requests.setdefault(pending.ticket.request_tick,pending)
        contacts={s.end.tick:s.event.get("contacts") for s in controls}
        rows=[]
        for tick,request in requests.items():
            row=describe_request(request.snapshot,adapter,0,episode)
            contact=contacts.get(tick)
            if contact:row["contact"]="double_support" if contact["left_foot"] and contact["right_foot"] else "single_support" if contact["left_foot"]!=contact["right_foot"] else "airborne"
            rows.append(row)
        for horizon in (16,32):
            engine.horizon_steps=horizon
            for sample in anchors_from_episode(controls,generations,horizon):
                row=describe_sample(sample,adapter);row.update(episode=episode,horizon=horizon,stage="B_formed_H_anchor")
                rows.append(dict(row))
                try:
                    with torch.no_grad():
                        fast,_,_=engine.rollout_anchor(sample,plan_override=sample.generation.plan,support_check_only=True)
                        if horizon not in verified:
                            full,_,_=engine.rollout_anchor(sample,plan_override=sample.generation.plan)
                            error=float((fast.end.state-full.end.state).abs().max())
                            if error!=0.:raise ValueError("纯支持域路径与完整奖励路径状态不同")
                            parity.append({"horizon":horizon,"state_max_abs":error});verified.add(horizon)
                    row.update(stage="B_legal_H_anchor",legal=True)
                except UnsupportedModelState as error:
                    row.update(stage="C_outside_support",legal=False,details=error.details)
                rows.append(row)
        all_rows.extend(rows)
        with (output/"funnel_rows.jsonl").open("a") as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
        print(json.dumps({"event":"real_request_funnel","task":task["id"],"requests":len(requests),
            "episode_complete":payload["complete"],"outside":sum(x["stage"]=="C_outside_support" for x in rows)}),flush=True)
    for path in map(Path,args.candidate_reports):
        for row in map(json.loads,path.read_text().splitlines()):
            for sample in row["sample_distribution"]:
                all_rows.append(dict(sample,stage="D_drawn",horizon=row["horizon_steps"],scope="independent_candidate_audit"))
                if sample["accepted"]:all_rows.append(dict(sample,stage="E_effective_update",horizon=row["horizon_steps"],scope="independent_candidate_audit"))
    adapter.bundle.assert_unchanged(before)
    result={"support_report":str(support_path),"support_not_relaxed":True,
        "support_check_vs_full_rollout_parity":parity,
        "horizons":{str(h):{stage:coverage_summary([r for r in all_rows if r["stage"]==stage and r.get("horizon",h)==h])
            for stage in ("A_all_real_requests","B_formed_H_anchor","B_legal_H_anchor","C_outside_support","D_drawn","E_effective_update")}
            for h in (16,32)},"D_E_scope":"independent_candidate_audit_not_formal_long_term_training"}
    (output/"sampling_funnel.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":main()
