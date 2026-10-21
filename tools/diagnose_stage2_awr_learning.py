"""只读定位AWR学习退化的梯度、去噪目标及参考消费信号，不更新任何训练状态。

读取已完成运行的不可变Replay、原始标签、Stage1及B/C断点，核对内容身份后分析：
一、加权梯度与普通自蒸馏梯度的方向、差值和最终参数变化；二、相同条件/噪声下
不同扩散时间的重建以及完整DDIM输出漂移；三、优势权重与真正暴露期间物理误差
的关系，以及暴露中有多少来自已知前缀。输入包括真实执行账本，但不启动PhysX、
不重新拟合Critic、不调用Optimizer.step，也不改原标签、模型或运行state。
报告只能解释该冻结样本的相关性和优化方向，不能把它当成新策略的因果效果实验。
使用原运行源码导入模型；诊断工具来自独立交付目录。输出必须是全新的独立目录。
新目标按运行配置应用固定时间/来源因子，并额外比较均匀时间对照；旧运行无这些
配置时保持旧公式。四组结果按实际checkpoint身份读取，不改正在运行的源码。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


def summary(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return dict(count=len(x), mean=float(x.mean()) if len(x) else None,
        p50=float(np.median(x)) if len(x) else None,
        p95=float(np.quantile(x, .95)) if len(x) else None,
        minimum=float(x.min()) if len(x) else None, maximum=float(x.max()) if len(x) else None)


def relation(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x)>1 and min(x.std(), y.std())>1e-12 else None
    return dict(count=len(x), correlation=corr, mean=float(y.mean()) if len(y) else None,
        weighted_mean=float(np.dot(x, y)/x.sum()) if len(x) and x.sum()>0 else None)


def vector_comparison(first, second):
    """分块FP64计算方向；second-first是额外加权信号，不等于物理收益贡献。"""
    aa = bb = ab = 0.
    for name in first.keys() | second.keys():
        a = first.get(name); b = second.get(name)
        if a is None: a = torch.zeros_like(b)
        if b is None: b = torch.zeros_like(a)
        a, b = a.reshape(-1), b.reshape(-1)
        for start in range(0, a.numel(), 2**20):
            u, v = a[start:start+2**20].double(), b[start:start+2**20].double()
            aa += float(u.square().sum()); bb += float(v.square().sum()); ab += float((u*v).sum())
    diff = max(0., aa+bb-2*ab)
    return dict(first_norm=math.sqrt(aa), second_norm=math.sqrt(bb),
        cosine=ab/math.sqrt(aa*bb) if aa*bb>0 else None,
        difference_norm=math.sqrt(diff), difference_to_first=math.sqrt(diff/aa) if aa else None,
        second_projection_on_first=ab/aa if aa else None,
        second_orthogonal_to_first_ratio=math.sqrt(max(0., bb-ab*ab/aa)/aa) if aa else None)


def reference_analysis(episodes, labels):
    from gem.closedloop.awr.reference_credit import reference_exposure
    weights = dict(zip(labels.sample_ids, labels.weights.tolist()))
    rows = []; delays = []; by_offset = np.zeros(12); copied = total = outside = 0.
    first_context = None
    for e in episodes:
        ex = reference_exposure(e, mode="current_plus_lookahead", current_weight=.5, lookahead_weight=.5)
        ledger = e.reference_ledger
        if not ex["valid"]: continue
        window = np.asarray(ledger["window_plan_indices"])
        ticks = np.asarray(ledger["consumed_reference_ticks"])
        errors = np.asarray(ledger["tracking_errors"])
        points = np.asarray(ex["point_weights"])
        for i, r in enumerate(e.rows):
            if not r.actor_valid: continue
            plan = ledger["plan_dictionary"].index(str(r.identity["plan_id"]))
            match = window == plan; control = (match*points).sum(axis=1)
            if control.sum() == 0: continue
            field = dict(zip(ledger["error_names"], (control[:, None]*errors).sum(axis=0)/control.sum()))
            future = np.rint(r.context["future_times"][0].double().numpy()*600).astype(np.int64)
            if not np.all(np.diff(future)==20): raise ValueError("Expected actual 30Hz target ticks")
            if first_context is None:
                first_context = dict(decision_tick=r.control_tick_begin,
                    decision_time=float(r.context["decision_time"].flatten()[0]), first_future_tick=int(future[0]))
            q = (ticks-future[0])/20.; left = np.floor(q).astype(np.int64); f = q-left
            valid = (q>=0) & (q<=len(future)-1)
            free = r.free_mask[:, 9:].double().mean(dim=1).numpy()
            interp = ((1-f)*free[np.clip(left, 0, len(free)-1)]+f*free[np.clip(left+1, 0, len(free)-1)])
            mass = match*points
            known_mass = float((mass*valid*(1-interp)).sum()); all_mass = float(mass.sum())
            outside_mass = float((mass*~valid).sum())
            copied += known_mass; total += all_mass; outside += outside_mass
            p = ex["plans"][i]
            if p["first_current_tick"] is not None:
                delays.append((p["first_current_tick"]-r.control_tick_begin)/600)
            for offset in range(min(12, len(e.rows)-i)):
                by_offset[offset] += float(ex["matrix"][i,i+offset])*e.rows[i+offset].executed_control_steps*.02
            rows.append(dict(sample_id=r.sample_id, source=e.source, group=e.group_id, weight=weights[r.sample_id],
                episode_progress=i/max(1,len(e.rows)-1), remaining_seconds=r.metadata["remaining_music_seconds"],
                current_consumed_seconds=p["actually_consumed_seconds"], exposure_seconds=p["weighted_exposure_seconds"],
                unknown_joint_frames=float(free.sum()), exposed_known_joint_fraction=known_mass/all_mass,
                **field))
    grouped = {}
    for source in sorted({r["source"] for r in rows}):
        subset = [r for r in rows if r["source"]==source]
        keys = ["root_position_error_m","body_position_rmse_m","yaw_error_rad","joint_position_rmse_rad",
                "episode_progress","remaining_seconds","exposed_known_joint_fraction"]
        grouped[source] = {k:relation([r["weight"] for r in subset],[r.get(k,np.nan) for r in subset]) for k in keys}
    return dict(first_context=first_context, first_current_delay_seconds=summary(delays),
        known_joint_fraction_of_weighted_exposure=copied/total, outside_target_fraction=outside/total,
        weighted_exposure_by_decision_offset=by_offset.tolist(), by_source=grouped,
        current_consumed_seconds=summary([r["current_consumed_seconds"] for r in rows]),
        unknown_joint_frames=summary([r["unknown_joint_frames"] for r in rows]),
        semantics="plan exposure includes copied prefix; corrected diagnostics average actual exposed controls")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runtime-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gradient-batches", type=int, default=4)
    parser.add_argument("--timestep-gradients", action="store_true",
                        help="在同一32样本上核对固定扩散时间的无权重梯度和低噪声方向")
    args=parser.parse_args()
    if not 1<=args.gradient_batches<=8: parser.error("Use a bounded 1..8 batch audit")
    root, repo, output=(p.resolve() for p in (args.run_dir,args.runtime_repo,args.output))
    if output.exists() or any(output.is_relative_to(root/x) for x in ("replay","checkpoints")):
        parser.error("Use a new independent output directory")
    before=(root/"state.json").read_bytes(); state=json.loads(before)
    if state["status"]!="complete": parser.error("Only completed immutable runs are supported")
    if subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()!=state["source_commits"]["genmo"]:
        parser.error("Runtime source commit differs")
    sys.path.insert(0,str(repo))
    from gem.closedloop.awr.io import read_tree, digest, atomic_json
    from gem.closedloop.awr.records import AWRCompleteEpisode, TRAINING_METADATA_KEYS
    from gem.closedloop.awr.advantages import AdvantageLabelSet
    from gem.closedloop.awr.dataset import ReplayDataset
    from gem.closedloop.awr.checkpoint import Checkpoints
    from gem.closedloop.awr.config import physical_config
    from gem.closedloop.awr.losses import per_sample_denoising
    from gem.closedloop.awr.metrics import actor_module_name
    from gem.closedloop.awr.sampler import AWRSampler
    from gem.closedloop.common.supervision import load_actor
    from gem.closedloop.common.evaluation import _model_fingerprint
    torch.set_num_threads(4); torch.manual_seed(9871)
    device=torch.device(args.device); started=time.perf_counter(); output.mkdir(parents=True)
    report=dict(schema="genmo.awr.learning_diagnosis.v1",source_run=str(root),
        runtime_sources=state["source_commits"],tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        optimizer_updates=0,physical_controls=0)
    def save(phase):
        report["phase"]=phase;report["seconds"]=time.perf_counter()-started
        atomic_json(output/"report.json",report);print(phase,round(report["seconds"],2),flush=True)
    snap=json.loads((root/"replay/snapshots"/(state["artifacts"]["macro000_replay"]+".json")).read_text())
    episodes=[]; entries=[]
    for key in snap["episode_ids"]:
        path=root/"replay/episodes"/key;manifest=json.loads((path/"manifest.json").read_text())
        e=AWRCompleteEpisode.from_dict(read_tree(path,episode_metadata_keys=TRAINING_METADATA_KEYS))
        if digest(e.contract)!=snap["contract_sha256"] or e.episode_id!=key: raise ValueError("Replay identity changed")
        episodes.append(e);entries.append([key,manifest["sha256"]])
    if digest(dict(contract=snap["contract_sha256"],episodes=entries))!=snap["snapshot_id"]: raise ValueError("Snapshot SHA changed")
    labels=AdvantageLabelSet.load(state["artifacts"]["macro000_labels"])
    data=ReplayDataset(episodes,labels)
    report["replay_snapshot"]=snap["snapshot_id"];report["reference"]=reference_analysis(episodes,labels)
    save("replay_and_reference")
    actor,_,_=load_actor(physical_config(state["config"],run_id=root.name,device=str(device)))
    actor.eval(); initial={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
    fingerprint=_model_fingerprint(actor); parameters={k:p for k,p in actor.named_parameters() if p.requires_grad}
    objective = state["config"]["awr"]["actor"]
    if "timestep_weighting" in objective or "source_balance" in objective:
        from gem.closedloop.awr.objective import timestep_weights, source_weights
        time_weights = timestep_weights(actor.train_diffusion.alphas_cumprod, objective.get("timestep_weighting"))
        source_factors = source_weights(data.sources, objective.get("source_balance", "none"))
    else:
        time_weights = torch.ones(actor.diffusion_steps, dtype=torch.float64)
        source_factors = torch.ones(len(data), dtype=torch.float64)
    report["training_objective"] = dict(bc_enabled=objective.get("bc_enabled", True),
        timestep_weighting=objective.get("timestep_weighting", {"mode":"uniform"}),
        source_balance=objective.get("source_balance", "none"),
        time_weights=time_weights.tolist(),
        source_factors={s:float(source_factors[data.sources.index(s)]) for s in sorted(set(data.sources))})
    def noise_for(ids,step):
        ts=[];noise=[]
        for i in ids:
            rng=torch.Generator().manual_seed(int(digest([state["config"]["awr"]["seed"],step,data.sample_ids[i]])[:15],16))
            ts.append(torch.randint(actor.diffusion_steps,(),generator=rng))
            noise.append(torch.randn(data.rows[i].target_qpos30.shape,generator=rng))
        return torch.stack(ts).to(device),torch.stack(noise).to(device)
    def gradients(ids,step,weighted,fixed_t=None,uniform_time=False):
        actor.zero_grad(set_to_none=True); total=0.
        for offset in range(0,len(ids),8):
            part=ids[offset:offset+8];b=data.batch(part,device);t,noise=noise_for(part,step)
            if fixed_t is not None: t=torch.full_like(t,fixed_t)
            out=actor.denoising_forward(b["conditions"],b["target"],b["target_valid"],timesteps=t,noise=noise,guidance_scale=2.5)
            losses=per_sample_denoising(out["pred_x_start"],out["target_x_start"],b["free_mask"])
            w=b["weights"] if weighted else torch.ones_like(b["weights"])
            w = w*source_factors[part].to(device)
            if not uniform_time:
                w = w*time_weights[t.detach().cpu()].to(device)
            loss=(losses.double()*w).sum()/len(ids);loss.backward();total+=float(loss.detach())
        return {n:p.grad.detach().cpu().clone() for n,p in parameters.items() if p.grad is not None},total
    order=torch.randperm(len(data),generator=torch.Generator().manual_seed(state["config"]["awr"]["seed"])).tolist()
    report["initial_gradient_comparisons"]=[]
    report["initial_time_objective_comparisons"]=[]
    for step in range(args.gradient_batches):
        ids=order[step*256:(step+1)*256]
        if not ids: break
        unweighted,lu=gradients(ids,step,False);weighted,lw=gradients(ids,step,True)
        entry=dict(step=step,samples=len(ids),unit_loss=lu,weighted_loss=lw,**vector_comparison(unweighted,weighted))
        entry["modules"]={m:vector_comparison({k:v for k,v in unweighted.items() if actor_module_name(k)==m},
            {k:v for k,v in weighted.items() if actor_module_name(k)==m}) for m in sorted({actor_module_name(k) for k in unweighted})}
        report["initial_gradient_comparisons"].append(entry)
        if objective.get("timestep_weighting", {}).get("mode", "uniform") != "uniform":
            uniform, loss = gradients(ids,step,False,uniform_time=True)
            report["initial_time_objective_comparisons"].append(dict(step=step,samples=len(ids),
                uniform_loss=loss,configured_loss=lu,**vector_comparison(uniform,unweighted)))
            del uniform
        del unweighted,weighted
        save("initial_gradients_"+str(step+1))
    actor.zero_grad(set_to_none=True)
    if _model_fingerprint(actor)!=fingerprint: raise ValueError("Gradient diagnosis mutated Actor parameters")
    probe=[]
    for source in sorted(set(data.sources)):
        ids=[i for i,s in enumerate(data.sources) if s==source]
        probe.extend(ids[int(i)] for i in np.linspace(0,len(ids)-1,8))
    first=[];seen=set()
    for e in episodes:
        if e.group_id not in seen:
            eligible=[r for r in e.rows if r.actor_valid]
            if eligible: first.append(data.sample_ids.index(eligible[0].sample_id));seen.add(e.group_id)
    report["probe_sample_ids"]=[data.sample_ids[i] for i in probe]
    report["generation_probe_sample_ids"]=[data.sample_ids[i] for i in first]
    if args.timestep_gradients:
        report["timestep_gradient_comparisons"]=[]
        clean=None
        for timestep in (0,100,300,600,900,999):
            grad,loss=gradients(probe,7000,False,fixed_t=timestep)
            if clean is None: clean=grad
            report["timestep_gradient_comparisons"].append(dict(timestep=timestep,loss=loss,
                **vector_comparison(clean,grad)))
            if timestep: del grad
            save("timestep_gradient_"+str(timestep))
        del clean;actor.zero_grad(set_to_none=True)
    checkpoint=Checkpoints(root/"checkpoints",dict(request=state["request_sha256"],sources=state["source_commits"]))
    methods = ["self_distillation", "awr"]
    if "macro000_self_distillation_uniform_final" in state["artifacts"]:
        methods.insert(0, "self_distillation_uniform")
    payloads={m:checkpoint.load(state["artifacts"]["macro000_"+m+"_final"])["payload"] for m in methods}
    delta={m:{k:p["actor"][k]-initial[k] for k in parameters} for m,p in payloads.items()}
    report["parameter_delta_comparison"]=vector_comparison(delta["self_distillation"],delta["awr"])
    if "self_distillation_uniform" in delta:
        report["time_objective_parameter_delta_comparison"] = vector_comparison(
            delta["self_distillation_uniform"],delta["self_distillation"])
    del delta;gc.collect()
    report["probes"]={};baseline_output=None
    for name,weights in [("stage1",initial),*((m,p["actor"]) for m,p in payloads.items())]:
        actor.load_state_dict(weights);actor.eval();results=[]
        with torch.no_grad():
            for timestep in (0,100,300,600,900,999):
                values={k:[] for k in ("guided_mse","conditional_mse","unconditional_mse","joint_physical_rmse_rad")}
                for offset in range(0,len(probe),8):
                    ids=probe[offset:offset+8];b=data.batch(ids,device);_,noise=noise_for(ids,7000)
                    t=torch.full((len(ids),),timestep,dtype=torch.long,device=device)
                    out=actor.denoising_forward(b["conditions"],b["target"],b["target_valid"],timesteps=t,noise=noise,guidance_scale=2.5)
                    for key,pred in [("guided_mse","pred_x_start"),("conditional_mse","conditional_x_start"),("unconditional_mse","unconditional_x_start")]:
                        values[key].extend(per_sample_denoising(out[pred],out["target_x_start"],b["free_mask"]).cpu().tolist())
                    error=(actor.endecoder.denormalize(out["pred_x_start"])-b["target"])[...,9:]
                    mask=b["free_mask"][...,9:]
                    values["joint_physical_rmse_rad"].extend((torch.where(mask,error,0.).square().flatten(1).sum(1)/mask.flatten(1).sum(1)).sqrt().cpu().tolist())
                results.append(dict(timestep=timestep,**{k:summary(v) for k,v in values.items()}))
            generated=[]
            sampler=AWRSampler(actor)
            for offset in range(0,len(first),8):
                ids=first[offset:offset+8];b=data.batch(ids,device)
                generated.append(sampler.sample(b["conditions"],seeds=[90000+i for i in ids])["qpos30"].cpu())
            generated=torch.cat(generated)
            if baseline_output is None: baseline_output=generated.clone()
            mask=torch.stack([data.rows[i].free_mask for i in first])[...,9:]
            change=(generated-baseline_output)[...,9:]
            per_joint=[]
            for j,joint in enumerate(actor.endecoder.kinematics.joint_order):
                x=change[...,j][mask[...,j]]
                per_joint.append(dict(joint=joint,mean_delta_rad=float(x.mean()),rms_delta_rad=float(x.square().mean().sqrt())))
            report["probes"][name]=dict(actor_fingerprint=_model_fingerprint(actor),denoising=results,
                generated_joint_rms_delta_rad=float(change[mask].square().mean().sqrt()),per_joint=per_joint)
        save("probe_"+name)
    if (root/"state.json").read_bytes()!=before:raise ValueError("Run state changed during read-only diagnosis")
    report["state_unchanged"]=True
    report["peak_torch_memory_bytes"]=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0
    report["scope"]="frozen_replay_gradient_and_generation_diagnosis_not_new_learning_or_physics"
    save("complete")


if __name__=="__main__":
    main()
